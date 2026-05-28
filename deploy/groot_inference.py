import os
import time
import threading
import cv2
import numpy as np
import dataclasses
import logging
import pathlib
import zmq
import tyro
from typing import Optional
from multiprocessing import Array, Event

from teleop.master_whole_body import RobotTaskmaster
from teleop.robot_control.compute_tau import GetTauer
from groot_policy.server_client import PolicyClient
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    """Command line arguments."""

    # Host and port to connect to the server.
    host: str = "127.0.0.1"
    # Port to connect to the server. If None, the server will use the default port.
    port: Optional[int] = 8014

    api_key: Optional[str] = None
    # Number of steps to run the policy for.
    num_steps: int = 20
    # Path to save the timings to a parquet file. (e.g., timing.parquet)
    timing_file: Optional[pathlib.Path] = None
    # Environment to run the policy in.
    # env: EnvMode = EnvMode.ALOHA_SIM

args = tyro.cli(Args)
policy = PolicyClient(host=args.host, port=args.port)
VIDEO_KEY = "rs_view"

TASK_INSTRUCTION = "Push cart grasp and place grapes on plate."

FREQ_VLA = 30     
FREQ_CTRL = 60  
MAX_STEPS = 500

ACTION_REPEAT = max(1, int(round(FREQ_CTRL / FREQ_VLA)))


class RSCamera:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://192.168.123.164:5556")

    def get_frame(self):
        self.socket.send(b"get_frame")

        rgb_bytes, _, _ = self.socket.recv_multipart()

        rgb_array = np.frombuffer(rgb_bytes, np.uint8)
        rgb_image = cv2.imdecode(rgb_array, cv2.IMREAD_COLOR)
        return rgb_image



def get_observation(camera, state):
    frame = camera.get_frame()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height = float(state["height"][0]) if isinstance(state["height"], np.ndarray) else float(
        state["height"]
    )
    rpy = np.asarray(state["rpy"], dtype=np.float32)
    left_arm = np.asarray(state["left_arm"], dtype=np.float32)
    right_arm = np.asarray(state["right_arm"], dtype=np.float32)
    left_hand = np.asarray(state["left_hand"], dtype=np.float32)
    right_hand = np.asarray(state["right_hand"], dtype=np.float32)
    obs = {
        "video": {
            VIDEO_KEY: frame[None, None, :, :, :].astype(np.uint8),
        },
        "state": {
            "height": np.array([[[height]]], dtype=np.float32),
            "left_arm": left_arm[None, None, :],
            "right_arm": right_arm[None, None, :],
            "left_hand": left_hand[None, None, :],
            "right_hand": right_hand[None, None, :],
            "rpy": rpy[None, None, :],
        },
        "language": {
            "annotation.human.task_description": [[TASK_INSTRUCTION]],
        },
    }
    return obs


def main():
    shared_data = {
        "kill_event": Event(),
        "session_start_event": Event(),
        "failure_event": Event(),
        "end_event": Event(),
        "dirname": None,
    }
    kill_event = shared_data["kill_event"]

    robot_shm_array = Array("d", 512, lock=False)
    teleop_shm_array = Array("d", 64, lock=False)

    

    master = RobotTaskmaster(
        task_name="inference",
        shared_data=shared_data,
        robot_shm_array=robot_shm_array,
        teleop_shm_array=teleop_shm_array,
        robot="g1",
    )

    master.reset_yaw_offset = True

    get_tauer = GetTauer()
    camera = RSCamera()

    pred_action_buffer = {"actions": None, "idx": 0}
    pred_action_lock = threading.Lock()
    state_lock = threading.Lock()
    shared_robot_state = {
        "motor": None,
        "hand": None,
    }


    running = Event()
    running.set()

    sequence_done_event = Event()
    sequence_done_event.set() 

    def action_request_thread():
        for step in range(MAX_STEPS):
            if not running.is_set():
                break

            sequence_done_event.wait()

            time.sleep(1/FREQ_VLA)

            try:
                with state_lock:
                    motor = shared_robot_state["motor"].copy() if shared_robot_state["motor"] is not None else None
                    hand = shared_robot_state["hand"].copy() if shared_robot_state["hand"] is not None else None

                if motor is None or hand is None:
                    print("[VLA] Waiting for robot state...")
                    time.sleep(0.01)
                    continue

                arm_joints = motor[15:29]
                hand_joints = hand
                leg_joints = motor[:15]

                # HTTP obs payload
                state = {
                    "rpy": np.array([
                        master.torso_roll,
                        master.torso_pitch,
                        master.torso_yaw,
                    ], dtype=np.float32),
                    "height": np.array([master.torso_height], dtype=np.float32),
                    "left_arm": arm_joints[0:7],
                    "right_arm": arm_joints[7:14],
                    "left_hand": hand_joints[0:7],
                    "right_hand": hand_joints[7:14],
                }
                obs = get_observation(camera, state)
                action_dict, _ = policy.get_action(obs)

                keys = [
                    "left_hand",
                    "right_hand",
                    "left_arm",
                    "right_arm",
                    "rpy",
                    "height",
                    "torso_vx",
                    "torso_vy",
                    "torso_vyaw",
                    "target_yaw",
                ]
                actions = np.concatenate([action_dict[k][0] for k in keys], axis=1)
                assert actions.shape == (16, 36), (
                    f"expecting actions.shape = (16, 36), found {actions.shape}"
                )

                
                with pred_action_lock:
                    pred_action_buffer["actions"] = actions
                    pred_action_buffer["idx"] = 0

                print(f"[VLA] Got action sequence: {len(actions)} actions")

                sequence_done_event.clear()

            except Exception as e:
                print(f"[VLA] Inference error: {e}")
                time.sleep(0.05)


    def apply_action_from_buffer():
        current_lr_arm_q, current_lr_arm_dq = master.get_robot_data()
        with state_lock:
            shared_robot_state["motor"] = master.motorstate.copy()
            shared_robot_state["hand"] = master.handstate.copy()

        with pred_action_lock:
            actions = pred_action_buffer["actions"]
            idx = pred_action_buffer["idx"]

            action = None
            not_between_rollouts = False

            if actions is not None:
                real_idx = idx // ACTION_REPEAT
                if real_idx < len(actions):
                    action = actions[real_idx]
                    not_between_rollouts = True

                    pred_action_buffer["idx"] += 1

                    next_real_idx = pred_action_buffer["idx"] // ACTION_REPEAT
                    if next_real_idx >= len(actions):
                        pred_action_buffer["actions"] = None
                        pred_action_buffer["idx"] = 0
                        sequence_done_event.set()
                else:
                    pred_action_buffer["actions"] = None
                    pred_action_buffer["idx"] = 0
                    sequence_done_event.set()

        arm_cmd = None
        hand_cmd = None
        if not_between_rollouts:
            if action.shape[0] < 36:
                print("[CTRL] Invalid action shape:", action.shape)
            else:
                vx = action[32]
                vy = action[33]
                vyaw = action[34]
                target_yaw = action[35]

                vx = 0.35 if vx > 0.25 else 0
                vy = 0 if abs(vy) < 0.3 else 0.5 * (1 if vy > 0 else -1)

                rpyh   = action[28:32]
                arm_cmd = action[14:28]
                hand_cmd = action[0:14]

                master.torso_roll   = rpyh[0]
                master.torso_pitch  = rpyh[1]
                master.torso_yaw    = rpyh[2]
                master.torso_height = rpyh[3]

                master.vx = vx
                master.vy = vy
                master.vyaw = vyaw
                master.target_yaw = target_yaw


                master.prev_torso_roll   = master.torso_roll
                master.prev_torso_pitch  = master.torso_pitch
                master.prev_torso_yaw    = master.torso_yaw
                master.prev_torso_height = master.torso_height

                master.prev_vx   = master.vx
                master.prev_vy  = master.vy
                master.prev_vyaw    = master.vyaw
                master.prev_target_yaw = master.target_yaw

                master.prev_arm = arm_cmd
                master.prev_hand = hand_cmd

                print("VLA output vx, vy, vyaw, target_yaw, rpyh:", vx, vy, vyaw, target_yaw, rpyh)
        
        if not not_between_rollouts:
            master.torso_roll   = master.prev_torso_roll
            master.torso_pitch  = master.prev_torso_pitch
            master.torso_yaw    = master.prev_torso_yaw
            master.torso_height = master.prev_torso_height

            arm_cmd = master.prev_arm
            hand_cmd = master.prev_hand

            master.vx = master.prev_vx
            master.vy = 0
            master.vyaw = master.prev_vyaw
            master.target_yaw = master.prev_target_yaw
        

        master.get_ik_observation(record=False)


        pd_target, pd_tauff, raw_action = master.body_ik.solve_whole_body_ik(
            left_wrist=None,
            right_wrist=None,
            current_lr_arm_q=current_lr_arm_q,
            current_lr_arm_dq=current_lr_arm_dq,
            observation=master.observation,
            extra_hist=master.extra_hist,
            is_teleop=False,
        )

        master.last_action = np.concatenate([
            raw_action.copy(),
            (master.motorstate - master.default_dof_pos)[15:] / master.action_scale,
        ])


        if arm_cmd is not None:
            pd_target[15:] = arm_cmd
            tau_arm = np.asarray(get_tauer(arm_cmd), dtype=np.float64).reshape(-1)
            pd_tauff[15:] = tau_arm

        if hand_cmd is not None:
            with master.dual_hand_data_lock:
                master.hand_shm_array[:] = hand_cmd

        master.body_ctrl.ctrl_whole_body(
            pd_target[15:], pd_tauff[15:], pd_target[:15], pd_tauff[:15]
        )

        return pd_target
    


    def control_loop_thread():
        dt = 1.0 / FREQ_CTRL
        while running.is_set() and not kill_event.is_set():
            try:
                apply_action_from_buffer()
            except Exception as e:
                print("[CTRL] loop error:", e)
            time.sleep(dt) 
        print("[CTRL] Control loop stopped.")

    try:
        stabilize_thread = threading.Thread(target=master.maintain_standing, daemon=True)
        stabilize_thread.start()
        master.episode_kill_event.set()
        print("[MAIN] Initialize with standing pose...")
        time.sleep(30)
        master.episode_kill_event.clear()  

        master.reset_yaw_offset = True

        t_req = threading.Thread(target=action_request_thread, daemon=True)
        t_ctrl = threading.Thread(target=control_loop_thread, daemon=True)
        t_req.start()
        t_ctrl.start()

        print("[MAIN] Running. Ctrl+C to stop.")
        while not kill_event.is_set():
            time.sleep(0.5)

        print("[MAIN] kill_event set, preparing to stop...")
        running.clear()
        time.sleep(0.5)

        master.episode_kill_event.set()
        print("[MAIN] Returning to standing pose for 5s...")
        time.sleep(5)
        master.episode_kill_event.clear()

    except KeyboardInterrupt:
        print("[MAIN] Caught Ctrl+C, exiting...")
        running.clear()
        kill_event.set()
    finally:
        shared_data["end_event"].set()
        master.stop()
        print("[MAIN] Shutdown complete.")

if __name__ == "__main__":
    main()
