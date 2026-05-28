# -*- coding: utf-8 -*-
"""
Astribot inference client over Psi RTC WebSocket.

WebSocket counterpart of the gRPC client. The transport-specific bits live
in `PsiWebSocketClient`; everything else (ROS camera subscription, Astribot
SDK control, action filter/smoother/velocity limit, inference logging,
move-to-ready, initial transition) is preserved.

The Psi RTC server runs its own 30 Hz control loop and pushes one action
per tick over the WebSocket. The client only:
  (a) pushes obs (state + images + instruction) as fast as it can
  (b) reads `latest_action` and applies it to the robot at `--control-freq` Hz

Server side: src/psi/deploy/psi_serve_rtc-trainingtimertc_lyx.py

Example:
    # Server (in Psi training container)
    python src/psi/deploy/psi_serve_rtc-trainingtimertc_lyx.py \\
        --host 0.0.0.0 --port 8014 --action_exec_horizon 16 \\
        --policy psi0 --rtc true \\
        --run-dir=$RUN_DIR --ckpt-step=40000

    # Client (on robot side)
    python astribot_inference_ws.py \\
        --server ws://<psi-host>:8014/ws \\
        --task "Use the right arm to pick up the building block from the table" \\
        --state-with-chassis \\
        --enable-camera --cameras head,wrist_left,wrist_right,torso \\
        --no-head --no-torso       # first run: only arms+grippers
"""

import os
import sys
import json
import time
import signal
import logging
import threading
from base64 import b64encode, b64decode
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

# WebSocket transport
from websocket import WebSocketApp
from numpy.lib.format import dtype_to_descr, descr_to_dtype

# Project deps (same paths as the original gRPC client)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.common.config import ClientConfig, ActionConfig
from src.common.utils import (
    setup_logging,
    ActionSmoother,
    VelocityLimiter,
    lerobot_action_to_waypoint,
    waypoint_to_lerobot_action,
    filter_action,
)
from src.common.constants import (
    ASTRIBOT_NAMES_LIST,
    ASTRIBOT_NAMES_LIST_WITH_CHASSIS,
    LEROBOT_ACTION_DIM_NO_CHASSIS,
    LEROBOT_ACTION_DIM_WITH_CHASSIS,
    READY_POSITION_22,
    READY_POSITION_25,
)
from src.client.inference_logger import InferenceLogger

# Astribot SDK (optional — falls back to simulation if missing)
HAS_ASTRIBOT = False
try:
    from core.astribot_api.astribot_client import Astribot
    HAS_ASTRIBOT = True
except ImportError:
    pass

# ROS camera (optional)
HAS_ROS = False
try:
    import rospy
    from sensor_msgs.msg import CompressedImage
    HAS_ROS = True
except ImportError:
    pass

# OpenCV (required only when --enable-camera is on, to decode JPEG)
HAS_CV2 = False
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    pass

logger = logging.getLogger("astribot_inference_ws")


# ============================================================================
# Camera config — matches Psi0 finetune image_keys = [head, hand_left,
# hand_right, torso]. Note: original gRPC client did NOT include 'torso';
# we add it back here because the model was trained with 4 cameras.
# ============================================================================
ASTRIBOT_IMAGE_TOPICS = {
    "/astribot_camera/head_rgbd/color_compress/compressed":        "head",
    "/astribot_camera/left_wrist_rgbd/color_compress/compressed":  "wrist_left",
    "/astribot_camera/right_wrist_rgbd/color_compress/compressed": "wrist_right",
    "/astribot_camera/torso_rgbd/color_compress/compressed":       "torso",
}

# ROS topic-name -> training-side image-key suffix.
ASTRIBOT_CAMERA_NAME_MAPPING = {
    "head":        "head",
    "wrist_left":  "hand_left",
    "wrist_right": "hand_right",
    "torso":       "torso",
}

PSI_IMAGE_KEY_PREFIX = "observation.images."
PSI_STATE_KEY = "observation.state"


# ============================================================================
# Camera subscriber — returns decoded RGB uint8 arrays keyed by the training
# image_keys (e.g. "observation.images.head"). The Psi server's RequestMessage
# expects numpy arrays, not JPEG bytes, so we decode here.
# ============================================================================
class AstribotCameraSubscriber:
    def __init__(self, camera_names: Optional[List[str]] = None):
        if not HAS_ROS:
            raise ImportError("rospy / sensor_msgs not available")
        if not HAS_CV2:
            raise ImportError("opencv-python (cv2) not available — needed to decode JPEG")
        self.camera_names = camera_names or list(ASTRIBOT_CAMERA_NAME_MAPPING.keys())
        self._raw: Dict[str, bytes] = {}
        self._subs: List[Any] = []

    def start(self, init_node: bool = True):
        if init_node:
            try:
                rospy.init_node("astribot_psi_ws_subscriber", anonymous=True)
            except rospy.exceptions.ROSException:
                pass
        for topic, cam_name in ASTRIBOT_IMAGE_TOPICS.items():
            if cam_name in self.camera_names:
                sub = rospy.Subscriber(
                    topic, CompressedImage,
                    self._callback, callback_args=cam_name, queue_size=1,
                )
                self._subs.append(sub)
                logger.info(f"Subscribed: {cam_name} <- {topic}")

    def _callback(self, msg, cam_name):
        self._raw[cam_name] = bytes(msg.data)

    def get_decoded_images(self) -> Dict[str, np.ndarray]:
        """Return {image_key: HxWx3 uint8 RGB ndarray}, ready to ship to the
        Psi server. Decoded JPEG -> BGR -> RGB."""
        out: Dict[str, np.ndarray] = {}
        for cam_name, jpeg in self._raw.items():
            if not jpeg:
                continue
            mapped = ASTRIBOT_CAMERA_NAME_MAPPING.get(cam_name, cam_name)
            key = PSI_IMAGE_KEY_PREFIX + mapped
            bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            out[key] = rgb.astype(np.uint8)
        return out

    def wait_for_images(self, timeout: float = 5.0) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            if all(cam in self._raw for cam in self.camera_names):
                return True
            time.sleep(0.1)
        return False

    def stop(self):
        for sub in self._subs:
            sub.unregister()
        self._subs = []
        self._raw = {}


# ============================================================================
# WebSocket transport — must match the wire format used by
# src/psi/deploy/helpers.py {RequestMessage, ResponseMessage}
# ============================================================================
def _np_serialize(o):
    if isinstance(o, (np.ndarray, np.generic)):
        data = o.data if o.flags.get("C_CONTIGUOUS", False) else o.tobytes()
        return {
            "__numpy__": b64encode(data).decode(),
            "dtype": dtype_to_descr(o.dtype),
            "shape": list(o.shape) if hasattr(o, "shape") else [],
        }
    raise TypeError(f"not serializable: {type(o).__name__}")


def _np_deserialize(dct):
    if "__numpy__" in dct:
        arr = np.frombuffer(b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"]))
        shape = dct["shape"]
        return arr.reshape(shape) if shape else arr[0]
    return dct


def _convert_numpy(obj, func):
    if isinstance(obj, dict):
        if "__numpy__" in obj:
            return func(obj)
        return {k: _convert_numpy(v, func) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy(v, func) for v in obj]
    if isinstance(obj, (np.ndarray, np.generic)):
        return func(obj)
    return obj


# Global interrupt flag for clean Ctrl-C handling
_interrupted = False


def _signal_handler(signum, frame):
    global _interrupted
    _interrupted = True
    logger.warning(f"Got signal {signum}")


signal.signal(signal.SIGINT, _signal_handler)


class PsiWebSocketClient:
    """WebSocket client for the Psi RTC server.

    Background structure:
      * `_ws_thread`     runs WebSocketApp.run_forever (events fire on this thread)
      * `_send_thread`   pulls the latest obs and pushes it to the server
      * `_on_message`    updates `_latest_action` whenever the server pushes

    The caller does:
        client.set_obs(payload)          # produce new obs (any rate)
        action, ver, is_new = client.get_latest_action()  # consume latest action
    """

    def __init__(self, server_url: str, send_interval_sec: float = 0.005):
        self.server_url = server_url
        self.send_interval = send_interval_sec

        self._ws: Optional[WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._send_thread: Optional[threading.Thread] = None
        self._connected = threading.Event()
        self._running = threading.Event()

        self._action_lock = threading.Lock()
        self._latest_action: Optional[np.ndarray] = None
        self._latest_version: int = -1
        self._consumed_version: int = -1

        self._obs_lock = threading.Lock()
        self._latest_payload: Optional[Dict[str, Any]] = None

        self._n_send = 0
        self._n_recv = 0

    # ---- obs / action API ----
    def set_obs(self, payload: Dict[str, Any]):
        with self._obs_lock:
            self._latest_payload = payload

    def get_latest_action(self) -> Tuple[Optional[np.ndarray], int, bool]:
        """Return (action_array, version, is_new).

        action_array is shape (Da,) — the server returns (1, Da) and we
        flatten the leading singleton dim. `is_new` is True when the version
        has advanced since the last call.
        """
        with self._action_lock:
            if self._latest_action is None:
                return None, -1, False
            arr = self._latest_action.copy()
            ver = self._latest_version
            is_new = ver != self._consumed_version
            self._consumed_version = ver
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        return arr, ver, is_new

    # ---- WebSocket callbacks ----
    def _on_open(self, ws):
        logger.info("[ws] connected")
        self._connected.set()

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            action_raw = data.get("action")
            version = int(data.get("version", -1))
            if action_raw is None:
                return
            action = _convert_numpy(action_raw, _np_deserialize)
            if isinstance(action, np.ndarray):
                with self._action_lock:
                    self._latest_action = action
                    self._latest_version = version
                self._n_recv += 1
        except Exception as e:
            logger.error(f"[ws] on_message error: {e}")

    def _on_error(self, ws, error):
        logger.error(f"[ws] error: {error}")

    def _on_close(self, ws, code, msg):
        logger.warning(f"[ws] closed: {code} - {msg}")
        self._running.clear()

    # ---- threads ----
    def _send_loop(self):
        self._connected.wait()
        while self._running.is_set():
            with self._obs_lock:
                payload = self._latest_payload
            if payload is None:
                time.sleep(0.005)
                continue
            try:
                msg = json.dumps(_convert_numpy(payload, _np_serialize))
                if self._ws and self._ws.sock and self._ws.sock.connected:
                    self._ws.send(msg)
                    self._n_send += 1
                else:
                    logger.error("[ws] socket not connected; exiting send loop")
                    break
            except Exception as e:
                logger.error(f"[ws] send error: {e}")
                break
            if self.send_interval > 0:
                time.sleep(self.send_interval)
        logger.info("[ws] send loop exited")

    def start(self, connect_timeout: float = 10.0):
        self._running.set()
        self._ws = WebSocketApp(
            self.server_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws_thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._ws_thread.start()
        if not self._connected.wait(timeout=connect_timeout):
            raise ConnectionError(f"could not connect to {self.server_url} within {connect_timeout}s")
        self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._send_thread.start()

    def close(self):
        self._running.clear()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._send_thread:
            self._send_thread.join(timeout=2.0)
        if self._ws_thread:
            self._ws_thread.join(timeout=2.0)


# ============================================================================
# AstribotController (WebSocket variant)
# ============================================================================
class AstribotController:
    def __init__(
        self,
        config: ClientConfig,
        server_url: str,
        task_instruction: str,
        enable_camera: bool = False,
        camera_names: Optional[List[str]] = None,
        inference_logger: Optional[InferenceLogger] = None,
    ):
        self.config = config
        self.task_instruction = task_instruction
        self.inference_logger = inference_logger

        logger.info("Initializing AstribotController (WebSocket variant)")
        logger.info(f"  - server:        {server_url}")
        logger.info(f"  - task:          {task_instruction}")
        logger.info(f"  - control_freq:  {config.control_freq} Hz")
        logger.info(f"  - logging:       {'on' if inference_logger else 'off'}")

        # ---- transport ----
        self.client = PsiWebSocketClient(server_url, send_interval_sec=0.005)
        self.client.start(connect_timeout=15.0)

        # ---- control config ----
        self.control_freq = config.control_freq
        self.control_period = 1.0 / config.control_freq
        self.control_way = config.control_way

        # ---- action pipeline ----
        self.smoother = ActionSmoother(window_size=config.smooth_window) if config.smooth_window > 0 else None
        self.velocity_limiter = VelocityLimiter(max_delta=config.max_velocity) if config.max_velocity > 0 else None
        if self.smoother:
            logger.info(f"  - smooth window: {config.smooth_window}")
        if self.velocity_limiter:
            logger.info(f"  - max velocity:  {config.max_velocity}")

        # ---- SDK ----
        self.astribot = None
        if HAS_ASTRIBOT:
            self.astribot = Astribot(freq=config.control_freq)
            logger.info("Astribot SDK initialized")
        else:
            logger.warning("Astribot SDK not available; running in simulation mode")

        # ---- camera ----
        self._enable_camera = enable_camera
        self.camera_subscriber = None
        if enable_camera:
            if not HAS_ROS or not HAS_CV2:
                raise RuntimeError("--enable-camera requires both rospy and opencv-python")
            self.camera_subscriber = AstribotCameraSubscriber(
                camera_names or list(ASTRIBOT_CAMERA_NAME_MAPPING.keys())
            )
            self.camera_subscriber.start(init_node=True)
            logger.info(f"  - cameras:      {self.camera_subscriber.camera_names}")

        # ---- state ----
        self._current_waypoint: Optional[List[Any]] = None
        self._episode_id = 0
        self._frame_index = 0
        self._use_wbc = False

        # ---- dim config ----
        self._state_includes_chassis = config.action_config.state_includes_chassis
        self._enable_chassis = config.action_config.enable_chassis
        self._enable_head = config.action_config.enable_head
        self._enable_torso = config.action_config.enable_torso

        # NOTE: Psi0 was finetuned on 25-dim astribot state (with chassis).
        # If state_includes_chassis is False, get_current_joint_positions
        # returns 22-dim which the server's pipeline assumes is full state
        # and will misalign normalization stats. Refuse to run in that case.
        if not self._state_includes_chassis:
            logger.error(
                "Psi0 expects 25-dim state. Re-run with --state-with-chassis "
                "(state_includes_chassis=True) or you'll get garbage."
            )
            raise ValueError("state_includes_chassis must be True for Psi0")

        self._initial_transition_duration = 0.0

        logger.info(f"  - state dim:    {25 if self._state_includes_chassis else 22}")
        logger.info(f"  - exec chassis: {self._enable_chassis}")
        logger.info(f"  - exec head:    {self._enable_head}")
        logger.info(f"  - exec torso:   {self._enable_torso}")

    def _names_list(self, include_chassis: Optional[bool] = None) -> List[str]:
        use_chassis = include_chassis if include_chassis is not None else self._enable_chassis
        return ASTRIBOT_NAMES_LIST_WITH_CHASSIS if use_chassis else ASTRIBOT_NAMES_LIST

    def _apply_action_filter(self, action, current_state=None):
        action_arr = np.asarray(action, dtype=np.float32)
        state_arr = np.asarray(current_state, dtype=np.float32) if current_state is not None else None
        filtered = filter_action(
            action_arr, state_arr,
            enable_head=self._enable_head,
            enable_torso=self._enable_torso,
            enable_chassis=self._enable_chassis,
        )
        return filtered.tolist()

    # ---- state I/O ----
    def get_current_joint_positions(self) -> List[float]:
        """Return current lerobot-order state (25-dim with chassis)."""
        if self.astribot is not None:
            try:
                names = [
                    "astribot_torso",
                    "astribot_arm_left",
                    "astribot_gripper_left",
                    "astribot_arm_right",
                    "astribot_gripper_right",
                    "astribot_head",
                ]
                positions = self.astribot.get_current_joints_position(names)
                waypoint = positions
                if self._state_includes_chassis:
                    try:
                        chassis = self.astribot.get_current_joints_position(["astribot_chassis"])
                        waypoint = list(waypoint) + [chassis[0]]
                    except Exception as e:
                        logger.debug(f"chassis read failed: {e}; using zeros")
                        waypoint = list(waypoint) + [[0.0, 0.0, 0.0]]
                return waypoint_to_lerobot_action(waypoint, include_chassis=self._state_includes_chassis)
            except Exception as e:
                logger.warning(f"sdk read failed: {e}; falling back to last command")

        if self._current_waypoint:
            return waypoint_to_lerobot_action(self._current_waypoint, include_chassis=self._state_includes_chassis)
        dim = LEROBOT_ACTION_DIM_WITH_CHASSIS if self._state_includes_chassis else LEROBOT_ACTION_DIM_NO_CHASSIS
        return [0.0] * dim

    def move_to_home(self):
        logger.info("Moving to home")
        if self.astribot:
            self.astribot.move_to_home()

    def move_to_ready_position(self, duration: float = 5.0) -> bool:
        logger.info("=" * 60)
        logger.info(f"Phase 1: move to ready position ({duration:.1f}s)")
        logger.info("=" * 60)
        ready = READY_POSITION_25 if self._enable_chassis else READY_POSITION_22
        waypoint = lerobot_action_to_waypoint(ready, include_chassis=self._enable_chassis)
        if self.astribot:
            self.astribot.move_joints_waypoints(
                self._names_list(), [waypoint], [duration], use_wbc=self._use_wbc,
            )
        else:
            time.sleep(duration)
        self._current_waypoint = waypoint
        logger.info("Reached ready position")
        return True

    def set_initial_transition(self, duration: float):
        self._initial_transition_duration = duration
        if duration > 0:
            logger.info(f"Initial transition enabled: {duration:.2f}s")

    # ---- per-frame obs build / push ----
    def _push_obs(self, joint_positions: List[float], images: Optional[Dict[str, np.ndarray]]):
        """Hand a fresh payload to the WebSocket sender.

        Server expects:
            image[image_key]  -> HxWx3 uint8 ndarray, for each image_key in
                                 launch_config.data.transform.repack.image_keys
            state[state_key]  -> (Ds,) float32 ndarray, where state_key is
                                 launch_config.data.transform.repack.state_key
                                 (= "observation.state" for AstribotS1)
            instruction       -> str
        """
        state_arr = np.asarray(joint_positions, dtype=np.float32)
        payload = {
            "image":        images or {},
            "state":        {PSI_STATE_KEY: state_arr},
            "instruction":  self.task_instruction,
            "history":      {},
            "condition":    {},
            "gt_action":    np.zeros(1, dtype=np.float32),
            "dataset_name": "astribot",
            "timestamp":    str(time.time()),
        }
        self.client.set_obs(payload)

    # ---- main control step ----
    def step(self) -> bool:
        # 1. read sensors
        joint_positions = self.get_current_joint_positions()
        images = None
        if self._enable_camera and self.camera_subscriber:
            images = self.camera_subscriber.get_decoded_images()

        # 2. push obs (background sender will broadcast asap)
        self._push_obs(joint_positions, images)

        # 3. consume latest action pushed by server
        t0 = time.time()
        action, version, is_new = self.client.get_latest_action()
        ws_latency_ms = (time.time() - t0) * 1000  # not real latency, mostly lock-acquire

        if action is None:
            # Server hasn't pushed anything yet — keep running, just don't move.
            return True

        # 4. action pipeline: raw -> filter -> velocity limit -> smooth
        raw_action = action.tolist() if isinstance(action, np.ndarray) else list(action)

        action_list = self._apply_action_filter(raw_action, current_state=joint_positions)
        filtered_action = list(action_list)

        if self.velocity_limiter:
            action_list = self.velocity_limiter.limit(action_list)
        if self.smoother:
            action_list = self.smoother.smooth(action_list)
        smoothed_action = list(action_list) if (self.velocity_limiter or self.smoother) else None

        final_action = action_list

        # 5. log
        if self.inference_logger:
            self.inference_logger.log_step(
                frame_index=self._frame_index,
                state=joint_positions,
                action=final_action,
                images=images,
                episode_id=self._episode_id,
                latency_ms=ws_latency_ms,
                raw_action=raw_action,
                filtered_action=filtered_action,
                smoothed_action=smoothed_action,
                extra_info={
                    "is_inference_frame": is_new,
                    "action_version":     version,
                },
                save_images_this_step=is_new,
            )

        # 6. send to robot
        waypoint = lerobot_action_to_waypoint(final_action, include_chassis=self._enable_chassis)
        if self.astribot:
            self.astribot.set_joints_position(
                self._names_list(), waypoint,
                control_way=self.control_way, use_wbc=self._use_wbc,
            )
        self._current_waypoint = waypoint
        self._frame_index += 1
        return True

    def set_episode(self, episode: int):
        self._episode_id = episode
        self._frame_index = 0
        if self.smoother:
            self.smoother.reset()
        if self.velocity_limiter:
            self.velocity_limiter.reset()

    def _perform_initial_transition(self):
        """Wait for the server's first action, then planning-move to it."""
        if self._initial_transition_duration <= 0:
            return

        logger.info("=" * 60)
        logger.info(f"Initial transition: warm up server then move to first action "
                    f"({self._initial_transition_duration:.2f}s)")
        logger.info("=" * 60)

        action = None
        deadline = time.time() + 10.0
        while time.time() < deadline:
            joint_positions = self.get_current_joint_positions()
            images = None
            if self._enable_camera and self.camera_subscriber:
                images = self.camera_subscriber.get_decoded_images()
            self._push_obs(joint_positions, images)
            time.sleep(0.05)
            action, _, _ = self.client.get_latest_action()
            if action is not None:
                break

        if action is None:
            logger.warning("No action received within 10s; skipping initial transition")
            return

        joint_positions = self.get_current_joint_positions()
        first_action = self._apply_action_filter(action.tolist(), current_state=joint_positions)

        # max joint deviation in degrees (sanity check)
        cur = joint_positions[:len(first_action)]
        max_diff = max(abs(a - s) for a, s in zip(first_action, cur))
        logger.info(f"max delta to first action: {max_diff:.4f} rad ({np.degrees(max_diff):.2f} deg)")

        waypoint = lerobot_action_to_waypoint(first_action, include_chassis=self._enable_chassis)
        if self.astribot:
            self.astribot.move_joints_waypoints(
                self._names_list(), [waypoint],
                [self._initial_transition_duration], use_wbc=self._use_wbc,
            )
        else:
            time.sleep(self._initial_transition_duration)
        self._current_waypoint = waypoint
        logger.info("Initial transition done")

    def close(self):
        if self.camera_subscriber:
            self.camera_subscriber.stop()
        self.client.close()


# ============================================================================
# Main loop
# ============================================================================
def run_inference_loop(
    controller: AstribotController,
    episode: int = 0,
    max_frames: int = 10000,
    move_to_ready: bool = True,
    ready_move_duration: float = 5.0,
):
    global _interrupted
    _interrupted = False

    if controller.inference_logger:
        controller.inference_logger.start_session(
            episode_id=episode,
            model_path="(psi-ws-server)",
            config={
                "control_freq":            controller.config.control_freq,
                "control_way":             controller.config.control_way,
                "smooth_window":           controller.config.smooth_window,
                "max_velocity":            controller.config.max_velocity,
                "enable_head":             controller._enable_head,
                "enable_torso":            controller._enable_torso,
                "enable_chassis":          controller._enable_chassis,
                "state_includes_chassis":  controller._state_includes_chassis,
                "task_instruction":        controller.task_instruction,
            }
        )

    controller.set_episode(episode)

    # Wait for first camera images so initial obs has them
    if controller._enable_camera and controller.camera_subscriber:
        logger.info("Waiting for camera images ...")
        if not controller.camera_subscriber.wait_for_images(timeout=10.0):
            logger.warning("Some camera images not ready; continuing anyway")

    # Phase 1: move to ready position
    if move_to_ready:
        if not controller.move_to_ready_position(duration=ready_move_duration):
            logger.error("move_to_ready failed; aborting")
            return
        if _interrupted:
            return

    # Optional: smooth transition into the first inferred action
    if controller._initial_transition_duration > 0:
        controller._perform_initial_transition()
        if _interrupted:
            return

    # Phase 2: streaming inference @ control_freq
    logger.info("=" * 60)
    logger.info(f"Phase 2: streaming inference loop @ {controller.control_freq:.0f} Hz")
    logger.info("=" * 60)

    start_t = time.time()
    frame_count = 0
    while not _interrupted and frame_count < max_frames:
        loop_t0 = time.time()
        if not controller.step():
            logger.info("step() returned False; stopping")
            break
        frame_count += 1

        if frame_count % 100 == 0:
            el = time.time() - start_t
            actual_hz = frame_count / el if el > 0 else 0
            logger.info(f"frame {frame_count}/{max_frames}  ({actual_hz:.1f} Hz)")

        rest = controller.control_period - (time.time() - loop_t0)
        if rest > 0:
            time.sleep(rest)

    total_t = time.time() - start_t
    if frame_count > 0:
        logger.info(f"Done. frames={frame_count}  total={total_t:.2f}s  "
                    f"avg_hz={frame_count/total_t:.1f}")
    if controller.inference_logger:
        controller.inference_logger.end_session()


def main():
    import argparse

    global _interrupted
    parser = argparse.ArgumentParser(
        description="Astribot WebSocket inference client for Psi0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # connection
    parser.add_argument("--server", type=str, default="ws://localhost:8014/ws",
                        help="WebSocket URL of Psi RTC server (default: ws://localhost:8014/ws)")
    parser.add_argument("--task", type=str, required=True,
                        help="Language instruction sent with every obs")

    # state input
    parser.add_argument("--state-with-chassis", action="store_true", default=True,
                        help="Send 25-dim state including chassis [REQUIRED for Psi0]. "
                             "Set explicitly so it's logged.")

    # action exec
    parser.add_argument("--execute-chassis", action="store_true",
                        help="Execute chassis commands (default: off, safer)")
    parser.add_argument("--no-head", action="store_true",
                        help="Disable head actuation (replace with current head state)")
    parser.add_argument("--no-torso", action="store_true",
                        help="Disable torso actuation (replace with current torso state)")

    # cameras
    parser.add_argument("--enable-camera", action="store_true")
    parser.add_argument("--cameras", type=str, default="head,wrist_left,wrist_right,torso",
                        help="Comma list. Default uses 4 cams matching Psi0 finetune.")

    # control
    parser.add_argument("--control-freq", type=float, default=30.0)
    parser.add_argument("--control-way", type=str, default="direct", choices=["filter", "direct"])
    parser.add_argument("--smooth", type=int, default=0,
                        help="Smoothing window size (0=disabled)")
    parser.add_argument("--max-velocity", type=float, default=0.0,
                        help="Max per-frame action delta in rad (0=disabled)")

    # loop
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=10000)
    parser.add_argument("--move-to-ready", action="store_true", default=True)
    parser.add_argument("--no-move-to-ready", action="store_true")
    parser.add_argument("--ready-duration", type=float, default=5.0)
    parser.add_argument("--initial-transition", type=float, default=0.0,
                        help="Planning-move to first inferred action (s). 0=disabled.")

    # logging
    parser.add_argument("--enable-logging", action="store_true", default=True)
    parser.add_argument("--no-logging", action="store_true")
    parser.add_argument("--log-dir", type=str, default="./inference_logs")
    parser.add_argument("--log-session-name", type=str, default=None)
    parser.add_argument("--log-save-images", action="store_true", default=True)
    parser.add_argument("--no-log-save-images", action="store_true")
    parser.add_argument("--log-image-format", type=str, default="jpg", choices=["jpg", "png"])

    args = parser.parse_args()
    setup_logging("INFO")

    if not args.state_with_chassis:
        # we default-True'd it, but reject if user manually disabled it somehow
        raise SystemExit("Psi0 requires --state-with-chassis (25-dim state).")

    action_config = ActionConfig(
        state_includes_chassis=args.state_with_chassis,
        enable_chassis=args.execute_chassis,
        enable_head=not args.no_head,
        enable_torso=not args.no_torso,
    )

    # ClientConfig was originally designed for the gRPC variant — keep the
    # same shape but most fields (model_path/device/policy_type/host/port) are
    # not used in WS mode (server is pre-configured via its own CLI args).
    config = ClientConfig(
        server_host="",
        server_port=0,
        timeout=15.0,
        model_path="",
        device="",
        policy_type="",
        control_freq=args.control_freq,
        control_way=args.control_way,
        smooth_window=args.smooth,
        max_velocity=args.max_velocity,
        action_config=action_config,
    )

    camera_names = [c.strip() for c in args.cameras.split(",") if c.strip()]
    move_to_ready = args.move_to_ready and not args.no_move_to_ready

    inference_logger = None
    if args.enable_logging and not args.no_logging:
        save_images = args.log_save_images and not args.no_log_save_images
        inference_logger = InferenceLogger(
            log_dir=args.log_dir,
            session_name=args.log_session_name,
            save_images=save_images,
            image_format=args.log_image_format,
            enabled=True,
        )
        logger.info(f"Inference logging on: dir={args.log_dir}  save_images={save_images}  fmt={args.log_image_format}")

    controller: Optional[AstribotController] = None
    try:
        controller = AstribotController(
            config=config,
            server_url=args.server,
            task_instruction=args.task,
            enable_camera=args.enable_camera,
            camera_names=camera_names,
            inference_logger=inference_logger,
        )
        if args.initial_transition > 0:
            controller.set_initial_transition(args.initial_transition)

        run_inference_loop(
            controller,
            episode=args.episode,
            max_frames=args.max_frames,
            move_to_ready=move_to_ready,
            ready_move_duration=args.ready_duration,
        )

        if not _interrupted:
            logger.info("Returning to home")
            controller.move_to_home()

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as e:
        logger.error(f"error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if inference_logger:
            inference_logger.end_session()
        if controller:
            controller.close()
        logger.info("Exit")


if __name__ == "__main__":
    main()
