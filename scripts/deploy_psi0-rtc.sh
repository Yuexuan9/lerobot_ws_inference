#!/bin/bash

PORT=8014
TASK="Spray the bowl and wipe it and stack it up."

cd "$(dirname "$0")/.."

python deploy/psi-inference_rtc.py \
    --port "$PORT" \
    --task "$TASK"
