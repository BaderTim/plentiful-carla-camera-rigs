#!/bin/bash

#
# Wrapper to record traffic/pedestrian trajectories for deterministic scene replay.
# Starts a single CARLA server instance, runs record_trajectories.py for the
# specified split(s), then shuts CARLA down. No mid-run restarts are needed
# because trajectory recording does not hold long-lived map state like scene_runner does.
#

# Configuration
# CARLA_ROOT can be overridden at runtime to point to any CARLA installation.
CARLA_ROOT="${CARLA_ROOT:-/home/timbader/CARLA_0.9.16}"
CARLA_SH="${CARLA_ROOT}/CarlaUE4.sh"
# find with "which python" or "which python3" in the terminal, or set to your Python environment's python executable
PYTHON="/home/timbader/miniforge3/envs/carla/bin/python"
OUTPUT_DIR="./output/trajectories"
SCENES_CONFIG="configs/scenes.json"
SPLITS=(
    "mini"
    "trainval"
    "test"
)
MAX_PEDESTRIANS=100

echo "Starting trajectory recording wrapper..."
echo "Splits:     ${SPLITS[*]}"
echo "Output dir: $OUTPUT_DIR"
echo "Scenes:     $SCENES_CONFIG"

# 1. Start CARLA server in background, with NVIDIA offloading and Vulkan ICD set for compatibility (optional but improves performance on some setups)
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia  __VK_LAYER_NV_optimus=NVIDIA_only $CARLA_SH -nosound -vulkan -RenderOffScreen &
CARLA_PID=$!

# 2. Wait for CARLA to become fully ready (poll instead of fixed sleep)
echo "Waiting for CARLA to become ready (PID: $CARLA_PID)..."
CARLA_READY=0
for i in $(seq 1 60); do
    if $PYTHON -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(2.0); c.get_server_version()" 2>/dev/null
    then    dmesg --ctime | tail -n 200 | grep -i -E 'out of memory|oom|killed process'
        echo "CARLA is ready (after ${i}s)."
        CARLA_READY=1
        break
    fi
    sleep 1
done

if [ $CARLA_READY -eq 0 ]; then
    echo "Error: CARLA did not become ready within 60 seconds. Aborting."
    kill -9 $CARLA_PID 2>/dev/null
    pkill -9 -f CarlaUE4 2>/dev/null
    exit 1
fi

# 3. Run trajectory recording for each split
for SPLIT in "${SPLITS[@]}"; do
    echo "============================================================"
    echo "RECORDING SPLIT: $SPLIT"
    echo "============================================================"

    $PYTHON ./core/record_trajectories.py \
      --output "$OUTPUT_DIR" \
      --scenes "$SCENES_CONFIG" \
      --split "$SPLIT" \
      --max-pedestrians "$MAX_PEDESTRIANS"

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "Warning: record_trajectories.py exited with an error on split '$SPLIT' (exit code $EXIT_CODE)."
        echo "Cleaning up CARLA process..."
        kill -9 $CARLA_PID 2>/dev/null
        pkill -9 -f CarlaUE4 2>/dev/null
        wait $CARLA_PID 2>/dev/null
        exit $EXIT_CODE
    fi
done

# 4. Shut down CARLA
echo "All splits finished. Cleaning up CARLA process..."
kill -9 $CARLA_PID 2>/dev/null
pkill -9 -f CarlaUE4 2>/dev/null
wait $CARLA_PID 2>/dev/null

echo "Done. Trajectories written to: $OUTPUT_DIR"
