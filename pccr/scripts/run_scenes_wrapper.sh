#!/bin/bash

#
# This tool is a wrapper to run the dataset generation script.
# It iterates over a list of camera rigs, starting and stopping the CARLA server
# for each rig to manage memory usage effectively. 
#

# Configuration
# CARLA_ROOT can be overridden at runtime to point to any CARLA installation.
CARLA_ROOT="${CARLA_ROOT:-/home/timbader/CARLA_0.9.16}"
CARLA_SH="${CARLA_ROOT}/CarlaUE4.sh"
PYTHON="/home/timbader/miniforge3/envs/carla/bin/python"
OUTPUT_DIR="./output/data"
SCENES_CONFIG="./output/trajectories_pruned/scenes.json"
TRAJECTORIES_DIR="./output/trajectories_pruned"

# List of camera rigs to process
RIGS=(
    "configs/rigs/R1-c6.json"
    "configs/rigs/R1-c10.json"
    "configs/rigs/R1-f.json"
    "configs/rigs/R1-r.json"
    "configs/rigs/R1-t.json"
    "configs/rigs/R1.json"
    "configs/rigs/R2.json"
    "configs/rigs/R3.json"
    "configs/rigs/R4.json"
    "configs/rigs/R5.json"
    "configs/rigs/R6.json"
    "configs/rigs/R7.json"
    "configs/rigs/R8.json"
    "configs/rigs/R9.json"
)

echo "Starting dataset generation wrapper..."
echo "Output directory: $OUTPUT_DIR"

for RIG in "${RIGS[@]}"; do
    RIG_NAME=$(basename "$RIG" .json)
    echo "============================================================"
    echo "PROCESSING RIG: $RIG_NAME"
    echo "============================================================"
    
    # 1. Start CARLA server in background, with NVIDIA offloading and Vulkan ICD set for compatibility (optional but improves performance on some setups)
    __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only $CARLA_SH -nosound -quality-level=Epic -RenderOffScreen &
    CARLA_PID=$!
    
    # 2. Wait for CARLA to become fully ready (poll instead of fixed sleep)
    echo "Waiting for CARLA to become ready (PID: $CARLA_PID)..."
    CARLA_READY=0
    for i in $(seq 1 60); do
        if $PYTHON -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(2.0); c.get_server_version()" 2>/dev/null
        then
            echo "CARLA is ready (after ${i}s)."
            CARLA_READY=1
            break
        fi
        sleep 1
    done

    if [ $CARLA_READY -eq 0 ]; then
        echo "Error: CARLA did not become ready within 60 seconds on rig $RIG_NAME. Aborting."
        kill -9 $CARLA_PID 2>/dev/null
        pkill -9 -f CarlaUE4 2>/dev/null
        exit 1
    fi
    
    # 3. Run the recording script for just this specific rig
    # The script will check resume_state.json and skip if already finished.
    $PYTHON ./core/scene_runner.py \
      --output "$OUTPUT_DIR" \
      --scenes "$SCENES_CONFIG" \
      --camera-rigs "$RIG" \
      --trajectories "$TRAJECTORIES_DIR" 
    
    EXIT_CODE=$?
    
    # 4. Shutdown CARLA to clear all VRAM and host RAM
    echo "Recording for $RIG_NAME finished with exit code $EXIT_CODE."
    echo "Cleaning up CARLA process..."
    kill -9 $CARLA_PID 2>/dev/null
    pkill -9 -f CarlaUE4 2>/dev/null
    wait $CARLA_PID 2>/dev/null
    
    # 5. Cool down period before next map load/server start
    sleep 5
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "Warning: run_scenes.py exited with an error on rig $RIG_NAME."
        # Optional: exit 1 to stop if a rig fails
    fi
done

echo "All requested rigs have been processed."
