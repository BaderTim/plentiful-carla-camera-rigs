#!/usr/bin/env bash
set -euo pipefail

# List of checkpoints to evaluate
CHECKPOINTS=(
    "./runs/train_R1/epoch_20.pth"
    "./runs/train_R1-c6/epoch_20.pth"
    "./runs/train_R1-c10/epoch_20.pth"
    "./runs/train_R1-f/epoch_20.pth"
    "./runs/train_R1-r/epoch_20.pth"
    "./runs/train_R1-t/epoch_20.pth"
    "./runs/train_R2/epoch_20.pth"
    "./runs/train_R3/epoch_20.pth"
    "./runs/train_R4/epoch_20.pth"
    "./runs/train_R5/epoch_20.pth"
    "./runs/train_R6/epoch_20.pth"
    "./runs/train_R7/epoch_20.pth"
    "./runs/train_R8/epoch_20.pth"
    "./runs/train_R9/epoch_20.pth"
)

# List of datasets to test against
DATA_ROOT="/data"
DATASETS=(
    "R1"
    "R1-c6"
    "R1-c10"
    "R1-f"
    "R1-r"
    "R1-t"
    "R2"
    "R3"
    "R4"
    "R5"
    "R6"
    "R7"
    "R8"
    "R9"
)

# Common parameters
CONFIG="configs/nuscenes/det/centerhead/lssfpn/camera/256x704/resnet/default.yaml"
PRETRAINED_CHECKPOINT="checkpoints/resnet50-pretrained.pth"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python}"
GPUS=1

for CKPT in "${CHECKPOINTS[@]}"; do
    CKPT_NAME=$(basename "$CKPT" .pth)
    CKPT_DIR=$(dirname "$CKPT")
    for DATASET_NAME in "${DATASETS[@]}"; do
        DATASET_PATH="$DATA_ROOT/$DATASET_NAME"
        #OUTPUT_DIR="./test_work_dir${CKPT_DIR}/test_${CKPT_NAME}_on_${DATASET_NAME}"
        OUTPUT_DIR="${CKPT_DIR}/test_${CKPT_NAME}_on_${DATASET_NAME}"
        
        echo "================================================================"
        echo "TESTING CHECKPOINT: $CKPT_NAME"
        echo "ON DATASET: $DATASET_NAME ($DATASET_PATH)"
        echo "OUTPUT DIRECTORY: $OUTPUT_DIR"
        echo "================================================================"
        
        mkdir -p "$OUTPUT_DIR"
        
        torchpack dist-run -np "$GPUS" "$PYTHON_BIN" tools/test.py "$CONFIG" "$CKPT" \
            --eval bbox \
            --cfg-options \
                dataset_root="$DATASET_PATH/" \
                dataset_info_prefix="${DATASET_NAME}_infos" \
                data.test.dataset_root="$DATASET_PATH/" \
                data.test.ann_file="$DATASET_PATH/${DATASET_NAME}_infos_test.pkl" \
                model.encoders.camera.backbone.init_cfg.checkpoint="$PRETRAINED_CHECKPOINT" \
            --out "$OUTPUT_DIR/results.pkl" 2>&1 | tee "${OUTPUT_DIR}/test.log"
            
        echo "Finished testing $CKPT_NAME on $DATASET_NAME"
    done
done
