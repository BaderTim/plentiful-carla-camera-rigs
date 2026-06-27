#!/usr/bin/env bash
set -euo pipefail

# List of dataset roots to train on
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
GPUS=2 # make sure to adjust LR in the config file if you change the number of GPUs or samples per GPU
SAMPLES_PER_GPU=4 # batch size = GPUS * SAMPLES_PER_GPU
WORKERS_PER_GPU=4

for DATASET_NAME in "${DATASETS[@]}"; do
    DATASET_PATH="$DATA_ROOT/$DATASET_NAME"

    echo "================================================================"
    echo "RUNNING TRAINING FOR DATASET: $DATASET_NAME"
    echo "ROOT PATH: $DATA_ROOT"
    echo "DATASET_PATH: $DATASET_PATH"
    echo "================================================================"

    torchpack dist-run -np "$GPUS" "$PYTHON_BIN" tools/train.py "$CONFIG" \
        --model.encoders.camera.backbone.init_cfg.checkpoint "$PRETRAINED_CHECKPOINT" \
        --dataset_root "$DATASET_PATH/" \
        --dataset_info_prefix "${DATASET_NAME}_infos" \
        --data.train.dataset.dataset_root "$DATASET_PATH/" \
        --data.train.dataset.ann_file "$DATASET_PATH/${DATASET_NAME}_infos_train.pkl" \
        --data.val.dataset_root "$DATASET_PATH/" \
        --data.val.ann_file "$DATASET_PATH/${DATASET_NAME}_infos_val.pkl" \
        --data.test.dataset_root "$DATASET_PATH/" \
        --data.test.ann_file "$DATASET_PATH/${DATASET_NAME}_infos_test.pkl" \
        --data.samples_per_gpu "$SAMPLES_PER_GPU" \
        --data.workers_per_gpu "$WORKERS_PER_GPU" \
        --run-dir "runs/train_$DATASET_NAME"

    echo "Finished training for $DATASET_NAME"
done
