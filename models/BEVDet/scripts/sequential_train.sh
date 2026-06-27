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
CONFIG="configs/bevdet/bevdet-r50-pccr.py"
GPUS=2 # adjust learning rate when changing this
SAMPLES_PER_GPU=4 # batch size = GPUS * SAMPLES_PER_GPU
WORKERS_PER_GPU=4

for DATASET_NAME in "${DATASETS[@]}"; do
    DATASET_PATH="$DATA_ROOT/$DATASET_NAME"

    echo "================================================================"
    echo "RUNNING TRAINING FOR DATASET: $DATASET_NAME"
    echo "ROOT PATH: $DATA_ROOT"
    echo "DATASET_PATH: $DATASET_PATH"
    echo "================================================================"

    ./tools/dist_train.sh "$CONFIG" $GPUS \
        --cfg-options \
            data_root="$DATASET_PATH/" \
            data.train.data_root="$DATASET_PATH/" \
            data.train.ann_file="$DATASET_PATH/${DATASET_NAME}_infos_train.pkl" \
            data.val.data_root="$DATASET_PATH/" \
            data.val.ann_file="$DATASET_PATH/${DATASET_NAME}_infos_val.pkl" \
            data.test.data_root="$DATASET_PATH/" \
            data.test.ann_file="$DATASET_PATH/${DATASET_NAME}_infos_test.pkl" \
            data.samples_per_gpu=$SAMPLES_PER_GPU \
            data.workers_per_gpu=$WORKERS_PER_GPU \
        --work-dir="./work_dirs/bevdet-r50-pccr/$DATASET_NAME"

    echo "Finished training for $DATASET_NAME"
done
