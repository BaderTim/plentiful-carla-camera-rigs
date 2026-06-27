#!/usr/bin/env bash
set -euo pipefail

# Common dataset naming
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
CONFIG="${CONFIG:-projects/configs/petr/petr_r50dcn_gridmask_p4_800x320_pccr.py}"
GPUS=1
SAMPLES_PER_GPU=4
WORKERS_PER_GPU=4
EVAL_SPLIT="${EVAL_SPLIT:-test}"

resolve_checkpoint() {
    local ckpt_dir="$1"
    local latest_ckpt="$ckpt_dir/latest.pth"
    local resolved_target

    if [[ -f "$latest_ckpt" ]]; then
        if [[ "$(head -n 1 "$latest_ckpt" 2>/dev/null || true)" == "XSym" ]]; then
            resolved_target="$(sed -n '4p' "$latest_ckpt" | tr -d '\r')"
            if [[ -n "$resolved_target" && -f "$ckpt_dir/$resolved_target" ]]; then
                printf '%s\n' "$ckpt_dir/$resolved_target"
                return 0
            fi
        else
            printf '%s\n' "$latest_ckpt"
            return 0
        fi
    fi

    find "$ckpt_dir" -maxdepth 1 -type f -name 'epoch_*.pth' | sort -V | tail -n 1
}

for CKPT_DATASET in "${DATASETS[@]}"; do
    CKPT_DIR="work_dirs/petr_r50dcn_gridmask_p4_800x320_pccr/${CKPT_DATASET}"
    CKPT="$(resolve_checkpoint "$CKPT_DIR")"
    if [[ -z "$CKPT" || ! -f "$CKPT" ]]; then
        echo "Skipping missing checkpoint in: $CKPT_DIR"
        continue
    fi

    CKPT_NAME=$(basename "$CKPT")
    LOG_DIR="$(dirname "$CKPT")/test_logs"
    mkdir -p "$LOG_DIR"

    for DATASET_NAME in "${DATASETS[@]}"; do
        DATASET_PATH="$DATA_ROOT/$DATASET_NAME"
        LOG_PATH="$LOG_DIR/${CKPT_NAME}_tested_on_${DATASET_NAME}.log"
        
        echo "================================================================"
        echo "TESTING CHECKPOINT: $CKPT_NAME"
        echo "ON DATASET: $DATASET_NAME ($DATASET_PATH)"
        echo "SAVING INTO LOG: $LOG_PATH"
        echo "================================================================"
        
        ./tools/dist_test.sh "$CONFIG" "$CKPT" $GPUS \
            --cfg-options \
                dataset_name="$DATASET_NAME" \
                data_root="$DATASET_PATH/" \
                test_ann_file="$DATASET_PATH/${DATASET_NAME}_infos_${EVAL_SPLIT}.pkl" \
                data.test.data_root="$DATASET_PATH/" \
                data.test.ann_file="$DATASET_PATH/${DATASET_NAME}_infos_${EVAL_SPLIT}.pkl" \
                data.samples_per_gpu=$SAMPLES_PER_GPU \
                data.workers_per_gpu=$WORKERS_PER_GPU \
            --eval bbox 2>&1 | tee "${LOG_PATH}"
            
        echo "Finished testing $CKPT_NAME on $DATASET_NAME"
    done
done
