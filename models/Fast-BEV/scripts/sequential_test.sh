#!/usr/bin/env bash
set -euo pipefail

# List of checkpoints to evaluate
CHECKPOINTS=(
    "work_dirs/train_R1/epoch_40.pth"
    "work_dirs/train_R1-c10/epoch_40.pth"
    "work_dirs/train_R1-c6/epoch_40.pth"
    "work_dirs/train_R1-f/epoch_40.pth"
    "work_dirs/train_R1-r/epoch_40.pth"
    "work_dirs/train_R1-t/epoch_40.pth"
    "work_dirs/train_R2/epoch_40.pth"
    "work_dirs/train_R3/epoch_40.pth"
    "work_dirs/train_R4/epoch_40.pth"
    "work_dirs/train_R5/epoch_40.pth"
    "work_dirs/train_R6/epoch_40.pth"
    "work_dirs/train_R7/epoch_40.pth"
    "work_dirs/train_R8/epoch_40.pth"
    "work_dirs/train_R9/epoch_40.pth"
)

# List of rigs to test against
DATA_ROOT="${DATA_ROOT:-/data}"
RIG_NAMES=(
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
EVAL_SPLIT="${EVAL_SPLIT:-test}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"
DRY_RUN="${DRY_RUN:-0}"

resolve_config_path() {
    local checkpoint_dir="$1"
    local config_paths=()

    while IFS= read -r config_path; do
        config_paths+=("$config_path")
    done < <(find "$checkpoint_dir" -maxdepth 1 -type f -name '*.py' | sort)

    if [[ ${#config_paths[@]} -eq 0 ]]; then
        echo "No config file found in checkpoint directory: $checkpoint_dir" >&2
        return 1
    fi

    if [[ ${#config_paths[@]} -ne 1 ]]; then
        echo "Expected exactly one config file in checkpoint directory: $checkpoint_dir" >&2
        printf '  %s\n' "${config_paths[@]}" >&2
        return 1
    fi

    printf '%s\n' "${config_paths[0]}"
}

run_test() {
    local config_path="$1"
    local checkpoint_path="$2"
    local output_dir="$3"
    local rig_name="$4"
    local dataset_path="$5"
    local ann_file="$6"

    local cmd=(
        "$PYTHON_BIN"
        tools/test.py
        "$config_path"
        "$checkpoint_path"
        --eval bbox
        --cfg-options
        "rig_name=$rig_name"
        "data_root=${dataset_path}/"
        "data.test.data_root=${dataset_path}/"
        "data.test.ann_file=$ann_file"
        --out "$output_dir/results.pkl"
    )

    if [[ "$GPUS" -gt 1 ]]; then
        cmd=(
            "$PYTHON_BIN"
            -m
            torch.distributed.run
            --nproc_per_node="$GPUS"
            --master_port="$MASTER_PORT"
            tools/test.py
            "$config_path"
            "$checkpoint_path"
            --launcher pytorch
            --eval bbox
            --cfg-options
            "rig_name=$rig_name"
            "data_root=${dataset_path}/"
            "data.test.data_root=${dataset_path}/"
            "data.test.ann_file=$ann_file"
            --out "$output_dir/results.pkl"
        )
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY RUN:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
        return 0
    fi

    PYTHONPATH="$(dirname "$0")/..:${PYTHONPATH:-}" "${cmd[@]}" 2>&1 | tee "$output_dir/test.log"
}

for CKPT in "${CHECKPOINTS[@]}"; do
    if [[ ! -f "$CKPT" ]]; then
        echo "Checkpoint not found: $CKPT" >&2
        exit 1
    fi

    CKPT_NAME=$(basename "$CKPT" .pth)
    CKPT_DIR=$(dirname "$CKPT")
    CONFIG_PATH=$(resolve_config_path "$CKPT_DIR")

    for RIG_NAME in "${RIG_NAMES[@]}"; do
        DATASET_PATH="$DATA_ROOT/$RIG_NAME"
        ANN_FILE="$DATASET_PATH/${RIG_NAME}_infos_${EVAL_SPLIT}.pkl"
        OUTPUT_DIR="${CKPT_DIR}/test_${CKPT_NAME}_on_${RIG_NAME}_${EVAL_SPLIT}"

        if [[ ! -d "$DATASET_PATH" ]]; then
            echo "Dataset root not found: $DATASET_PATH" >&2
            exit 1
        fi

        if [[ ! -f "$ANN_FILE" ]]; then
            echo "Annotation file not found: $ANN_FILE" >&2
            exit 1
        fi
        
        echo "================================================================"
        echo "TESTING CHECKPOINT: $CKPT_NAME"
        echo "CONFIG: $CONFIG_PATH"
        echo "ON RIG: $RIG_NAME ($DATASET_PATH)"
        echo "EVAL SPLIT: $EVAL_SPLIT"
        echo "OUTPUT DIRECTORY: $OUTPUT_DIR"
        echo "================================================================"
        
        mkdir -p "$OUTPUT_DIR"

        run_test "$CONFIG_PATH" "$CKPT" "$OUTPUT_DIR" "$RIG_NAME" "$DATASET_PATH" "$ANN_FILE"

        echo "Finished testing $CKPT_NAME on $RIG_NAME"
    done
done
