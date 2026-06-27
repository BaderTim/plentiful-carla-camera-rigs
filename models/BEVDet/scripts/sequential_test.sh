#!/usr/bin/env bash
set -euo pipefail

# List of run targets to evaluate.
# Each item can be either:
# - a work dir, in which case latest.pth and the single config .py are resolved
# - a checkpoint path, in which case the config is resolved from its directory
RUN_TARGETS=(
    "work_dirs/petr_r50dcn_gridmask_p4/R1"
    "work_dirs/petr_r50dcn_gridmask_p4/R1-c6"
    "work_dirs/petr_r50dcn_gridmask_p4/R1-c10"
    "work_dirs/petr_r50dcn_gridmask_p4/R1-f"
    "work_dirs/petr_r50dcn_gridmask_p4/R1-r"
    "work_dirs/petr_r50dcn_gridmask_p4/R1-t"
    "work_dirs/petr_r50dcn_gridmask_p4/R2"
    "work_dirs/petr_r50dcn_gridmask_p4/R3"
    "work_dirs/petr_r50dcn_gridmask_p4/R4"
    "work_dirs/petr_r50dcn_gridmask_p4/R5"
    "work_dirs/petr_r50dcn_gridmask_p4/R6"
    "work_dirs/petr_r50dcn_gridmask_p4/R7"
    "work_dirs/petr_r50dcn_gridmask_p4/R8"
    "work_dirs/petr_r50dcn_gridmask_p4/R9"
)


# List of rigs to test against
DATA_ROOT="${DATA_ROOT:-/data}"
DEFAULT_RIG_NAMES=(
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
ONLY_RIGS="${ONLY_RIGS:-}"
EXTRA_CFG_OPTIONS="${EXTRA_CFG_OPTIONS:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

select_rig_names() {
    if [[ -z "$ONLY_RIGS" ]]; then
        printf '%s\n' "${DEFAULT_RIG_NAMES[@]}"
        return 0
    fi

    tr ',' '\n' <<< "$ONLY_RIGS"
}

resolve_checkpoint_path() {
    local run_target="$1"

    if [[ -f "$run_target" ]]; then
        printf '%s\n' "$run_target"
        return 0
    fi

    if [[ -d "$run_target" ]]; then
        local latest_checkpoint="$run_target/latest.pth"
        if [[ -f "$latest_checkpoint" ]]; then
            printf '%s\n' "$latest_checkpoint"
            return 0
        fi

        echo "No latest checkpoint found in work dir: $run_target" >&2
        return 1
    fi

    echo "Run target not found: $run_target" >&2
    return 1
}

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

append_extra_cfg_options() {
    local -n cfg_options_ref=$1
    local extra_option

    if [[ -z "$EXTRA_CFG_OPTIONS" ]]; then
        return 0
    fi

    read -r -a extra_options <<< "$EXTRA_CFG_OPTIONS"
    for extra_option in "${extra_options[@]}"; do
        cfg_options_ref+=("$extra_option")
    done
}

run_test() {
    local config_path="$1"
    local checkpoint_path="$2"
    local output_dir="$3"
    local rig_name="$4"
    local dataset_path="$5"
    local ann_file="$6"

    local cfg_options=(
        "data_root=${dataset_path}/"
        "data.test.data_root=${dataset_path}/"
        "data.test.ann_file=$ann_file"
    )
    local cmd=()

    append_extra_cfg_options cfg_options

    if [[ "$GPUS" -gt 1 ]]; then
        cmd=(
            "$REPO_ROOT/tools/dist_test.sh"
            "$config_path"
            "$checkpoint_path"
            "$GPUS"
            --eval bbox
            --cfg-options
            "${cfg_options[@]}"
            --out "$output_dir/results.pkl"
        )
    else
        cmd=(
            "$PYTHON_BIN"
            "$REPO_ROOT/tools/test.py"
            "$config_path"
            "$checkpoint_path"
            --eval bbox
            --cfg-options
            "${cfg_options[@]}"
            --out "$output_dir/results.pkl"
        )
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        if [[ "$GPUS" -gt 1 ]]; then
            printf 'DRY RUN: PORT=%q' "$MASTER_PORT"
        else
            printf 'DRY RUN: PYTHONPATH=%q' "$REPO_ROOT:${PYTHONPATH:-}"
        fi
        printf ' %q' "${cmd[@]}"
        printf '\n'
        return 0
    fi

    if [[ "$GPUS" -gt 1 ]]; then
        PORT="$MASTER_PORT" "${cmd[@]}" 2>&1 | tee "$output_dir/test.log"
        return 0
    fi

    PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" "${cmd[@]}" 2>&1 | tee "$output_dir/test.log"
}

if [[ $# -gt 0 ]]; then
    RUN_TARGETS=("$@")
fi

mapfile -t RIG_NAMES < <(select_rig_names)

for RUN_TARGET in "${RUN_TARGETS[@]}"; do
    CKPT=$(resolve_checkpoint_path "$RUN_TARGET")
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
