# Metrics for Rig Shift Analysis

This folder contains scripts used to reproduce the paper-level rig-shift analysis from model outputs in `models/results` and rig definitions in `pccr/configs/rigs`.

The workflow combines two complementary rig descriptors:

- **RigV (Rig Variance)**: internal heterogeneity within one rig (camera position/orientation/FOV diversity).
- **RigCD (Rig Contrastive Distance)**: cross-rig discrepancy between a train rig and a test rig, including camera-count mismatch.

## Minimal Metric Summary

For a camera with translation $t$, rotation $r$, and FOV $f$:

- RigV averages pairwise within-rig differences:
	- normalized translation spread
	- normalized rotation spread
	- normalized FOV spread
- RigCD computes optimal camera matching between two rigs and combines:
	- matched geometric/FOV discrepancy
	- camera-count penalty

In code, these are implemented in:

- `metrics/rigv.py` (single-rig metric)
- `metrics/rcd.py` (pairwise rig distance)

The higher-level scripts in this folder calibrate RigCD against observed cross-rig mAP drops and evaluate correlation/prediction quality.

## Inputs

- **Standardized model results**: `../models/results`
- **Rig JSON files**: `../pccr/configs/rigs`

Expected rig names follow the `R*` naming scheme (for example `R1`, `R1-c6`, `R1-c10`, `R1-f`, `R1-r`, `R1-t`, `R2`, ...).

## Setup

From repository root:

```bash
pip install -r metrics/requirements.txt
```

## Quick Metric Checks (single rig / rig pair)

From `metrics/`:

```bash
# RigV for one or multiple rigs
python3 -m metrics.rigv ../pccr/configs/rigs/R1.json ../pccr/configs/rigs/R1-c10.json

# RigCD between two rigs
python3 -m metrics.rcd ../pccr/configs/rigs/R1.json ../pccr/configs/rigs/R3.json
```

## Reproduce Paper Analysis

Run these from `metrics/`.

```bash
# 1) Cross-rig mAP matrices + heatmaps per model and combined
python3 cross_rig_evaluation.py \
	--results-root ../models/results \
	--output-root output

# 2) Per-model RigCD calibration (fit on control rigs, evaluate held-out)
python3 rigcd_calibration_and_evaluation.py \
	--results-root ../models/results \
	--rigs-root ../pccr/configs/rigs \
	--output-root output

# 3) Single shared RigCD calibration across all models
python3 multi_model_rigcd_calibration_and_evaluatuion.py \
	--results-root ../models/results \
	--rigs-root ../pccr/configs/rigs \
	--output-root output

# 4) Component knockout / contribution analysis
python3 component_contribution_analysis.py \
	--results-root ../models/results \
	--rigs-root ../pccr/configs/rigs \
	--output-root output
```

## Outputs

Outputs are written under `metrics/output/` in subfolders:

- `cross_rig_evaluation/`
- `rigcd_calibration_and_evaluation/`
- `multi_model_rigcd_calibration_and_evaluation/`
- `component_contribution_analysis/`

Each subfolder includes CSV tables, JSON summaries, and heatmaps used for analysis figures.

## Optional: Standardize Raw Logs

If you start from raw testing logs instead of `models/results`, use:

```bash
python3 utils/standardize_results.py \
	--raw-root raw_results \
	--output-root results
```

Then pass `--results-root results` to the analysis scripts above.
