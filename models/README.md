# Model Baselines (Paper Reproduction)

This folder contains the baseline model artifacts used in the paper.

- All baselines are MMDetection3D-style model pipelines.
- For each baseline, this repo provides:
	- PCCR dataset adapter file,
	- PCCR model config,
	- Docker / devcontainer setup,
	- sequential train/test (and for some models inference) scripts across all rigs.

For installation details, framework-specific dependency versions, and upstream training internals, use the original repositories:

- BEVFusion: https://github.com/mit-han-lab/bevfusion
- PETR: https://github.com/megvii-research/PETR
- Fast-BEV: https://github.com/sense-gvt/fast-bev
- BEVDet: https://github.com/HuangJunJie2017/BEVDet

## Folder Layout

### BEVDet

- Config: `BEVDet/bevdet-r50-pccr.py`
- Dataset adapter: `BEVDet/pccr_dataset.py`
- Docker: `BEVDet/docker/Dockerfile`
- Devcontainer: `BEVDet/.devcontainer/devcontainer.json`
- Scripts:
	- `BEVDet/scripts/sequential_train.sh`
	- `BEVDet/scripts/sequential_test.sh`

### BEVFusion

- Config: `BEVFusion/configs.yaml`
- Dataset adapter: `BEVFusion/pccr_dataset.py`
- Docker: `BEVFusion/Dockerfile`
- Devcontainer: `BEVFusion/.devcontainer/devcontainer.json`
- Scripts:
	- `BEVFusion/scripts/sequential_train.sh`
	- `BEVFusion/scripts/sequential_test.sh`

### Fast-BEV

- Config: `Fast-BEV/fastbev_m4_r50_s320x576_v250x250x6_c256_d6_f4_pccr.py`
- Dataset adapter: `Fast-BEV/pccr_dataset.py`
- Docker: `Fast-BEV/Dockerfile`
- Devcontainer: `Fast-BEV/.devcontainer/devcontainer.json`
- Scripts:
	- `Fast-BEV/scripts/sequential_train.sh`
	- `Fast-BEV/scripts/sequential_test.sh`

### PETR

- Config: `PETR/petr_r50dcn_gridmask_p4_800x320_pccr.py`
- Dataset adapter: `PETR/pccr_dataset.py`
- Devcontainer: `PETR/.devcontainer/devcontainer.json`
- Scripts:
	- `PETR/scripts/sequential_train.sh`
	- `PETR/scripts/sequential_test.sh`

## Dataset Conventions Used by the Scripts

Most scripts assume:

- dataset root mounted at `/data`
- one folder per rig, e.g. `/data/R1`, `/data/R1-c6`, ..., `/data/R9`
- info files named:
	- `${RIG}_infos_train.pkl`
	- `${RIG}_infos_val.pkl`
	- `${RIG}_infos_test.pkl`

Rig names expected by the scripts:

`R1`, `R1-c6`, `R1-c10`, `R1-f`, `R1-r`, `R1-t`, `R2`, `R3`, `R4`, `R5`, `R6`, `R7`, `R8`, `R9`.

## Typical Usage

Run commands from the corresponding model directory (for example from `models/BEVDet`, `models/PETR`, ...).

### 1) Train sequentially across all rigs

```bash
bash scripts/sequential_train.sh
```

### 2) Test checkpoints sequentially across all rigs

```bash
bash scripts/sequential_test.sh
```

### 3) Export predictions (models that provide inference scripts)

```bash
# BEVFusion
bash scripts/sequential_inference.sh

# PETR
bash scripts/sequential_inference.sh
```

## Devcontainer / Docker Notes

- Each model folder includes a `.devcontainer/devcontainer.json` and `postcreate.sh`.
- Data mounts in devcontainer files point to local machine paths and may need adjustment for your environment.
- Dockerfiles are provided per model variant to match the paper setup.

## Results Folder

`models/results/` contains standardized cross-rig results used by the metrics pipeline.

- Per-model files such as `trained_on_R*.json`
- Aggregate summaries such as `average_over_all_rigs.json`
- Global manifest: `models/results/manifest.json`

These outputs are consumed directly by the analysis scripts in `metrics/`.