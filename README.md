# CD-MambAtt

This repository is now in the **baseline reproduction + cross-domain experimentation + next-innovation design** stage.

## Current repository status

- supervised `MambAtt` baseline: implemented and structurally aligned to the target paper
- cross-domain `CD-MambAtt v2`: implemented and validated on multiple C-MAPSS transfer pairs
- reproduced `FOMLN` baseline: implemented and running on CUDA
- next research branch: `DD-SSM / SPD`

## Main scripts

- `train_supervised.py` — supervised target-paper baseline reproduction
- `train_self_supervised.py` — same-subset self-supervised reproduction
- `train_cross_domain_baseline.py` — direct / fine-tune cross-domain baselines
- `train_cd_mambatt_v1.py` — first cross-domain MMD version
- `train_cd_mambatt_v2.py` — current stable cross-domain runner
- `train_cd_mambatt_v3.py` — SPD / DD-SSM and later mechanism experiments
- `train_fomln_baseline.py` — reproduced FOMLN baseline runner
- `train_minimal.py` — small smoke / sanity entrypoint

## Repository layout

- `cd_mambatt/` — core package (`data`, `models`, `losses`, `self_supervised`, etc.)
- `experiments/` — one-off experiment runners, now grouped into:
  - `ablations/`
  - `diagnostics/`
  - `prototypes/`
  - `quick_tests/`
- `scripts/` — reusable batch runners and result aggregation tools
- `docs/` — project docs and experiment records
- `runs/` — experiment outputs

## Documentation

Start here:

- `docs/README.md`
- `docs/project_overview_zh_2026-04-08.md`
- `docs/project_status.md`
- `docs/supervised_reproduction.md`
- `docs/self_supervised_reproduction.md`
- `docs/published_baselines.md`
- `docs/dd_ssm_roadmap.md`
- `docs/cross_domain_experiment_log.md`

Historical notes were cleaned from the docs root and preserved under:

- `docs/history/`

## Environment

Expected conda environment: `cd_mamba`

Key versions:

- Python `3.10.20`
- torch `2.1.2+cu121`
- mamba-ssm `2.2.0`
- transformers `4.38.2`
- nvcc `12.1.66`

Important compatibility decision:

- `causal-conv1d` is intentionally not used
- the project keeps `d_conv = 8` and relies on the standard Conv1d path inside `mamba-ssm`

## Data

- Windows path: `E:\datasets\CMAPSS\CMAPSSData`
- WSL path: `/mnt/e/datasets/CMAPSS/CMAPSSData`
- symlink: `/home/shelterpl/data/CMAPSS`
