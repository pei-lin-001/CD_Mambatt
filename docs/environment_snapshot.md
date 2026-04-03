# Environment Snapshot

Date: `2026-04-02`

Conda environment: `cd_mamba`

## Installed versions relevant to reproduction

- Python `3.10.20`
- torch `2.1.2+cu121`
- torchvision `0.16.2+cu121`
- torchaudio `2.1.2+cu121`
- numpy `1.24.3`
- mamba-ssm `2.2.0`
- transformers `4.38.2`
- nvcc `12.1.66`
- gcc_linux-64 `12.4.0`
- gxx_linux-64 `12.4.0`

## CUDA status

- project policy: **all serious experiments run on CUDA**
- `torch.cuda.is_available()` has been validated in `cd_mamba`
- `mamba_ssm` CUDA forward has been validated
- `nvidia-smi` is expected to work on the host and inside the usable WSL path

If WSL temporarily loses the CUDA path, the practical recovery order is:

1. restart the shell / WSL session
2. reactivate `cd_mamba`
3. re-check `torch.cuda.is_available()` and `nvidia-smi`

## Compatibility decision

`causal-conv1d` was removed on purpose.

Reason:

1. the target paper uses `d_conv = 8`
2. the official `causal-conv1d` package only supports widths `2`, `3`, and `4`
3. keeping that package would force an incompatible kernel path
4. letting `mamba-ssm` fall back to its standard Conv1d path is more paper-consistent for this project

## Data location

- Windows: `E:\datasets\CMAPSS\CMAPSSData`
- WSL: `/mnt/e/datasets/CMAPSS/CMAPSSData`
- Symlink: `/home/shelterpl/data/CMAPSS`
