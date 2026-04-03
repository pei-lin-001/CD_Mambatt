# Supervised Reproduction Status

Last updated: `2026-04-02`

This file consolidates the old supervised-reproduction notes into one place.

## 1. Scope

We have reproduced the **supervised baseline path** of the target paper
`Mamba-attention: A self-supervised framework for efficient remaining useful life prediction`.

Important boundary:

- the paper's **self-supervised Section 3.4 losses** are **not yet implemented**
- therefore the current work covers the paper's **supervised baseline reproduction**, not the full SSL paper package

## 2. Paper-explicit items already aligned

| Item | Current status |
|---|---|
| Dataset = NASA C-MAPSS | aligned |
| Input features = 21 sensors | aligned |
| Window size = 20 | aligned |
| Sliding stride = 1 | aligned |
| RUL cap = 125 | aligned |
| `d_state = 16` | aligned |
| `d_conv = 8` | aligned |
| Transformer layers = 3 | aligned |
| Attention heads = 7 | aligned |
| Optimizer = Adam | aligned |
| Paper LR = `1e-3` | supported |
| Dropout rate = `0.5` | implemented as output dropout |
| Engine-level `80/20` split | aligned |
| Multi-run model selection by validation RMSE | aligned |

## 3. Adopted implementation choices for paper ambiguities

These are necessary choices because the paper is not fully explicit.

| Ambiguity | Current adopted judgment |
|---|---|
| Transformer norm order | paper is internally inconsistent; both `pre` and `post` were tested |
| FFN hidden size `d_ff` | `84` is the strongest currently supported value |
| Validation scoring | `val_all_windows` is empirically much stronger than last-window-only validation |
| Split reuse vs resampling | `--resample-split-per-seed` is closer to the paper wording |
| Inner Transformer dropout | keep internal dropout at `0.0`; only output dropout `0.5` is supported |
| Mamba external width | keep external width at `21`; treat `d_model != 21` only as exploratory |
| Mamba block wrapper | `bare` is paper-closer and empirically better than an added residual wrapper |

## 4. Controlled findings from local experiments

| Issue | What was tested | Current conclusion |
|---|---|---|
| `pre` vs `post` norm | with weak `d_ff=2048`, `post` was worse; with `d_ff=84 + val_all_windows`, `post` became stronger | norm order interacts with other settings; `post` is currently the best empirical direction |
| `d_ff` | `84 / 128 / 256 / 512 / 2048` | `84` is the most defensible current choice |
| validation windows | last-window vs all-window validation | `val_all_windows` clearly improves model selection |
| internal Transformer dropout | `0.1` and `0.5` tested | both worse than output-only dropout |
| extra residual around Mamba block | tested | worse |
| official PyTorch Transformer implementation | tested | worse than the custom implementation |
| `expand=1` | tested | worse than `expand=2` |
| gradient clipping | tested as a standalone fix | not the main missing factor |
| `d_model = 84` exploratory branch | tested | did not beat `d_model = 21` |

## 5. Current working definitions

### 5.1 Strict paper Table-2 reproduction

Keep only paper-explicit settings fixed and openly acknowledge unresolved ambiguities:

- LR `1e-3`
- batch size `64`
- `d_state = 16`
- `d_conv = 8`
- 3 Transformer layers
- 7 heads
- Adam
- 50-epoch / 50-run interpretation
- engine-level `80/20` split

Remaining ambiguity still exists in:

- norm order
- `d_ff`
- validation window policy
- split reuse vs per-seed resampling

### 5.2 Adopted reproduction choice

This is the most reasonable paper-close line for controlled work:

- external width `21`
- Mamba block mode = `bare`
- output dropout only
- documented split protocol (`fixed` or `resample`) must be stated explicitly

### 5.3 Best empirical supervised baseline

Current strongest local supervised line:

- `transformer_norm_mode = post`
- `dim_feedforward = 84`
- `val_all_windows = True`
- `lr = 5e-4`
- `weight_decay = 1e-4`
- output dropout `0.5`
- seeds `42,43,44`

Results:

- fixed split mean test RMSE = **15.0937**
- resampled-split mean test RMSE = **15.3024**

## 6. Gap to the paper

Paper references:

- Table 3 `FD001` RMSE = **11.46**
- Table 7 Transformer-decoder mean = **12.34**

Current conclusion:

> We have a near-paper supervised implementation, but not proof of exact author-level numerical reproduction.

The remaining gap is more likely due to hidden training protocol details,
undocumented implementation choices, or the non-fused Conv1d fallback path,
not a missing top-level backbone module.

## 7. Environment-specific note

`causal-conv1d` was intentionally removed.

Reason:

- the paper uses `d_conv = 8`
- the official `causal-conv1d` package only supports widths `2/3/4`
- letting `mamba-ssm` use its standard Conv1d path is more paper-consistent than forcing an incompatible fused kernel path

## 8. Where the detailed old notes went

Historical detailed writeups were preserved in:

- `history/reproduction_notes/paper_alignment_supervised.md`
- `history/reproduction_notes/supervised_reproduction_tiers.md`
- `history/reproduction_notes/claude_findings_response.md`
