# Supervised Reproduction Tiers

This note separates the current supervised FD001 reproduction work into
three layers:

1. paper-explicit items
2. paper-implicit or paper-ambiguous items that still need an adopted implementation choice
3. empirical enhancement items that improve results in the current codebase but are not a strict paper reproduction

The goal is to keep "paper reproduction" and "best empirical baseline"
clearly separated before later cross-domain or self-supervised work.

## Tier 1: Paper-Explicit Items

These are directly stated in the paper and should be treated as
non-negotiable for a strict supervised reproduction unless we later find
evidence that the paper text itself is wrong.

| Item | Paper value | Current implementation status | Code / note |
|---|---|---|---|
| Dataset | NASA C-MAPSS | aligned | [data.py](/home/shelterpl/cd_mambatt/cd_mambatt/data.py#L195) |
| Input features | 21 sensors | aligned | [data.py](/home/shelterpl/cd_mambatt/cd_mambatt/data.py#L18) |
| Window size | 20 | aligned | [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py#L312) |
| Sliding stride | 1 | aligned | [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py#L313) |
| RUL cap | 125 | aligned | [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py#L314) |
| Batch size | 64 | aligned | [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py#L325) |
| Mamba d_state | 16 | aligned | [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py#L330) |
| Mamba d_conv | 8 | aligned | [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py#L331) |
| Transformer layers | 3 | aligned | [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py#L334) |
| Attention heads | 7 | aligned | [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py#L335) |
| Optimizer | Adam | aligned | [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py#L233) |
| Learning rate | 1e-3 | supported, but no longer the strongest local setting | [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py#L327) |
| Dropout rate | 0.5 | partially aligned; output dropout is 0.5 | [mambatt.py](/home/shelterpl/cd_mambatt/cd_mambatt/models/mambatt.py#L198) |
| Iterations | 50 | implemented as 50 epochs per run and 50 seed-trials by default | [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py#L323) and [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py#L326) |
| Validation split | 80% train engines / 20% val engines | aligned | [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py#L317) |
| Model selection | choose lowest val RMSE, then test | aligned | [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py#L367) |

## Tier 2: Paper-Ambiguous But Required Choices

These items are not fully determined by the paper text. We still have to
choose something in code, but those choices should be labeled as
"adopted reproduction choices", not "paper-confirmed facts".

| Item | Paper situation | Current options / evidence | Current best-supported choice |
|---|---|---|---|
| Transformer norm order | Algorithm 1 is Pre-LN, Equation 8 / figure read more like Post-LN | both implemented | `post` is currently stronger empirically, but not paper-certain |
| Transformer FFN hidden size `d_ff` | not published | swept locally | `84` is currently the most defensible empirical choice |
| Validation window policy | paper only says 80/20 engine split; does not say last window vs all windows for validation | both implemented | `val_all_windows` is stronger empirically |
| Split resampling policy | paper does not clearly say whether one fixed engine split is reused across all seeds | both implemented | `--resample-split-per-seed` is closer to the paper wording |
| Dropout placement | paper table gives dropout rate, Algorithm 1 only shows output dropout | output dropout and inner Transformer dropout both supported | output-only dropout is the current best-supported interpretation |
| Weight decay | not published | supported | no paper-confirmed value |
| Gradient clipping | not published | supported | no paper-confirmed value |
| Mamba residual wrapper | not shown around the full block in Algorithm 1 | supported for testing | `bare` is the paper-closer and empirically better choice |
| External model width | paper's external encoder/decoder shape is `S=21`; internal Mamba uses `E` | `d_model` is configurable | paper-closer supervised reproduction keeps external width at `21` |

## Tier 3: Empirical Enhancement Items

These settings improved performance in the current codebase, but they
should be treated as "best empirical supervised baseline" rather than
"strict paper Table 2 reproduction".

| Item | Current best empirical choice | Why it is not strict paper reproduction |
|---|---|---|
| Transformer norm mode | `post` | paper is internally inconsistent |
| FFN hidden size | `84` | `d_ff` is not published |
| Validation windows | `--val-all-windows` | paper does not explicitly specify this |
| Learning rate | `5e-4` | paper Table 2 says `1e-3` |
| Weight decay | `1e-4` | paper does not publish weight decay |

## Experiment Status For The Main Ambiguities

### 1. Post-LN

This has been tested.

- old weaker setting:
  `d_ff=2048 + last-window validation`
  `post` was worse than `pre`
- current stronger setting:
  `d_ff=84 + val_all_windows`
  `post` became better than `pre`

Important results:

- fixed split, seed42:
  `post + d_ff=84 + val_all_windows`
  test RMSE `14.6280`
- fixed split, seeds 42,43,44:
  `post + d_ff=84 + val_all_windows`
  mean test RMSE `15.4355`

Sources:

- [ablation_fd001_summary_3seed.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_summary_3seed.json)
- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dff84_valall_postnorm/FD001/summary.json)
- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dff84_valall_postnorm_3seed/FD001/summary.json)

### 2. Transformer Inner Dropout = 0.5

This has also been tested.

Configuration:

- `post + d_ff=84 + val_all_windows + lr=5e-4 + weight_decay=1e-4 + transformer_inner_dropout=0.5`

Result:

- seed42 test RMSE `15.3951`

Compared to the same line with `transformer_inner_dropout=0.0`:

- seed42 test RMSE `14.7126`

Conclusion:

- applying `0.5` inside Transformer residual branches is worse than the
  current output-only dropout interpretation

Sources:

- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dff84_valall_postnorm_innerdrop05_lr5e4_wd1e4/FD001/summary.json)
- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dff84_valall_postnorm_lr5e4_wd1e4/FD001/summary.json)

## Recommended Working Definitions

Use these labels going forward:

### A. Strict Paper Table-2 Reproduction

Use only paper-explicit hyperparameters:

- `lr=1e-3`
- `dropout=0.5`
- `window_size=20`
- `batch_size=64`
- `d_state=16`
- `d_conv=8`
- `num_transformer_layers=3`
- `num_heads=7`
- `Adam`
- `epochs=50`
- engine-level `80/20` split

But still document the unresolved ambiguities:

- norm order
- `d_ff`
- validation window policy
- split reuse vs resampling

### B. Adopted Reproduction Choice

This means:

- keep all paper-explicit items fixed
- for paper-ambiguous items, choose one documented implementation and
  keep it stable across experiments

Current most reasonable adopted choice:

- external width `21`
- `mamba_block_mode=bare`
- `transformer_inner_dropout=0.0`
- `--resample-split-per-seed`

### C. Best Empirical Supervised Baseline

This is the strongest current supervised line before any later
self-supervised or cross-domain work:

- `transformer_norm_mode=post`
- `dim_feedforward=84`
- `val_all_windows=True`
- `lr=5e-4`
- `weight_decay=1e-4`

Current reference results:

- fixed split, seeds 42,43,44:
  mean test RMSE `15.0937`
- per-seed resampled split, seeds 42,43,44:
  mean test RMSE `15.3024`

## Practical Conclusion

The current codebase already matches the paper at the level of main
model structure, data pipeline, and published top-level hyperparameters.
What is still missing is not the broad architecture, but the exact
resolution of several under-specified implementation details.

So the technically correct statement is:

- we have a near-paper supervised implementation
- we do not yet have proof of exact author-level reproduction
- future reproduction work should target Tier 2 ambiguity resolution,
  not the Tier 1 backbone
