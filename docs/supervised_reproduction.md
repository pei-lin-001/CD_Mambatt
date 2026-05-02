# Supervised Reproduction Status

Last updated: `2026-04-13`

This file consolidates the old supervised-reproduction notes into one place.

## 1. Scope

We have reproduced the **supervised baseline path** of the target paper
`Mamba-attention: A self-supervised framework for efficient remaining useful life prediction`.

Important boundary:

- this file only tracks the paper's **supervised baseline reproduction**
- the paper's **self-supervised Section 3.4 path is now implemented separately**
- see [`self_supervised_reproduction.md`](./self_supervised_reproduction.md) for the current SSL code/result snapshot

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

### 5.4 Target-domain oracle control on `FD003`

To avoid over-attributing the canonical `FD001→FD003` ceiling to source-side
backbone weakness alone, we added a target-domain full-supervised control using
the same current best local supervised recipe:

- subset = `FD003`
- `transformer_norm_mode = post`
- `dim_feedforward = 84`
- `val_all_windows = True`
- `lr = 5e-4`
- `weight_decay = 1e-4`
- seeds `42,43,44`
- output:
  - `/home/shelterpl/cd_mambatt/runs/oracle_supervised_fd003_post_dff84_valall_lr5e4_wd1e4/FD003`
- generated summary:
  - `/home/shelterpl/cd_mambatt/docs/generated/mambatt_target_oracle_fd003_decomposition_2026-04-13.json`

Results:

- fixed-split mean test RMSE = **14.1128**
- fixed-split std test RMSE = **0.1932**

Why this matters:

- current local `FD001` supervised reproduction:
  - fixed split = **15.0937**
- current `FD001→FD003` cross-domain few-shot references:
  - stable v2 `5`-seed mean = **21.9608**
  - union-SSL `3`-seed optimistic mean = **19.8483**
  - plain no-spec `5`-seed mean = **20.8917**

Evidence-supported takeaway:

> The current canonical transfer ceiling cannot be explained only by the source-side supervised reproduction gap. Under the same reproduced MambAtt family, full target supervision on `FD003` already reaches about **14.11**, which is still far below every current `FD001→FD003` cross-domain few-shot result.

Important caveat:

- this is **not** a pure domain-shift estimate:
  - it also removes the few-shot label constraint
- therefore it should be read as:
  - a **target-domain oracle headroom control**
  - not as a direct decomposition of domain shift alone

### 5.5 Target-domain oracle control on `FD004`

We added the same target-domain full-supervised oracle on the harder
multi-condition subset `FD004`, again using the current best local supervised
recipe:

- subset = `FD004`
- `transformer_norm_mode = post`
- `dim_feedforward = 84`
- `val_all_windows = True`
- `lr = 5e-4`
- `weight_decay = 1e-4`
- seeds `42,43,44`
- output:
  - `/home/shelterpl/cd_mambatt/runs/oracle_supervised_fd004_post_dff84_valall_lr5e4_wd1e4/FD004`
- generated summary:
  - `/home/shelterpl/cd_mambatt/docs/generated/mambatt_target_oracle_fd004_decomposition_2026-04-13.json`

Results:

- fixed-split mean test RMSE = **16.4495**
- fixed-split std test RMSE = **0.0096**

Why this matters:

- current `FD001→FD004` cross-domain references:
  - stable v2 `5`-seed mean = **24.3449**
  - SPD high-LR `3`-seed mean = **23.72**
- the target oracle is still roughly **7.3 to 7.9 RMSE** below those transfer
  results

Evidence-supported takeaway:

> Even on the harder `FD004` target, the current reproduced MambAtt family can
> reach about **16.45** under full target supervision, so the present
> `FD001→FD004` transfer ceiling still leaves substantial headroom beyond the
> current few-shot adaptation line.

Important caveat:

- this is again **not** a pure domain-shift estimate:
  - it also removes the few-shot label constraint
- `FD004` itself is harder than `FD001` under the current reproduced recipe:
  - so this control should be used to measure remaining headroom on the target
    task, not to compare subset difficulty and transfer penalty with one number

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
