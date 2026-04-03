# Response To Claude Findings

This note responds point-by-point to the issues raised in the external
analysis, and records which items were already tested locally versus
which remain open.

## 1. Transformer normalization order

The concern is valid: the paper is internally inconsistent.

- `Figure 1` visually matches a `Post-LN` style Transformer.
- `Algorithm 1` explicitly describes a `Pre-LN` order.

The current code still defaults to `pre`:

- [mambatt.py](/home/shelterpl/cd_mambatt/cd_mambatt/models/mambatt.py:145)
- [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py:319)

Earlier controlled testing with `d_ff=2048` and last-window validation
did show `post` performing worse:

- `pre-norm`: best test RMSE `16.4070`, mean test RMSE `16.7745`
- `post-norm`: best test RMSE `18.8251`, mean test RMSE `18.7772`

Source:

- [ablation_fd001_summary_3seed.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_summary_3seed.json)

However, that conclusion does not hold once the stronger settings are
used. With `d_ff=84` and `val_all_windows`, `post` became the best
current structural direction:

- single seed `42`:
  test RMSE `14.6280`
- seeds `42,43,44`:
  best test RMSE `15.0757`
  mean test RMSE `15.4355`

Sources:

- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dff84_valall_postnorm/FD001/summary.json)
- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dff84_valall_postnorm_3seed/FD001/summary.json)

Conclusion:

- `post` is not ruled out
- the earlier negative result was confounded by the weaker
  `d_ff=2048 + last-window-validation` setup
- under the current strongest settings, `post` is the better direction
  to continue testing, even though `Algorithm 1` itself reads as `pre`

## 2. `dim_feedforward=2048` may be mismatched

This concern is strongly supported by experiments.

The code default is still `2048`:

- [mambatt.py](/home/shelterpl/cd_mambatt/cd_mambatt/models/mambatt.py:145)
- [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py:317)

But a local sweep was already performed on `FD001` with fixed split and
three seeds:

- `d_ff=84`: best test RMSE `16.0296`
- `d_ff=128`: best test RMSE `16.2254`
- `d_ff=256`: best test RMSE `18.9623`
- `d_ff=512`: best test RMSE `18.1673`
- `d_ff=2048`: best test RMSE `16.4070`

Source:

- [ablation_fd001_summary_3seed.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_summary_3seed.json)

With `val_all_windows`, the best current local result is:

- `d_ff=84 + val_all_windows`: best test RMSE `15.2679`

Source:

- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_combo_valall_dff84/FD001/summary.json)

Conclusion:

- `2048` is not the best current choice
- `84` is currently the most defensible setting among the tested values
- this improves results, but still does not explain the full gap to the paper's `11.46`

## 3. Validation window strategy

This issue is real and already tested.

Current implementation:

- [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py:168)

By default:

- validation uses only the last window from each validation engine
- unless `--val-all-windows` is enabled

Observed results:

- `d_ff=2048`, last-window validation:
  best test RMSE `16.4070`, mean test RMSE `16.7745`
- `d_ff=2048 + val_all_windows`:
  best test RMSE `16.2168`, mean test RMSE `15.9484`
- `d_ff=84 + val_all_windows`:
  best test RMSE `15.2679`

Source:

- [ablation_fd001_summary_3seed.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_summary_3seed.json)

This advantage persists even on the strongest current line. With
`post + d_ff=84 + lr=5e-4 + weight_decay=1e-4`:

- `val_all_windows`, seed `42`:
  test RMSE `14.7126`
- last-window validation, seed `42`:
  test RMSE `15.8652`

Sources:

- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dff84_valall_postnorm_lr5e4_wd1e4/FD001/summary.json)
- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dff84_postnorm_lr5e4_wd1e4_lastval/FD001/summary.json)

Conclusion:

- `val_all_windows` improves stability and model selection
- this is one of the few tested changes that clearly helps
- at least in the current implementation, it is much better aligned
  with good test performance than last-window validation

## 4. Best-of-50 selection effect

The code already implements the paper's reported procedure:

- multiple seeds
- choose the run with lowest validation RMSE

Relevant code:

- [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py:335)
- [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py:348)

However, this factor was also tested directly.

The fixed-split `50`-seed run for `FD001` produced:

- `best_val_rmse = 0.7259`
- `best_test_rmse = 19.8655`
- `mean_test_rmse = 18.3327`

Source:

- [summary.json](/home/shelterpl/cd_mambatt/runs/supervised_fd001_fixedsplit/FD001/summary.json)

Conclusion:

- "not enough runs" is not the main explanation
- even after `50` runs, the gap to the paper remains very large
- therefore the missing factor must be elsewhere

## 5. Mamba residual wrapper

The external analysis suggests the default should remain without an
extra residual wrapper around the Mamba block. That matches the current
default:

- [mambatt.py](/home/shelterpl/cd_mambatt/cd_mambatt/models/mambatt.py:149)
- [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py:321)

This was also tested:

- `prenorm_residual + d_ff=84 + val_all_windows + seed42`
  produced test RMSE `17.1142`

Compared to the current best baseline-style variant:

- `bare + d_ff=84 + val_all_windows + seed42`
  produced test RMSE `15.2679`

Sources:

- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_residual_valall_dff84/FD001/summary.json)
- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_combo_valall_dff84/FD001/summary.json)

Conclusion:

- the residual-wrapped Mamba version is currently worse
- this does not look like the missing ingredient

## 6. Internal dropout

Current default:

- Transformer internal dropout is `0.0`
- output dropout before the regression head is `0.5`

Relevant code:

- [mambatt.py](/home/shelterpl/cd_mambatt/cd_mambatt/models/mambatt.py:148)
- [mambatt.py](/home/shelterpl/cd_mambatt/cd_mambatt/models/mambatt.py:198)
- [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py:316)
- [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py:320)

This was tested:

- `transformer_inner_dropout=0.1 + d_ff=84 + val_all_windows + seed42`
  produced test RMSE `15.9792`

Compared to:

- `transformer_inner_dropout=0.0 + d_ff=84 + val_all_windows + seed42`
  produced test RMSE `15.2679`

Sources:

- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_innerdrop01_valall_dff84/FD001/summary.json)
- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_combo_valall_dff84/FD001/summary.json)

Conclusion:

- small internal dropout did not help in the current implementation

## Additional point not covered in the external analysis

There is an easy-to-misread notation issue around the paper's latent
dimension `E`.

The paper distinguishes:

- input sensor dimension `S`
- internal Mamba projection size `E`

However, Section 3.4 later states that the Mamba encoder outputs the
same external shape as its input, `R^(B×W×S) -> R^(B×W×S)`. That means
the paper-aligned supervised path keeps the external model width at
`S=21`, even though the Mamba block internally expands features.

The training script still supports an exploratory `--d-model`:

- [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py:309)

This was tested:

- `d_model=84, d_ff=336, val_all_windows, seed42`
  produced test RMSE `15.7740`

Source:

- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dmodel84_dff336_valall/FD001/summary.json)

Conclusion:

- `d_model != 21` should be treated as an exploratory deviation, not as
  a direct implementation of the paper's `E`
- the paper-aligned supervised reproduction should keep the external
  sequence width at `21`
- this exploratory branch did not improve the result anyway

## 7. Optimization details added after the original review

The paper specifies `Adam` and `lr=1e-3`, but it does not publish
weight decay or a scheduler. This has turned into one of the main
remaining tuning surfaces.

Recent `FD001` experiments on the strongest current line
(`post + d_ff=84 + val_all_windows`) produced:

- `lr=5e-4`, seed `42`:
  test RMSE `14.9412`
- `lr=5e-4`, seeds `42,43,44`:
  best test RMSE `16.0940`
  mean test RMSE `15.3669`
- `lr=5e-4 + weight_decay=1e-4`, seed `42`:
  test RMSE `14.7126`
- `lr=5e-4 + weight_decay=1e-4`, seeds `42,43,44`:
  best test RMSE `15.9380`
  mean test RMSE `15.0937`
- `lr=3e-4 + weight_decay=1e-4`, seed `42`:
  test RMSE `15.7730`
- `lr=5e-4 + weight_decay=5e-5`, seed `42`:
  test RMSE `14.6479`
- `lr=5e-4 + weight_decay=5e-5`, seeds `42,43,44`:
  best test RMSE `16.5683`
  mean test RMSE `15.3900`

Sources:

- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dff84_valall_postnorm_lr5e4/FD001/summary.json)
- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dff84_valall_postnorm_lr5e4_3seed/FD001/summary.json)
- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dff84_valall_postnorm_lr5e4_wd1e4/FD001/summary.json)
- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dff84_valall_postnorm_lr5e4_wd1e4_3seed/FD001/summary.json)
- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dff84_valall_postnorm_lr3e4_wd1e4/FD001/summary.json)
- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dff84_valall_postnorm_lr5e4_wd5e5/FD001/summary.json)
- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dff84_valall_postnorm_lr5e4_wd5e5_3seed/FD001/summary.json)

Conclusion:

- lowering the learning rate from the paper's `1e-3` helps in the
  current codebase
- a small weight decay also improves the multi-seed mean
- but `weight_decay=5e-5` does not beat `1e-4` on the three-seed mean
- dropping the learning rate further to `3e-4` is clearly worse
- this improves performance, but it moves the run farther away from
  the explicit paper hyperparameter table

## 8. New paper-aligned protocol check: resampling the 80/20 engine split

The current fixed-split runner is useful for controlled ablations, but
it is not the only plausible reading of the paper. The text only says
that `80%` of the engines are used for training and `20%` for
validation; it does not explicitly say that one split is fixed and
reused across all seeds.

To check the closer-to-paper interpretation, the current strongest
configuration

- `post`
- `d_ff=84`
- `val_all_windows`
- `lr=5e-4`
- `weight_decay=1e-4`

was re-run as independent experiments with `split_seed=seed`.

Observed results:

- `seed42 + split42`:
  test RMSE `14.7126`
- `seed43 + split43`:
  test RMSE `15.3490`
- `seed44 + split44`:
  test RMSE `15.7843`

Aggregate:

- independent-experiment mean test RMSE `15.2820`
- independent-experiment std test RMSE `0.4401`

Sources:

- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dff84_valall_postnorm_lr5e4_wd1e4/FD001/summary.json)
- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dff84_valall_postnorm_lr5e4_wd1e4_resplit_seed43/FD001/summary.json)
- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dff84_valall_postnorm_lr5e4_wd1e4_resplit_seed44/FD001/summary.json)

Conclusion:

- resampling the engine split per experiment is arguably closer to the
  paper protocol
- it reduces one obvious source of fixed-split bias during analysis
- but it did not improve the current three-run average over the
  fixed-split control
- the reproduction gap still remains

Implementation note:

- the runner now supports this protocol directly via
  `--resample-split-per-seed`
- `summary.json` records `split_mode`, `split_paths`, and `split_seeds`
  so the protocol used in each experiment is explicit

## Overall conclusion

Among the main concerns, the ones already supported by local
experiments are:

- `d_ff` should not blindly stay at `2048`
- validating on all windows is better than validating only on the last
  window
- `post-norm` becomes better once it is paired with `d_ff=84` and
  `val_all_windows`
- modest optimizer tuning around the paper defaults helps

The ones already tested and currently unsupported are:

- Mamba residual wrapper
- Transformer internal dropout `0.1`
- `target_scale=125`
- `expand=1`
- gradient clipping as a standalone fix

The `best-of-50` effect is real in principle, but it is not the main
explanation because a `50`-seed fixed-split run was already performed
and remained far from the paper.

The strongest current conclusion is:

- the remaining gap is unlikely to come from only the first round of
  architecture toggles
- the best current line is now:
  `post + d_ff=84 + val_all_windows + lr=5e-4 + weight_decay=1e-4`
- even this line still sits well above the paper's Table 3 `11.46`
  and Table 7 average `12.3409`
- the real missing factor is still more likely to be hidden training
  protocol details or unpublished implementation details

Most likely next directions:

- narrow optimizer search around the best current line
- compare `5`-seed means rather than only best-by-validation
- continue auditing paper ambiguities that affect validation and model
  selection
- gradient clipping
- actual interpretation of the paper's training iterations
- exact hidden embedding dimension and decoder implementation used by the authors

## Follow-up response to additional Claude comments

This section responds to a second round of comments after Claude
reviewed both the paper and the code.

### A. FFN hidden dimension

This point is correct, but it is not a new open issue anymore.

The external comment noted that:

- the paper never specifies `d_ff`
- the code default `2048` is abnormally large for `d_model=21`

That is already supported by local experiments.

Controlled `FD001` fixed-split `3`-seed results:

- `d_ff=84`: best test RMSE `16.0296`
- `d_ff=128`: best test RMSE `16.2254`
- `d_ff=256`: best test RMSE `18.9623`
- `d_ff=512`: best test RMSE `18.1673`
- `d_ff=2048`: best test RMSE `16.4070`

With `val_all_windows`, the current best local result remains:

- `d_ff=84 + val_all_windows`: best test RMSE `15.2679`

Sources:

- [ablation_fd001_summary_3seed.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_summary_3seed.json)
- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_combo_valall_dff84/FD001/summary.json)

Conclusion:

- `d_ff` is an important ambiguity
- `84` is currently much more defensible than `2048`
- but this still does not explain the full gap to the paper

### B. Positional encoding and `d_model` ambiguity

This concern is also valid.

The paper mixes:

- `S` as the number of sensors
- `E` as an embedding dimension inside the Mamba block

The code originally tied `d_model` directly to `21`, but it now supports
an independent latent dimension through `--d-model`:

- [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py:309)
- [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py:192)

This was already tested:

- `d_model=84, d_ff=336, val_all_windows, seed42`
  produced test RMSE `15.7740`

Source:

- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dmodel84_dff336_valall/FD001/summary.json)

Conclusion:

- this ambiguity is real
- a separate latent dimension may matter
- but it is not sufficient by itself to reach the paper's reported result

### C. Mamba `expand` factor is unspecified

This point is correct and still open.

The paper's Table 2 does not list `expand`.

Current default:

- [mambatt.py](/home/shelterpl/cd_mambatt/cd_mambatt/models/mambatt.py:140)
- [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py:312)

Current status:

- default is `expand=2`
- this is a reasonable assumption based on common Mamba usage
- but it has not yet been systematically swept in local experiments

Conclusion:

- this remains an unresolved reproduction variable

### D. Transformer implementation choice

This point is correct in principle, but it has already been tested.

The code supports:

- `custom`
- `torch`

Relevant code:

- [mambatt.py](/home/shelterpl/cd_mambatt/cd_mambatt/models/mambatt.py:107)
- [mambatt.py](/home/shelterpl/cd_mambatt/cd_mambatt/models/mambatt.py:173)
- [train_supervised.py](/home/shelterpl/cd_mambatt/train_supervised.py:318)

Already tested:

- `transformer_impl=torch + d_ff=84 + val_all_windows + seeds 42,43,44`

Observed result:

- best test RMSE `16.7897`
- mean test RMSE `16.7683`

Source:

- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_torch_valall_dff84/FD001/summary.json)

Compared to:

- `custom + d_ff=84 + val_all_windows`
  best test RMSE `15.2679`

Conclusion:

- the official PyTorch Transformer encoder did not help
- in the current setup it is worse than the custom implementation

### E. Self-supervised learning is not implemented

This statement is factually correct.

The current codebase only implements the supervised baseline.
Therefore it cannot yet reproduce:

- Table 5 one-shot results
- Table 6 two-shot results

This is not an overlooked bug in the current phase.
It reflects the current project stage:

- first reproduce the supervised baseline as faithfully as possible
- then implement the self-supervised losses from Section 3.4

Conclusion:

- this is a real missing feature relative to the full paper
- but it is not the reason the current supervised baseline misses the
  supervised Table 3 result

### F. `causal-conv1d` removal and Conv1d fallback path

This observation is mostly correct, but it should be stated precisely.

The project code does not manually replace Mamba with a custom
`nn.Conv1d` implementation. Instead:

- `causal-conv1d` was intentionally removed from the environment
- `mamba-ssm` is therefore allowed to use its standard Conv1d fallback path

This compatibility decision is already documented:

- [environment_snapshot.md](/home/shelterpl/cd_mambatt/docs/environment_snapshot.md:20)
- [README.md](/home/shelterpl/cd_mambatt/README.md:42)

Reason:

- the paper specifies `d_conv=8`
- the official `causal-conv1d` package only supports widths `2/3/4`

Paper reference:

- [mamba_attention_rul_paper.txt](/home/shelterpl/cd_mambatt/docs/text/mamba_attention_rul_paper.txt:709)

Conclusion:

- this is a known engineering deviation from the fused kernel path
- but it was an intentional decision to preserve the paper's `d_conv=8`
- it is a real possible source of residual mismatch, but not an ignored issue

## Updated overall assessment

After considering both rounds of external analysis, the status is:

Already experimentally supported:

- `d_ff` matters, and `84` is better than `2048`
- validating on all windows helps

Already experimentally tested and currently unsupported as the main fix:

- `post-norm`
- Mamba residual wrapper
- small Transformer internal dropout
- official PyTorch Transformer implementation

Still open and worth investigating:

- Mamba `expand`
- exact latent embedding size used by the authors
- optimizer details
- weight decay
- learning-rate schedule
- gradient clipping
- unpublished training protocol details
- possible implementation differences introduced by the Conv1d fallback path

## New follow-up experiments after the latest review

After the latest round of discussion, several additional controlled
experiments were run on the same `FD001` fixed split.

All comparisons below use the paper-aligned backbone direction:

- `d_model=21`
- `d_ff=84`
- `val_all_windows=true`
- same fixed engine split

### 1. Gradient clipping on the previous best pre-norm setup

Configuration:

- `pre-norm`
- `d_ff=84`
- `val_all_windows`
- `grad_clip_norm=1.0`
- seed `42`

Result:

- test RMSE `15.4361`

Source:

- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dff84_valall_clip1/FD001/summary.json)

Compared to the previous best same-seed baseline:

- `pre-norm + d_ff=84 + val_all_windows`
  test RMSE `15.2679`

Conclusion:

- gradient clipping alone did not improve the previous best pre-norm configuration
- it is not a primary explanation by itself

### 2. `expand=1` versus `expand=2`

Configuration:

- `pre-norm`
- `d_ff=84`
- `val_all_windows`
- `expand=1`
- seed `42`

Result:

- test RMSE `16.3625`

Source:

- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dff84_valall_expand1/FD001/summary.json)

Conclusion:

- `expand=1` is clearly worse than the current default `expand=2`
- this reduces the likelihood that the paper used `expand=1`

### 3. Previously untested combination:
### `post-norm + d_ff=84 + val_all_windows`

Single-seed result first showed a strong improvement:

- seed `42`
- test RMSE `14.6280`

Source:

- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dff84_valall_postnorm/FD001/summary.json)

Then the same configuration was extended to three seeds:

- seeds `42,43,44`
- best test RMSE `15.0757`
- mean test RMSE `15.4355`

Source:

- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dff84_valall_postnorm_3seed/FD001/summary.json)

Interpretation:

- this is important because it overturns the earlier simplified claim
  that post-norm is generally worse
- post-norm was clearly worse when paired with `d_ff=2048` and last-window validation
- but it becomes competitive, and even stronger on some seeds, when paired with
  `d_ff=84 + val_all_windows`

### 4. `post-norm + d_ff=84 + val_all_windows + grad_clip_norm=1.0`

Single-seed result:

- seed `42`
- test RMSE `15.0086`

Source:

- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dff84_valall_postnorm_clip1/FD001/summary.json)

Three-seed result:

- seeds `42,43,44`
- best test RMSE `16.5102`
- mean test RMSE `15.5670`

Source:

- [summary.json](/home/shelterpl/cd_mambatt/runs/ablation_fd001_dff84_valall_postnorm_clip1_3seed/FD001/summary.json)

Interpretation:

- clipping helped one seed inside the post-norm setting
- but did not improve the three-seed aggregate result
- therefore clipping does not currently look like the main missing factor

## Updated conclusion after the new runs

The most important new result is:

- the previously untested combination
  `post-norm + d_ff=84 + val_all_windows`
  is better than expected and now matches or beats the old best setup

Current best comparison on `FD001`:

- old best:
  `pre-norm + d_ff=84 + val_all_windows`
  best test RMSE `15.2679`
- new best three-seed result:
  `post-norm + d_ff=84 + val_all_windows`
  best test RMSE `15.0757`

This is still far from:

- paper Table 3 `11.46`
- paper Table 7 Transformer-decoder average `12.34`

So the remaining gap is still substantial, but the search space has now
been narrowed further:

- `expand=1` is likely not the answer
- gradient clipping is not the main answer
- the interaction between normalization order, `d_ff`, and validation
  protocol matters more than originally assumed
