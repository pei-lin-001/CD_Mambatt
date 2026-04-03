# CD-MambAtt Project Status

Last updated: `2026-04-03`

## 1. Current phase

The project is no longer in environment setup. It is now in the
**baseline-reproduction + cross-domain experimentation + next-innovation design** stage.

Current status in one paragraph:

> Environment, data, codebase, and CUDA workflow are ready. The supervised MambAtt
> baseline is structurally aligned to the target paper but still numerically above the
> reported paper RMSE. A working cross-domain `CD-MambAtt v2` package exists and has
> been tested on multiple C-MAPSS transfer pairs. A reproduced `FOMLN` baseline also
> exists, but the current head-to-head comparison should be described as
> **matched-shot in-house comparison with remaining protocol mismatches**, not as a
> fully strict apples-to-apples benchmark. The architecture-level `DD-SSM / SPD`
> branch is now implemented and runnable on CUDA. The latest finding is that
> **the first novelty-driven semantic-SPD redesign is conceptually stronger but
> empirically underperforms the earlier SPD v0 invariant-MMD reference**, so the
> next step should be a softer semantic decomposition rather than another blind
> scalar sweep.

## 2. What is already completed

### 2.1 Infrastructure

- Conda environment: `cd_mamba`
- CUDA path validated in WSL2
- dataset prepared on `E:` and linked into WSL
- main runners available:
  - `train_supervised.py`
  - `train_cross_domain_baseline.py`
  - `train_cd_mambatt_v2.py`
  - `train_fomln_baseline.py`

### 2.2 Baseline reproduction

- target-paper supervised MambAtt baseline: **implemented and aligned at the structure level**
- cross-domain `CD-MambAtt v2`: **implemented and working**
- reproduced `FOMLN` baseline: **implemented and runnable on CUDA**

## 3. Key current results

### 3.1 Supervised target-paper reproduction

Paper reference points:

- Table 3 supervised RMSE on `FD001`: **11.46**
- Table 7 Transformer-decoder average: **12.34** (`5` runs average in the paper)

Current local status:

- best current empirical fixed-split supervised line (`post + d_ff=84 + val_all_windows + lr=5e-4 + wd=1e-4`, seeds `42,43,44`):
  - mean test RMSE = **15.0937**
- same empirical line under per-seed resampled split (`42,43,44`):
  - mean test RMSE = **15.3024**

Conclusion:

- the supervised backbone is **structurally reproduced**
- the numerical gap to the paper is **still open**
- remaining mismatch is more likely hidden protocol / unpublished implementation detail than a missing top-level module

### 3.2 Cross-domain `CD-MambAtt v2` (`5-shot`)

Current main working configuration:

- `lambda_mmd = 0.1`
- `lambda_source_stage = 1.0`
- `lambda_pseudo = 0.5`
- `lambda_monotonic = 0.05` or `0.1`
- `lambda_contrastive = 0.0`
- pseudo quantile schedule = `0.5 -> 0.9`

Multi-task snapshot (`5 seeds`, mean RMSE):

| Task | Direct transfer | CD-MambAtt | Gain |
|---|---:|---:|---:|
| `FD001 -> FD003` | 34.8204 | **21.9608** | 12.8595 |
| `FD003 -> FD001` | 24.6059 | **19.8137** | 4.7922 |
| `FD002 -> FD004` | 29.5028 | **22.1776** | 7.3252 |
| `FD001 -> FD004` | 32.3820 | **24.3449** | 8.0371 |
| `FD004 -> FD002` | 21.3666 | **19.5311** | 1.8354 |
| `FD001 -> FD002` | 31.1029 | **22.9457** | 8.1572 |

Interpretation:

- the cross-domain route is real; gains over direct transfer are no longer limited to one task
- current method strength mainly comes from `MMD + pseudo + monotonic`
- current method weakness is that the innovation still lives mostly at the feature/loss level, not inside Mamba itself

### 3.3 Matched-shot comparison against reproduced `FOMLN` (`15-shot`)

| Task | CD-MambAtt 15-shot | FOMLN 15-shot | Delta (CD - FOMLN) |
|---|---:|---:|---:|
| `FD001 -> FD003` | **20.1384** | 25.4748 | -5.3364 |
| `FD003 -> FD001` | **19.5788** | 20.1261 | -0.5474 |
| `FD002 -> FD004` | **23.2879** | 26.0206 | -2.7327 |
| `FD001 -> FD004` | **23.9469** | 25.0808 | -1.1340 |

Negative delta means CD-MambAtt is better.

### 3.4 `DD-SSM / SPD v0` status on canonical `FD001 -> FD003`

Current best SPD single-seed runs:

- `inv_alignment_mode = mmd`, `domain_feature_tap = inv_mean`, `lambda_inv_mmd = 0.2`, seed `42`:
  - target RMSE = **20.8015**

Current multi-seed check:

- SPD v0 + GRL (`seeds = 42,43,44`):
  - mean target RMSE = **24.0269**
  - std = **2.3729**
- SPD v0 + invariant-path MMD (`seeds = 42,43,44`):
  - best current configuration:
    - `lambda_inv_mmd = 0.1`
    - `lambda_mmd = 0.1`
  - mean target RMSE = **23.3498**
  - std = **2.3153**
- SPD v0 + selective freezing `spec_gate_head` (`seeds = 42,43`):
  - mean target RMSE = **27.0818**
  - std = **1.7057**
- SPD v0 + selective freezing `spec_gate_transformer_head`:
  - `2-seed` pilot (`42,43`):
    - mean target RMSE = **23.0026**
    - std = **2.4471**
  - `3-seed` full run (`42,43,44`):
    - mean target RMSE = **24.1388**
    - std = **1.5378**
- SPD v0 + `freeze 5 epochs -> full unfreeze` (`spec_gate_transformer_head`, `seeds = 42,43`):
  - mean target RMSE = **23.6333**
  - std = **2.1958**
- SPD v0 + stronger invariant-path MMD (`lambda_inv_mmd = 0.2`, `seeds = 42,43,44`):
  - mean target RMSE = **23.3726**
  - std = **2.3401**
- `CD-MambAtt v2` matched `3-seed` reference:
  - mean target RMSE = **21.1291**
  - std = **1.5928**

Interpretation:

- the SPD implementation is real and can be competitive on a good seed
- replacing GRL with invariant-path MMD improves SPD
- but the current SPD branch is **still not yet strong enough** to beat v2
  under multi-seed evaluation
- scalar loss-weight scans have mostly saturated:
  - removing or weakening the original global MMD did not help
  - increasing `lambda_inv_mmd` to `0.2` did not improve the `3-seed` mean
- selective freezing was tested and is currently **not sufficient**:
  - very restrictive freezing (`spec_gate_head`) collapses badly
  - broader freezing (`spec_gate_transformer_head`) reduces variance but hurts
    the `3-seed` mean
  - staged unfreezing approximately ties the non-freeze baseline on `2` seeds
    but does not beat it
- the next likely issue remains architectural:
  the inv/spec role separation may still be too weak in the current adaptation protocol

### 3.5 Semantic-SPD v1/v2 status on canonical `FD001 -> FD003`

Purpose of this branch:

- move beyond generic feature-level alignment
- test the novelty-safe direction from `innovation_novelty_assessment.md`
- make the invariant path semantically responsible for the main RUL trend and
  restrict the specific path to residual correction

Implemented changes:

- `spd_predictor_mode = decomposed_residual`
- invariant prediction head `inv_head`
- specific residual head `spec_head`
- stage-conditional invariant MMD
- invariant/specific orthogonality regularization
- specific residual magnitude regularization

Current semantic-SPD results:

- semantic-SPD v1 (`2` seeds, direct decomposed predictor):
  - mean direct RMSE = **47.1927**
  - mean adapted RMSE = **26.2643**
- semantic-SPD v2 (`2` seeds, source-stage shared-head warm start):
  - mean direct RMSE = **42.4143**
  - mean adapted RMSE = **25.0020**
- semantic-SPD v3 probe (`1` seed, add back global MMD):
  - direct RMSE = **45.7994**
  - adapted RMSE = **25.9911**
- earlier SPD v0 invariant-MMD reference (`3` seeds):
  - mean adapted RMSE = **23.3498**

Interpretation:

- the semantic redesign is **numerically stable**
- the shared-head warm-start is the right stabilization move
- but the current fully decomposed predictor is still **too disruptive**
- the specific branch becomes too strong and the gate stays too large
- simply restoring the old global MMD does **not** rescue this branch
- therefore, the next semantic-SPD iteration should keep the novelty direction
  but adopt a **softer decomposition protocol**

## 4. What we can and cannot claim right now

### Can claim

- the supervised MambAtt baseline has been reproduced at the **architecture / pipeline** level
- `CD-MambAtt v2` clearly improves over direct transfer on multiple C-MAPSS tasks
- reproduced `FOMLN` exists and current matched-shot numbers favor CD-MambAtt on 4 overlapping tasks
- `DD-SSM / SPD v0` is implemented inside the Mamba path and validated on CUDA
- semantic-SPD code path is implemented and trainable on CUDA

### Cannot claim yet

- exact author-level reproduction of the target paper supervised number
- strict publication-grade apples-to-apples superiority over `FOMLN`
- stable architecture-level superiority from the current `SPD v0` branch
- empirical superiority from the first semantic-SPD redesign

## 5. Current bottlenecks

1. supervised reproduction gap to the target paper remains
2. current `CD-MambAtt v2` innovation depth is still insufficient for a strong paper
3. the original GRL-based SPD branch is unstable across seeds
4. after replacing GRL with MMD, SPD is more stable but still not clearly better than v2
5. the first semantic-SPD redesign proves the novelty direction is implementable,
   but its predictor decomposition is too aggressive and hurts performance
6. simple loss-weight tuning around the current SPD scaffold shows diminishing returns
7. simple selective-freezing / warm-start freeze schedules do not yet produce a net gain
8. `FOMLN` comparison is useful but still not perfectly fair
9. `MetaDFKN` remains a reported threat, but protocol ambiguity is not yet cleaned enough for fair reproduction

## 6. Next priority

Current recommended priority order:

1. **revise semantic-SPD into a softer decomposition**:
   keep the novelty claim inside Mamba, but reduce disruption by preserving the
   stable shared prediction route and adding semantic auxiliary constraints or a
   bounded residual path instead of a hard predictor replacement
2. after SPD is stable, add **SSDA** if needed
3. treat **DAAD** as an auxiliary enhancement, not a co-equal main innovation
4. baseline track, if resumed, should target a cleaner `MetaDFKN` protocol audit before coding

## 7. Important files

- main experiment record: [`cross_domain_experiment_log.md`](./cross_domain_experiment_log.md)
- supervised consolidation: [`supervised_reproduction.md`](./supervised_reproduction.md)
- published baseline consolidation: [`published_baselines.md`](./published_baselines.md)
- innovation roadmap: [`dd_ssm_roadmap.md`](./dd_ssm_roadmap.md)
- generated tables: [`generated/formal_result_tables.md`](./generated/formal_result_tables.md)
