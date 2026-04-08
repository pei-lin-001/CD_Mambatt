# CD-MambAtt Project Status

Last updated: `2026-04-05`

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
> branch is now implemented and runnable on CUDA. A paper-style same-subset
> **self-supervised MambAtt pipeline is also now implemented**, with FD001 nearly
> matching the paper's one-shot SSL number, although the four-subset match is
> incomplete and that line is currently paused. The latest finding is that
> **the first hard semantic-SPD redesign underperformed, but the newer
> shared-head-anchored semantic variant with source semantic warmup is now
> showing the first real positive signal**, even though it still has not
> surpassed the older SPD v0 invariant-MMD reference on the current limited
> checks.

## 2. What is already completed

### 2.1 Infrastructure

- Conda environment: `cd_mamba`
- CUDA path validated in WSL2
- dataset prepared on `E:` and linked into WSL
- main runners available:
  - `train_supervised.py`
  - `train_self_supervised.py`
  - `train_cross_domain_baseline.py`
  - `train_cd_mambatt_v2.py`
  - `train_fomln_baseline.py`

### 2.2 Baseline reproduction

- target-paper supervised MambAtt baseline: **implemented and aligned at the structure level**
- target-paper same-subset self-supervised MambAtt path: **implemented and archived**
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

### 3.4 `DD-SSM / SPD` current best results (2026-04-04)

**Best config**: SPD (dd_spd) + inv-MMD (0.1) + spec-domain-predictive (0.1) +
Adam LR=2e-3 + CosineAnnealingLR(T_max=20, eta_min=1e-5) +
matched protocol (resample + val-all-windows)

#### Single-task result (FD001→FD003, 3 seeds):

- mean RMSE = **20.37 ± 0.47**
- per-seed: 42=21.04, 43=19.99, 44=20.09
- v2 reference: 21.13 ± 1.59
- **improvement: -0.76 mean, -70% variance**

#### Multi-task validation (3 seeds each):

| Task | SPD best | v2 ref (5-shot) | Delta | Status |
|---|---:|---:|---:|---|
| FD001→FD003 | **20.37 ± 0.47** | 21.96 | -1.59 | ✅ 赢 |
| FD001→FD004 | 23.72 ± 2.38 | 24.34 | -0.62 | ⚠️ 微赢，方差大 |
| FD003→FD001 | 21.03 ± 1.70 | 19.81 | **+1.22** | ❌ 输 |

#### Diagnostic findings:

- Mamba hidden state domain drift: **59× amplification** from step 1 to step 20
- SPD disentanglement is weak: inv vs combined domain accuracy Δ = 0.3%
- Gate barely moves (0.12→0.18), spec branch contributes ~15%
- SSDA (hidden state alignment) saturates immediately, insensitive to λ
- Spec domain-predictive loss was the most effective auxiliary loss
- Higher LR (2e-3 + cosine) was orthogonal to SPD and they stack

### 3.5 Semantic-SPD v1/v2 status (SUPERSEDED)

> **Note**: the semantic-SPD experiments below were all conducted under the
> mismatched protocol. Given that SPD v0 + inv-MMD already beats v2 with the
> correct protocol, the semantic-SPD direction is no longer the priority.
> These results are retained for reference only.

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

### 3.6 Shared-aux semantic-SPD + source semantic warmup

New code state:

- `spd_predictor_mode = shared_aux_residual`
- final prediction remains on the stable shared head
- invariant and specific heads are trained as semantic auxiliary branches
- a new `semantic_warmup` stage calibrates `inv_head/spec_head` on source data
  before cross-domain adaptation

Current key pilot results:

- shared-aux semantic-SPD without warmup (`2` seeds):
  - mean adapted RMSE = **24.5367**
- seed `43` with shared-aux + warmup:
  - **27.0563 -> 25.7429**
- seed `42` with shared-aux + warmup:
  - **22.0172 -> 21.9192**
- seed `43` with warmup + invariant MMD:
  - **25.7429 -> 25.7050**

Interpretation:

- source semantic warmup is the **first semantic-SPD modification that clearly
  improves the softer semantic branch**
- the main gain seems to come from **semantic grounding of the invariant
  branch**, not from simply increasing alignment pressure
- the warmup-only paired single-seed estimate is approximately:
  - **23.8310**
- that is still slightly weaker than the earlier SPD v0 invariant-MMD
  reference (**23.3498**), but it closes much of the gap

### 3.7 Same-subset self-supervised MambAtt reproduction snapshot (PAUSED)

Primary record:

- see [`self_supervised_reproduction.md`](./self_supervised_reproduction.md)

Current same-subset one-shot SSL snapshot:

| Subset | Paper Table 5 | Current local result | Note |
|---|---:|---:|---|
| `FD001` | 31.7270 | **31.6116 ± 2.5595** | 3 seeds, essentially matched |
| `FD002` | 30.3269 | **23.5244** | seed 42 only |
| `FD003` | 32.3329 | **35.5230 ± 2.6474** | 3 seeds |
| `FD004` | 32.6633 | **38.8006** | seed 42 only |

Implemented code for this branch:

- `cd_mambatt/self_supervised.py`
- `train_self_supervised.py`
- `cd_mambatt/models/mambatt.py` encoder-sequence export helpers

Current judgment:

- the SSL path is **implemented and reusable**
- FD001 is strong enough to validate the implementation direction
- FD003/FD004 remain off-paper
- therefore this branch is being **recorded and paused**, rather than pushed further right now

## 4. What we can and cannot claim right now

### Can claim

- the supervised MambAtt baseline has been reproduced at the **architecture / pipeline** level
- `CD-MambAtt v2` clearly improves over direct transfer on multiple C-MAPSS tasks
- reproduced `FOMLN` exists and current matched-shot numbers favor CD-MambAtt on 4 overlapping tasks
- `DD-SSM / SPD v0` is implemented inside the Mamba path and validated on CUDA
- **SPD v0 + inv-MMD beats v2 on both mean RMSE and cross-seed stability** on the canonical `FD001 -> FD003` task (3 seeds, matched protocol)
- **SPD provides a net -0.91 RMSE improvement** over v3+bare under the same protocol

### Cannot claim yet

- exact author-level reproduction of the target paper supervised number
- strict publication-grade apples-to-apples superiority over `FOMLN`
- SPD superiority across **multiple tasks** (only FD001→FD003 tested so far)
- SPD superiority at 5-seed scale (only 3 seeds so far)

## 5. Current bottlenecks

1. supervised reproduction gap to the target paper remains
2. ~~SPD innovation depth insufficient~~ → SPD is now the main innovation
3. ~~GRL unstable~~ → replaced with inv-MMD + spec-domain-predictive
4. ~~protocol mismatch~~ → resolved (v3 defaults fixed)
5. **SPD wins on FD001→FD003 and FD001→FD004 but loses on FD003→FD001** — need to diagnose task-direction asymmetry
6. **disentanglement effect is weak** (gate barely moves, domain separability Δ only 0.3%)
7. **only 3 seeds tested** — need 5 seeds for statistical confidence
8. `FOMLN` comparison not yet done with SPD
9. second dataset (XJTU-SY) not yet started
10. no ablation study yet under matched protocol

## 6. Next priority

Current recommended priority order:

1. **diagnose FD003→FD001 failure**: check if LR=2e-3 is too aggressive for small source domains (FD003 has only 100 engines); try LR=1e-3 on this task
2. **5-seed expansion** on FD001→FD003 with best config
3. **ablation study** under matched protocol: bare vs SPD vs SPD+inv-MMD vs SPD+inv-MMD+spec-domain
4. **run SPD on 15-shot** for fair comparison against FOMLN
5. explore **stronger disentanglement** mechanisms (current gate/separation is too weak)
6. second dataset (XJTU-SY)

## 7. Important files

- main experiment record: [`cross_domain_experiment_log.md`](./cross_domain_experiment_log.md)
- supervised consolidation: [`supervised_reproduction.md`](./supervised_reproduction.md)
- published baseline consolidation: [`published_baselines.md`](./published_baselines.md)
- innovation roadmap: [`dd_ssm_roadmap.md`](./dd_ssm_roadmap.md)
- generated tables: [`generated/formal_result_tables.md`](./generated/formal_result_tables.md)
