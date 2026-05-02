# CD-MambAtt Project Status

Last updated: `2026-04-13`

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
> incomplete and that line is currently paused. The `2026-04-09` union-SSL
> `FD001→FD003` result was a real positive **3-seed** signal, but the
> `2026-04-12` extension to `5` seeds changed the aggregate picture:
> plain no-spec measures **20.89 ± 2.28**, while union SSL full adaptation measures
> **21.83 ± 3.11**, so robust superiority is **not** supported in its original form.
> Direct diagnostics on the failing `seed 46` show that SSL still improves direct
> transfer there, but enters adaptation in a worse regime from epoch `1`, with
> smaller gate values and higher invariant-MMD throughout the run. Follow-up probes
> then showed two additional facts: SSL is **not** worse on the few-shot labeled
> target windows at epoch `1`, and removing unlabeled adaptation losses only
> **partially** rescues the bad seed (`26.43 → 24.85`, still above the baseline
> probe `19.07`). A later selective-freeze sweep established a more concrete
> stabilization result: keeping the shared Mamba core frozen while adapting the
> spec/gate branch plus Transformer/head reduces the union-SSL `5`-seed mean from
> **21.83 ± 3.11** to **20.90 ± 2.03**. That nearly ties the plain no-spec mean
> (**20.89 ± 2.28**) and materially lowers variance, but it still does **not**
> create a new canonical best. That means the immediate research focus should remain
> **adaptation-dynamics diagnosis / stabilization**, not adding more loss terms. As
> of `2026-04-12`, the maintained `train_cd_mambatt_v3.py` objective has also been
> consolidated to the verified loss core; contrastive, GRL/domain-adversarial, and
> semantic-SPD auxiliary loss branches are archived rather than kept in the main
> training path. A fresh hard-task validation on `FD003→FD001` then sharpened the
> SSL conclusion further: the canonical selective-freeze candidate
> (`spec_gate_transformer_head`) does **not** generalize as a new default, with a
> `3`-seed mean of **20.67 ± 2.16** versus **20.63 ± 1.60** for original SSL and
> **20.25 ± 0.65** for the plain no-spec baseline. Additional `seed 44` rescue
> probes showed that neither lower-LR full adaptation nor lighter
> `transformer_head` adaptation fixes that hard-task failure. The immediate focus
> should therefore be **task-conditional adaptation dynamics diagnosis** and the
> longer-standing **source-backbone gap**, not broader rollout of the current
> freeze policy. The supervision-side decomposition also moved forward on
> `2026-04-13`: target-domain oracle controls now exist on both `FD003` and
> `FD004` under the current best reproduced MambAtt recipe. Those controls show
> that full target supervision reaches **14.11 ± 0.19** on `FD003` and
> **16.45 ± 0.01** on `FD004`, both still far below the corresponding current
> cross-domain few-shot transfer lines. That means the source-backbone gap is a
> real issue, but it is **not sufficient** to explain the whole transfer ceiling;
> there is still a large few-shot / adaptation headroom gap relative to target
> oracles. That follow-up control has now also been refined into a three-way
> decomposition. On `FD001→FD003`, matched target-only `5-shot` scratch is
> **22.47 ± 0.57**, source-init-only supervised finetune is
> **21.94 ± 0.75**, and the full current no-spec CD pipeline is
> **20.62 ± 1.18**. On `FD001→FD004`, the corresponding numbers are
> **24.76 ± 1.77**, **23.86 ± 2.32**, and **24.34 ± 2.48**. So source
> initialization helps on both tasks, but the extra unlabeled/domain-adaptation
> machinery is **task-dependent**:
> it helps on `FD003`, while slightly hurting on `FD004`. The next controlled
> comparison should therefore be **why the same adaptation objective helps one
> task and hurts another**, not another new auxiliary loss. A first loss-level
> decomposition on `FD001→FD004` now sharpens that further: none of the
> maintained individual terms (`MMD`, source-stage, pseudo, monotonic) beats
> simple source-init-only supervised finetuning on the `5`-seed mean, and
> `source-stage + pseudo` also fails to recover the gap. The seed ordering is
> almost unchanged across all variants, which supports a narrower conclusion:
> the current `FD001→FD004` regression is not one obviously broken loss term,
> but a distributed small-regression pattern on top of the same
> task/partition-difficulty structure.

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
- a new target-domain full-supervised control on `FD003` now adds an
  important boundary condition:
  - with the current local best supervised recipe, `FD003` itself reaches
    **14.11 ± 0.19** (`3` seeds, fixed split)
  - this is still far below current `FD001→FD003` cross-domain few-shot
    results
  - therefore the canonical transfer ceiling is **not explained only** by the
    source supervised reproduction gap; there is also substantial
    cross-domain/few-shot headroom remaining relative to a target oracle
- the same target-oracle control is now also available on `FD004`:
  - with the same recipe, `FD004` reaches **16.45 ± 0.01** (`3` seeds, fixed split)
  - this remains well below current `FD001→FD004` cross-domain few-shot
    references (`24.34` stable v2 `5`-seed, `23.72` SPD high-LR `3`-seed)
  - therefore the harder multi-condition target still shows large remaining
    transfer/few-shot headroom relative to a target oracle

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

### 3.4 `DD-SSM / SPD` best non-SSL results (2026-04-04)

**Best non-SSL config**: SPD (dd_spd) + inv-MMD (0.1) + spec-domain-predictive (0.1) +
Adam LR=2e-3 + CosineAnnealingLR(T_max=20, eta_min=1e-5) +
matched protocol (resample + val-all-windows)

#### Single-task result (FD001→FD003, 3 seeds):

- mean RMSE = **20.37 ± 0.47**
- per-seed: 42=21.04, 43=19.99, 44=20.09
- v2 reference: 21.13 ± 1.59
- **improvement: -0.76 mean, -70% variance**
- this remains the strongest **non-SSL SPD baseline**
- a later `3`-seed union-SSL result beat it, but the subsequent `5`-seed
  reassessment no longer supports a robust default switch to union SSL

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

### 3.5 Cross-domain union SSL + no-spec adaptation (3-seed signal, 5-seed reassessment, selective-freeze follow-up)

Configuration:

- cross-domain SSL pretraining on **source train ∪ target train**
- SSL preset: `paper_full`
- downstream adaptation:
  - `lambda_spec_domain = 0.0`
  - `target_lr = 1.5e-3`
  - cosine scheduler
  - `domain_feature_tap = frontend_mean`

Initial matched result (`FD001→FD003`, seeds `42,43,44`):

| Seed | Baseline no-spec | SSL + no-spec | Delta |
|---|---:|---:|---:|
| 42 | 22.2752 | 21.3735 | -0.9018 |
| 43 | 19.6408 | 18.6110 | -1.0298 |
| 44 | 19.9500 | 19.5604 | -0.3896 |
| **mean** | **20.6220 ± 1.1758** | **19.8483 ± 1.1460** | **-0.7737** |

Comparison to prior best:

- previous SPD best mean (`spec_domain=0.1`, 3 seeds): **20.0151**
- union SSL + no-spec mean: **19.8483**
- further gain vs previous best: **-0.1668**

Additional observations:

- direct target RMSE mean:
  - baseline no-spec: **43.1865**
  - SSL + no-spec: **36.1212**
  - delta: **-7.0653**
- source RMSE mean:
  - baseline no-spec: **15.8368**
  - SSL + no-spec: **16.6510**
  - delta: **+0.8142**

Initial interpretation from the `3`-seed read:

- union SSL + no-spec looked like the strongest canonical `FD001→FD003` path
- this improved target-domain generalization even though source-domain fitting
  became slightly worse
- the gain pattern matched the actual cross-domain objective better than pure
  source fitting

Positive-case mechanism snapshot (seed `43`):

- `x_conv` MMD: **0.286 → 0.213**
- combined-core MMD: **0.638 → 0.181**
- invariant-state drift ratio (`step20 / step1`): **7.73 → 1.98**
- mixed-state drift ratio (`step20 / step1`): **6.73 → 1.46**
- target gate mean: **0.119 → 0.058**
- target `inv_only` ablation RMSE: **19.80 → 18.03**

Positive-case mechanism conclusion:

- the improvement does **not** mainly come from opening the specific branch
- instead, the strongest evidence is that union SSL **stabilizes the Mamba state
  dynamics** and makes the **invariant path** substantially stronger

`5`-seed reassessment (`42,43,44,45,46`):

| Aggregate | Baseline no-spec | SSL + no-spec | Delta |
|---|---:|---:|---:|
| mean adapted RMSE | **20.8917 ± 2.2820** | **21.8293 ± 3.1084** | **+0.9376** |
| mean direct target RMSE | 43.3926 | **39.3195** | **-4.0731** |
| mean source RMSE | **16.1985** | 16.6621 | +0.4637 |

Added seeds:

| Seed | Baseline no-spec | SSL + no-spec | Delta |
|---:|---:|---:|---:|
| 45 | 24.1368 | 23.1767 | -0.9601 |
| 46 | **18.4558** | 26.4250 | **+7.9693** |

Updated interpretation after the `5`-seed extension:

- the earlier `3`-seed signal was real, but it is **not robust enough** to
  support “union SSL + no-spec is the current best canonical default”
- SSL still helps direct transfer on average and helps `4 / 5` seeds in adapted
  RMSE, so the encoder-side signal is not fake
- however, one severe adaptation collapse (`seed 46`) is enough to reverse the
  mean and substantially increase variance
- follow-up seed-46 probes show:
  - SSL is **not** worse on the few-shot labeled target set at epoch `1`
  - removing unlabeled losses helps SSL on that seed
  - but does **not** close the gap to the baseline
- the current evidence therefore supports:
  - union SSL is a **promising but unstable** direction
  - the immediate next step is to diagnose early adaptation behavior on bad
    seeds rather than add new losses

Selective-freeze stabilization follow-up (`spec_gate_transformer_head`):

| Aggregate | Baseline no-spec | Original SSL full | Selective-freeze SSL |
|---|---:|---:|---:|
| mean adapted RMSE | **20.8917 ± 2.2820** | 21.8293 ± 3.1084 | 20.8995 ± 2.0281 |

Per-seed sign count:

- vs original SSL full:
  - better on **4 / 5** seeds
  - worse on **1 / 5** seed
- vs baseline no-spec:
  - better on **3 / 5** seeds
  - worse on **2 / 5** seeds

Interpretation:

- this is the first tested stabilization mechanism that materially repairs the
  original union-SSL variance problem
- the gain does **not** come from adding more losses; it comes from changing
  the **trainable scope during adaptation**
- the line is still **not a new default best**, because its `5`-seed mean is
  effectively tied with baseline no-spec rather than better
- the main positive claim is therefore narrower:
  - **adaptation scope / dynamics are a real bottleneck for SSL**
  - and selective freezing of the shared Mamba core is a credible lever

Hard-task transfer-direction check (`FD003→FD001`, seeds `42,43,44`):

| Aggregate | Baseline no-spec | Original SSL full | Selective-freeze SSL |
|---|---:|---:|---:|
| mean adapted RMSE | **20.2514 ± 0.6521** | 20.6290 ± 1.5962 | 20.6724 ± 2.1637 |

Hard-task interpretation:

- the canonical selective-freeze candidate does **not** generalize to this
  direction as a new default
- the freeze still changes behavior materially:
  - it lowers best-epoch invariant-MMD on all three seeds
- but lower invariant-MMD is **not sufficient** for better hard-task transfer:
  - `seed 44` still worsens from **22.4487 → 23.0168**
- targeted `seed 44` rescue probes then showed:
  - lower-LR full adaptation: **23.3594**
  - `transformer_head`: **23.0122**
  - `spec_gate_transformer_head`: **23.0168**
- this means the current hard-task failure is **not** explained by a simple
  “update too aggressively” story
- the supported conclusion is narrower:
  - selective freezing is a **task-conditional** stabilization mechanism
  - not a generally correct SSL replacement policy

Primary records:

- experiment log: [`cross_domain_experiment_log.md`](./cross_domain_experiment_log.md)
- generated summaries:
  - [`generated/ssl_union_nospec_frontend_summary_2026-04-09.json`](./generated/ssl_union_nospec_frontend_summary_2026-04-09.json)
  - [`generated/ssl_union_nospec_frontend_5seed_reassessment_2026-04-12.json`](./generated/ssl_union_nospec_frontend_5seed_reassessment_2026-04-12.json)
  - [`generated/ssl_union_seed46_collapse_diagnostic_fd001tofd003_2026-04-12.json`](./generated/ssl_union_seed46_collapse_diagnostic_fd001tofd003_2026-04-12.json)
  - [`generated/ssl_union_seed46_epoch1_fewshot_fit_probe_2026-04-12.json`](./generated/ssl_union_seed46_epoch1_fewshot_fit_probe_2026-04-12.json)
  - [`generated/seed46_supervised_only_ablation_fd001tofd003_2026-04-12.json`](./generated/seed46_supervised_only_ablation_fd001tofd003_2026-04-12.json)
  - [`generated/ssl_union_hard_task_freeze_reassessment_fd003tofd001_2026-04-12.json`](./generated/ssl_union_hard_task_freeze_reassessment_fd003tofd001_2026-04-12.json)

### 3.6 Semantic-SPD v1/v2 status (SUPERSEDED)

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

### 3.7 Shared-aux semantic-SPD + source semantic warmup (ARCHIVED)

> Historical branch only. As of `2026-04-12`, the semantic warmup and related
> auxiliary semantic losses are no longer part of the maintained v3 main path,
> because Sections `11.21` to `11.28` never beat the simpler inv-MMD core
> objective.

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

### 3.8 Same-subset self-supervised MambAtt reproduction snapshot (PAUSED)

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

### 3.9 Loss-function consolidation status (2026-04-12)

Maintained `v3` adaptation objective:

- source supervised regression
- target few-shot regression
- global MMD (`lambda_mmd`)
- source stage CE (`lambda_source_stage`)
- target pseudo-stage CE (`lambda_pseudo`)
- local monotonicity (`lambda_monotonic`)
- optional invariant-path MMD (`lambda_inv_mmd`)
- optional specific-branch domain CE (`lambda_spec_domain`)
- optional Transformer adapter L2 penalty when that branch is enabled

Retired from the maintained main path:

- cross-domain contrastive
- GRL / domain-adversarial alignment
- conditional semantic-SPD losses
- inv/spec orthogonality and related residual / reconstruction auxiliaries
- semantic warmup

Why this is evidence-supported:

- contrastive stayed neutral or negative in Sections `3.5`, `3.6`, and `3.10`
- direct inv-MMD beat GRL in Sections `11.12` to `11.15`
- the semantic-SPD auxiliary line in Sections `11.21` to `11.28` never beat
  the inv-MMD core reference
- the current best verified path in Sections `13.11` to `13.13` does not rely
  on the retired loss family

### 3.10 Task-Embedding MAML paper-aligned probe (`FD001→FD003`, `K=1`)

To check whether our current method can remain competitive under the
**Task-Embedding MAML** paper's much harsher `1-shot` regime, we added:

- `paper14` sensor subset support
- Min-Max normalization support
- paper-aligned v3 / baseline comparison runs

Matched settings:

- `target_shots = 1`
- `window_size = 30`
- `sensor_subset = paper14`
- `normalization_mode = minmax`

Seed-42 probe results:

| Method | Result |
|---|---:|
| Task-Embedding MAML paper (`FD001→FD003`) | **24.34** |
| `CD-MambAtt v3` strict probe (`val_units=1`) | **46.30** |
| `CD-MambAtt v3` stable-val probe (`val_units=10`) | **46.37** |
| minimal baseline (`pretrain + full finetune`) | **56.32** |

Interpretation:

- our source encoder is **not the main problem** here:
  - source RMSE under this paper-aligned preprocessing is already about **12.8**
- the true failure point is the **strict `K=1` cross-domain adaptation regime**
- our current DA-style scaffold still helps relative to naive fine-tuning
  (`56.32 → 46.37`)
- but it is still **far behind** the paper's `24.34`
- this is strong evidence that the paper's **meta-learning / task-embedding**
  mechanism matters in the `1-shot` setting

Primary note:

- [`history/experiment_notes/task_embedding_maml_paper_alignment_probe_2026-04-09.md`](./history/experiment_notes/task_embedding_maml_paper_alignment_probe_2026-04-09.md)

## 4. What we can and cannot claim right now

### Can claim

- the supervised MambAtt baseline has been reproduced at the **architecture / pipeline** level
- `CD-MambAtt v2` clearly improves over direct transfer on multiple C-MAPSS tasks
- reproduced `FOMLN` exists and current matched-shot numbers favor CD-MambAtt on 4 overlapping tasks
- `DD-SSM / SPD v0` is implemented inside the Mamba path and validated on CUDA
- **SPD v0 + inv-MMD beats v2 on both mean RMSE and cross-seed stability** on the canonical `FD001 -> FD003` task (3 seeds, matched protocol)
- **SPD provides a net -0.91 RMSE improvement** over v3+bare under the same protocol
- mechanism evidence now supports a more specific claim:
  **union SSL mainly strengthens the invariant path and reduces Mamba state drift**
- the strongest verified SSL stabilization lever so far is
  **selective freezing of the shared Mamba core while keeping the
  spec/gate + Transformer/head path trainable**
- that selective-freeze line improves the original union-SSL `5`-seed mean
  from **21.83 ± 3.11** to **20.90 ± 2.03**, but only **ties** the plain
  no-spec mean rather than beating it
- a new hard-task validation now rules out the strong generalization claim:
  the same selective-freeze policy is **not** a robust default on
  `FD003→FD001`
- matched target-only few-shot controls now exist on both `FD001→FD003` and
  `FD001→FD004`
- those controls show two things simultaneously:
  - target 5-shot labels alone already explain a large part of the gain over
    direct transfer
  - source initialization still provides an extra gain, but the effect size is
    modest and the extra unlabeled/domain-adaptation branch is now known to be
    task-dependent rather than uniformly helpful

### Cannot claim yet

- exact author-level reproduction of the target paper supervised number
- strict publication-grade apples-to-apples superiority over `FOMLN`
- cross-domain union SSL superiority across **multiple tasks** (currently only verified on FD001→FD003)
- new best SSL-enhanced line superiority at `5`-seed scale
- a generally correct rule for when selective freezing should replace plain
  full adaptation
- a simple “lower LR or more freezing will rescue hard-task SSL failures”
  explanation

## 5. Current bottlenecks

1. supervised reproduction gap to the target paper remains
2. ~~SPD innovation depth insufficient~~ → SPD is now the main innovation
3. ~~GRL unstable~~ → replaced with inv-MMD + spec-domain-predictive
4. ~~protocol mismatch~~ → resolved (v3 defaults fixed)
5. **selective freezing is now known to be task-conditional rather than universal** — we still do not know when it should be enabled, and the current hard-task evidence is negative
6. **SPD disentanglement effect is still weak** (gate barely moves; current best result does not rely on a strong spec branch)
7. **most of the remaining headroom is now known not to come from source initialization alone** — and the current unlabeled/domain-adaptation machinery is now known to help `FD001→FD003` but slightly hurt `FD001→FD004`
8. **the current `FD001→FD004` loss family does not contain an obvious single fix** — matched component controls show no maintained individual term beats source-init-only on the `5`-seed mean, so the next diagnosis should focus more on partition sensitivity / representation quality than on another minor loss reweighting
9. `FOMLN` comparison not yet refreshed against the latest stabilized / non-stabilized SSL lines
10. second dataset (XJTU-SY) not yet started
11. no matched-protocol ablation yet isolating **union SSL vs no union SSL** on multiple tasks

## 6. Next priority

Current recommended priority order:

1. **close the supervised source-backbone gap**, because the target-oracle controls show there is still major headroom before adaptation even becomes the whole story
2. **diagnose where the remaining few-shot-to-oracle gap is being lost**, now that matched decompositions show source initialization helps only partially
3. **diagnose task-conditional adaptation dynamics**, especially why the current unlabeled/domain-adaptation objective helps `FD001→FD003` but slightly hurts `FD001→FD004`, why the same seed ordering survives almost every `FD004` loss-component variant, and why the same freeze policy helps `FD001→FD003` but hurts `FD003→FD001`
4. **matched multi-task ablation**: plain no-spec vs original union SSL vs selective-freeze SSL, with the hard-task negative result treated as part of the benchmark
5. **refresh the 15-shot comparison against FOMLN** using whichever non-SSL / SSL branch is still evidence-supported after the ablation
6. only after the above, revisit whether stronger SPD disentanglement is still worth pursuing
7. second dataset (XJTU-SY)

## 7. Important files

- main experiment record: [`cross_domain_experiment_log.md`](./cross_domain_experiment_log.md)
- supervised consolidation: [`supervised_reproduction.md`](./supervised_reproduction.md)
- published baseline consolidation: [`published_baselines.md`](./published_baselines.md)
- innovation roadmap: [`dd_ssm_roadmap.md`](./dd_ssm_roadmap.md)
- generated tables: [`generated/formal_result_tables.md`](./generated/formal_result_tables.md)
