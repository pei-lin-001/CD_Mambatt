# CD-MambAtt Reflection Benchmark (2026-04-12)

This file is the working benchmark for future reflection and decision-making.
Its purpose is not to list every run again, but to answer four practical
questions:

1. What directions have already been tested thoroughly enough?
2. Which of them actually improved the project?
3. Which of them consumed substantial time but should now be treated as
   low-return or stopped?
4. What problems remain genuinely unsolved?

Primary sources:

- `/home/shelterpl/cd_mambatt/docs/cross_domain_experiment_log.md`
- `/home/shelterpl/cd_mambatt/docs/project_status.md`
- `/home/shelterpl/cd_mambatt/docs/generated/ssl_union_nospec_frontend_5seed_reassessment_2026-04-12.json`
- `/home/shelterpl/cd_mambatt/docs/generated/ssl_union_seed46_collapse_diagnostic_fd001tofd003_2026-04-12.json`
- `/home/shelterpl/cd_mambatt/docs/generated/ssl_union_seed46_epoch1_fewshot_fit_probe_2026-04-12.json`
- `/home/shelterpl/cd_mambatt/docs/generated/seed46_supervised_only_ablation_fd001tofd003_2026-04-12.json`
- `/home/shelterpl/cd_mambatt/docs/generated/ssl_union_hard_task_freeze_reassessment_fd003tofd001_2026-04-12.json`
- `/home/shelterpl/cd_mambatt/docs/generated/validation_diagnostics_snapshot_2026-04-12.json`

---

## 1. Direction-Level Retrospective

| Direction | Representative evidence | What the evidence says now | Judgment | Future policy |
|---|---|---|---|---|
| Direct transfer -> target few-shot full finetune | log sections `2.1` to `2.4` | This is the first major gain source. Direct transfer is poor; target few-shot supervision recovers most of the lost performance. | **Core foundation, correct direction** | Keep as the baseline foundation for all comparisons. |
| Global MMD (`v1`) | sections `3.1`, `3.2` | MMD gives a real gain over strict full finetune, but the gain is only about `~1 RMSE`. The weight sweep responds, so it is not a dead loss. | **Effective but limited** | Keep as part of the maintained core; do not expect it to solve the main bottleneck alone. |
| Pseudo-stage + monotonic (`v2` core) | sections `3.3`, `3.4`, `3.7`, `3.8`, `7.*`, `9.*` | This is the most stable cross-task working recipe. It improves multiple transfer pairs and is the strongest broad working line. | **Main working path** | Keep as the stable baseline / production path. |
| Cross-domain contrastive | sections `3.5`, `3.6`, `3.9`, `3.10` | Repeatedly failed to improve the mean result; combination with monotonic also hurt. | **Direction empirically rejected** | Stop allocating main effort here unless a sharply different formulation is proposed and justified. |
| GRL / domain-adversarial alignment | sections `11.5` to `11.15` | GRL was unstable and inferior to direct invariant-path MMD. | **Direction empirically rejected for maintained path** | Stop as a mainline loss direction. |
| SPD core architecture (`dd_spd`) | sections `11.*`, `13.1` to `13.10` | SPD can help on canonical `FD001->FD003`, especially with high LR + cosine, but the gain is not universal across tasks. | **Conditionally useful, not universal** | Keep as a mechanism / architecture branch, but not as the default answer to every performance problem. |
| Semantic-SPD auxiliary family | sections `11.20` to `11.28` | Multiple redesigns, conditional losses, orthogonality, residual heads, and warmup did not beat the simpler inv-MMD core line. | **High effort, low return** | Stop in the maintained path. Treat as archived exploration, not an active priority. |
| Protocol debugging and matched re-runs | sections `12.1` to `12.4` | This work was essential. Earlier negative judgments were partially distorted by protocol mismatch. | **High-value diagnostic work** | Keep this standard permanently: verify protocol before interpreting results. |
| High LR + cosine schedule | sections `13.6`, `13.9` | One of the clearest positive levers. Optimization settings materially changed outcomes and stacked with SPD. | **Real orthogonal gain** | Keep. Any future new branch must be tested under a matched strong optimization protocol. |
| Union SSL + no-spec adaptation | sections `13.11` to `13.22` | `3`-seed signal was positive, but `5`-seed reassessment reversed the mean due to a severe bad seed. Canonical selective-freeze stabilization helps that task, yet a hard-task validation on `FD003->FD001` does not generalize. | **Promising but task-conditional** | Do not call it the default best path. Treat both union SSL and selective freeze as conditional branches until the task/seed regime is understood. |
| More and more loss terms after MMD | broad pattern across sections `3.*`, `11.*`, `13.15` | After MMD pulled the main alignment gain, most later loss additions only moved results within a narrow band near `20~21` on the canonical task. | **The main strategic mistake** | Stop loss-family expansion as the default research reflex. |
| Source backbone reproduction gap | project status `3.1` | Supervised source reproduction is still around `15.09` vs paper `11.46`. This unresolved gap likely constrains the transfer ceiling. | **Unresolved high-priority bottleneck** | Promote to top-tier priority. |
| Adaptation dynamics / split sensitivity | sections `13.16` to `13.19` | Bad seeds are not explained by simple training-set fit. Some failures are generalization / dynamics problems, not loss-weight problems. | **Unresolved high-priority bottleneck** | Promote to top-tier priority. |
| Task-Embedding MAML paper-aligned `K=1` probe | section `13.14` | Under paper-aligned `K=1`, our DA-style method is far worse than the paper even when source fitting is good. | **Important boundary condition** | Keep as evidence that DA-style transfer is insufficient for strict `K=1`; do not conflate this with the `5-shot` mainline. |

---

## 2. Effort vs Return Classification

| Classification | What belongs here | Why it belongs here |
|---|---|---|
| Proven high-value work | target few-shot adaptation baseline, MMD core, pseudo + monotonic core, protocol correction, high LR + cosine | These either created the main performance jump or prevented false conclusions. |
| Useful diagnostic work even without final gains | hidden-state drift probes, SPD disentanglement checks, SSL seed-46 collapse diagnosis, epoch-1 few-shot-fit probe, supervised-only SSL ablation | These did not necessarily improve the benchmark, but they changed what we now know is and is not the true bottleneck. |
| High-effort low-return work | contrastive, GRL, semantic-SPD auxiliary family, broad loss proliferation after MMD | These directions consumed substantial time but did not produce a stable new level of performance. |
| Still-open hard problems | source backbone gap, split-sensitive adaptation collapse, invariant/front-end representation quality, why SSL can improve few-shot fit but worsen validation/test generalization on bad seeds | These remain unresolved and are now more central than adding new loss formulas. |

---

## 3. Hard Conclusions vs Open Questions

### 3.1 Hard conclusions already supported by evidence

| Statement | Status | Evidence |
|---|---|---|
| Cross-domain few-shot CD-MambAtt is real and not a single-task artifact. | Supported | multi-task `5-seed` results in sections `7.*` and project status `3.2` |
| MMD works, but only as a secondary gain source. | Supported | sections `3.1`, `3.2` |
| Contrastive is not helping the maintained path. | Supported | sections `3.5`, `3.6`, `3.9`, `3.10` |
| GRL is inferior to inv-MMD in this project. | Supported | sections `11.12` to `11.15` |
| Semantic-SPD auxiliary loss expansion did not justify itself. | Supported | sections `11.21` to `11.28`, `13.15` |
| The supervised backbone gap to the paper is still real. | Supported | project status `3.1` |
| Union SSL is not yet a robust default best config on `FD001->FD003`. | Supported | sections `13.16` to `13.19` |
| The current selective-freeze SSL stabilization is not a universal default; it fails to generalize on `FD003->FD001`. | Supported | sections `13.21`, `13.22` |
| On SSL bad seed `46`, the failure is not explained by worse few-shot labeled fit. | Supported | section `13.18` |
| On SSL bad seed `46`, unlabeled losses worsen the collapse but do not fully explain it. | Supported | section `13.19` |

### 3.2 Open questions that are still legitimate

| Question | Why it remains open |
|---|---|
| Why does SSL sometimes improve few-shot labeled fit but worsen val/test generalization? | Current probes rule out the simplest explanations, and the new hard-task freeze failure shows that “just update less” is still too coarse. |
| How much of the transfer ceiling is imposed by the source backbone gap? | The gap is clearly real, but its exact downstream contribution is not yet quantified across tasks. |
| Which part of the representation pipeline is the main instability source: front-end, invariant path, or checkpoint initialization geometry? | Current diagnostics strongly suggest these areas, but do not yet fully separate them. |
| Can SPD be made robust across tasks rather than just strong on the canonical task? | Current evidence is mixed, not decisively positive. |

---

## 4. What We Should Treat As Strategic Mistakes

| Mistake | Why it was a mistake | Correction |
|---|---|---|
| Continuing to expand loss families after MMD without first proving that the previous loss was the actual bottleneck | The project repeatedly stayed in the same narrow performance band, implying diminishing returns in loss space. | New loss terms must now clear a much higher bar before implementation. |
| Interpreting early negative results before protocol alignment was verified | Protocol mismatch previously distorted conclusions. | Always verify split, validation, and checkpoint-selection comparability first. |
| Treating canonical-task improvements as if they automatically generalized across tasks | SPD best on `FD001->FD003` did not stay best everywhere else. | Any “best” claim must state the task scope explicitly. |
| Treating a `3`-seed positive SSL signal as if it were enough to redefine the default best configuration | The `5`-seed reassessment reversed the mean. | Default-best claims now require broader reassessment and variance checks. |

---

## 5. What We Should Treat As Non-Negotiable Rules Going Forward

| Rule | Practical meaning |
|---|---|
| No new loss term by default | A new loss is no longer the standard next move. It must be justified by a diagnosed mechanism gap, not by lack of ideas. |
| Failure must be localized before redesign | If a run regresses, first prove whether the issue is protocol, fit, alignment, pseudo-label use, or generalization. |
| Strong optimization is part of the baseline, not an afterthought | Future comparisons must use the matched strong schedule, not a weaker default. |
| Canonical-task wins are not enough | A new direction should be tested on at least one harder transfer direction before being treated as a meaningful new path. |
| Separate “helps a seed” from “changes the mean” | A positive anecdote is not a new default. |

---

## 6. Priority Table For Future Work

| Priority | Topic | Reason |
|---|---|---|
| Highest | Source supervised backbone gap | This is the clearest unresolved ceiling issue and affects every downstream transfer path. |
| Highest | Adaptation-dynamics diagnosis on bad seeds | Current evidence shows some failures are not training-fit failures but generalization / dynamics failures. |
| Medium | Representation diagnostics around front-end and invariant path | This is where several mechanism probes are already pointing. |
| Medium | Controlled SSL failure analysis | SSL still has signal; the question is when and why it breaks. |
| Low | New loss-family design | Current evidence says this is a saturated search space. |
| Low | Reviving archived semantic-SPD auxiliary lines | Too much effort already spent for too little verified return. |

---

## 7. Benchmark Summary In One Sentence

The project has already proven a valid cross-domain few-shot mainline, but it
has also shown that most post-MMD loss expansion was low-return; the real
remaining problems are now source-backbone quality and split-sensitive
adaptation generalization, not lack of another alignment loss.
