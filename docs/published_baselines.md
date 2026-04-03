# Published Baselines Status

Last updated: `2026-04-02`

This file consolidates the old baseline-audit, FOMLN progress, and fairness notes.

## 1. Current baseline landscape

| Baseline | Role in this project | Current status | Practical judgment |
|---|---|---|---|
| `FOMLN` | true reproduced published baseline | implemented and tested on CUDA | useful and important, but current comparison is not yet strictly apples-to-apples |
| `MetaDFKN` | strongest reported numerical threat | protocol audit only | do **not** reproduce blindly before target-domain setup is cleaned |
| Task-Embedding MAML | literature reference | reported baseline only | current C-MAPSS numbers are from `K=1`, so they are not a fair direct comparator to our `5-shot` line |

## 2. Reproduced `FOMLN` baseline snapshot

Current stable interpretation used in our reproduction:

- target regime: `15-shot`
- sensor subset: paper `15` sensors
- window length: `30`
- RUL cap: `125`
- condition-based standardization for `FD002/FD004`
- source-task construction: unit-level interpretation
- source inner optimizer: `Adam`
- target adaptation optimizer: `SGD`
- CUDA-only execution
- current working hidden width: `d_model = 512`

Important wording:

- early `fomln_*` smoke runs are **not** formal baseline results
- current formal line is the mini-batch, CUDA-stable, paper-alignment-in-progress branch

## 3. Current matched-shot comparison: `CD-MambAtt 15-shot` vs reproduced `FOMLN 15-shot`

| Task | CD-MambAtt 15-shot | FOMLN 15-shot | Delta (CD - FOMLN) |
|---|---:|---:|---:|
| `FD001 -> FD003` | **20.1384** | 25.4748 | -5.3364 |
| `FD003 -> FD001` | **19.5788** | 20.1261 | -0.5474 |
| `FD002 -> FD004` | **23.2879** | 26.0206 | -2.7327 |
| `FD001 -> FD004` | **23.9469** | 25.0808 | -1.1340 |

Negative delta means CD-MambAtt is better.

## 4. Fairness judgment

The old wording "fair comparison" was too strong.

The correct wording is:

> **matched-shot in-house comparison with remaining protocol mismatches**

Main reasons:

1. CD-MambAtt uses `15` target support units **plus** `10` labeled target validation units
2. CD-MambAtt uses an unlabeled target pool for `MMD / pseudo / monotonic`, while FOMLN does not use the same target information path
3. preprocessing pipelines are not unified (`21` sensors / `W=20` vs `15` sensors / `W=30`)
4. model selection rules differ
5. FOMLN is still marked `paper_alignment_in_progress`

Practical interpretation:

- large margins (especially `FD001 -> FD003`) are encouraging
- small margins should be stated cautiously

## 5. What we can honestly claim now

Can claim:

- we have a real reproduced `FOMLN` baseline in the same local environment
- under the current matched-shot protocol, CD-MambAtt beats that reproduced FOMLN line on 4 overlapping tasks

Cannot claim yet:

- strict publication-grade superiority over exact-paper `FOMLN`
- fully controlled same-pipeline benchmarking

## 6. MetaDFKN status

Current judgment after local-paper audit:

- `MetaDFKN` is still the next most dangerous **reported** baseline
- but its target-domain few-shot protocol remains ambiguous from the local PDF / extracted text
- therefore it should stay in **protocol-cleaning** status until the target setup is made explicit

What is still unclear enough to block coding:

- exact target few-shot construction
- exact support/query definition
- exact C-MAPSS window setting from the audited snippets
- optimizer / repetition details in a clean reproducible form

## 7. Recommended baseline priority from now on

1. keep `FOMLN` as the current reproduced published baseline reference
2. if baseline work resumes, do **one more MetaDFKN protocol-cleaning pass** before implementation
3. keep Task-Embedding MAML as a reported literature reference, not an immediate reproduction target

## 8. Historical notes preserved

- `history/experiment_notes/fomln_baseline_progress_2026-04-01.md`
- `history/experiment_notes/cd_mambatt_vs_fomln_formal_comparison_2026-04-02.md`
- `history/experiment_notes/comparison_fairness_audit_2026-04-02.md`
- `history/baseline_notes/published_baseline_paper_audit_2026-04-01.md`
- `history/baseline_notes/metadfkn_protocol_cleaning_2026-04-02.md`
