# Documentation Index

Last updated: `2026-04-11`

This directory has been cleaned and merged into a smaller set of primary documents.
Historical experiment notes were **not deleted**; they were moved into `docs/history/`.

## Primary documents

| File | Purpose |
|---|---|
| [`stage_report_for_advisor_zh_2026-04-11.md`](./stage_report_for_advisor_zh_2026-04-11.md) | Detailed Chinese stage report prepared for advisor briefing, including objectives, method evolution, key results, diagnostics, literature comparison, bottlenecks, and next steps |
| [`project_overview_zh_2026-04-08.md`](./project_overview_zh_2026-04-08.md) | Chinese project overview for quickly understanding the current code, results, bottlenecks, and next priorities |
| [`project_status.md`](./project_status.md) | Current project snapshot, key results, fairness judgment, and next priorities |
| [`supervised_reproduction.md`](./supervised_reproduction.md) | Consolidated target-paper supervised reproduction status |
| [`self_supervised_reproduction.md`](./self_supervised_reproduction.md) | Same-subset self-supervised MambAtt reproduction status, current code snapshot, and result archive |
| [`published_baselines.md`](./published_baselines.md) | Consolidated published-baseline audit and reproduction status |
| [`dd_ssm_roadmap.md`](./dd_ssm_roadmap.md) | Consolidated innovation feasibility judgment and implementation roadmap |
| [`cross_domain_experiment_log.md`](./cross_domain_experiment_log.md) | **Authoritative experiment log**; preserved for all future additions |
| [`environment_snapshot.md`](./environment_snapshot.md) | CUDA / package / dataset location snapshot |

## Preserved records and assets

| Path | Contents |
|---|---|
| [`../experiments/`](../experiments/) | One-off experiment scripts, now grouped into `ablations/`, `diagnostics/`, `prototypes/`, and `quick_tests/` |
| [`history/`](./history/README.md) | Historical experiment analyses, reproduction notes, and baseline audit notes |
| [`generated/`](./generated/) | Auto-generated result tables and JSON snapshots |
| [`../scripts/`](../scripts/) | Reusable batch runners, experiment drivers, and result aggregation helpers |
| [`text/`](./text/) | Extracted text from local PDFs |
| `*.pdf` | Local paper files and research guide PDFs |

## Cleanup policy used this round

1. Kept the experiment log untouched.
2. Merged the must-keep information into the main primary documents above.
3. Moved historical experiment/reproduction/baseline notes into `history/`.
4. Deleted redundant planning / review back-and-forth files that were fully superseded.

## Recommended reading order

1. `project_status.md`
2. `stage_report_for_advisor_zh_2026-04-11.md`
3. `supervised_reproduction.md`
4. `self_supervised_reproduction.md`
5. `published_baselines.md`
6. `dd_ssm_roadmap.md`
7. `cross_domain_experiment_log.md`
