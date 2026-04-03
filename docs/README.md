# Documentation Index

Last updated: `2026-04-02`

This directory has been cleaned and merged into a smaller set of primary documents.
Historical experiment notes were **not deleted**; they were moved into `docs/history/`.

## Primary documents

| File | Purpose |
|---|---|
| [`project_status.md`](./project_status.md) | Current project snapshot, key results, fairness judgment, and next priorities |
| [`supervised_reproduction.md`](./supervised_reproduction.md) | Consolidated target-paper supervised reproduction status |
| [`published_baselines.md`](./published_baselines.md) | Consolidated published-baseline audit and reproduction status |
| [`dd_ssm_roadmap.md`](./dd_ssm_roadmap.md) | Consolidated innovation feasibility judgment and implementation roadmap |
| [`cross_domain_experiment_log.md`](./cross_domain_experiment_log.md) | **Authoritative experiment log**; preserved for all future additions |
| [`environment_snapshot.md`](./environment_snapshot.md) | CUDA / package / dataset location snapshot |

## Preserved records and assets

| Path | Contents |
|---|---|
| [`history/`](./history/README.md) | Historical experiment analyses, reproduction notes, and baseline audit notes |
| [`generated/`](./generated/) | Auto-generated result tables and JSON snapshots |
| [`text/`](./text/) | Extracted text from local PDFs |
| `*.pdf` | Local paper files and research guide PDFs |

## Cleanup policy used this round

1. Kept the experiment log untouched.
2. Merged the must-keep information into the 4 primary documents above.
3. Moved historical experiment/reproduction/baseline notes into `history/`.
4. Deleted redundant planning / review back-and-forth files that were fully superseded.

## Recommended reading order

1. `project_status.md`
2. `supervised_reproduction.md`
3. `published_baselines.md`
4. `dd_ssm_roadmap.md`
5. `cross_domain_experiment_log.md`
