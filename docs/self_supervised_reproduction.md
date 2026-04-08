# Self-Supervised Reproduction Status

Last updated: `2026-04-05`

This file records the current state of the paper-faithful **same-subset
self-supervised MambAtt reproduction** and marks this line as **paused** for now.

## 1. Scope and boundary

Paper being aligned:

- `Mamba-attention: A self-supervised framework for efficient remaining useful life prediction`

Important boundary:

- this paper section is **not cross-domain**
- self-supervised pretraining and one-shot supervised prediction use the **same C-MAPSS subset**
- current implementation follows that same-subset interpretation

Current project decision:

> The self-supervised reproduction path is now implemented and runnable, but we
> are **not investing more time** into squeezing out a stricter paper match on
> every subset. The code and results are preserved here so later cross-domain
> work can reuse them if needed.

## 2. Implemented code

### 2.1 New/extended files

| File | What is recorded there |
|---|---|
| `cd_mambatt/self_supervised.py` | temporal triplet builder, consecutive pair builder, sensor pseudo-label statistics/assignment, stable N-tuplet loss, self-supervised pretrainer wrapper |
| `train_self_supervised.py` | full paper-style pipeline: same-subset SSL pretrain -> encoder export -> one-shot supervised fine-tune/eval |
| `cd_mambatt/models/mambatt.py` | added encoder-sequence exposure via `_forward_mamba_sequence()`, `forward_mamba_sequence()`, and `extract_encoder_state_dict()` |

### 2.2 Implemented paper tasks

Implemented SSL losses:

- temporal ordering loss
- N-tuplet loss
- pseudo-label loss

Implemented one-shot protocol choices:

- `shots = 1`
- `val_units = 1`
- encoder **frozen** during one-shot fine-tuning
- unlabeled pretraining data comes from the **same subset**

### 2.3 Stable implementation choices adopted locally

These choices were important for obtaining sensible behavior:

- `ntuplet_mode = flatten`
- `ntuplet_normalize = False`
- `pseudo_value_mode = last`
- `val_all_windows = True`
- `resample_few_shot_per_seed = True`

## 3. Main result snapshot

Paper Table 5 reference RMSE:

| Subset | Paper |
|---|---:|
| FD001 | 31.7270 |
| FD002 | 30.3269 |
| FD003 | 32.3329 |
| FD004 | 32.6633 |

Current local snapshot:

| Subset | Current result | Note |
|---|---:|---|
| FD001 | **31.6116 ± 2.5595** | 3 seeds, essentially at paper level |
| FD002 | **23.5244** | seed 42 only |
| FD003 | **35.5230 ± 2.6474** | 3 seeds, above paper |
| FD004 | **38.8006** | seed 42 only, farthest from paper |

Interpretation:

- **FD001** is the strongest reproduction and is close enough to treat the pipeline as basically validated
- **FD002** is promising but only has one seed
- **FD003/FD004** remain mismatched enough that we are not treating this as a full-paper reproduction

## 4. Additional supporting results

### 4.1 FD001 self-supervision vs no pretraining

| Setting | Mean test RMSE | Std |
|---|---:|---:|
| no pretrain | 32.8112 | 5.4313 |
| paper-full self-supervised pretrain | **31.6116** | **2.5595** |

Conclusion:

- self-supervised pretraining improved the **mean**
- self-supervised pretraining significantly reduced **variance**

### 4.2 FD001 loss-ablation sanity check (`seed 42`)

| Setting | Test RMSE |
|---|---:|
| temporal only | 32.1312 |
| temporal + N-tuplet | 26.0186 |
| full (temporal + N-tuplet + pseudo) | **24.9943** |

This ordering is consistent with the paper's claim that the full objective is best.

## 5. Exact result files

### 5.1 Aggregated summaries

- `runs/paper_full_3seed_manual.json`
- `runs/no_pretrain_3seed.json`
- `docs/generated/self_supervised_reproduction_snapshot_2026-04-05.json`

### 5.2 Per-subset run directories

- `runs/ssl_repro_fd001_paperfull_v2/`
- `runs/ssl_repro_fd001_paperfull_v2_3seed/`
- `runs/ssl_repro_fd001_paperfull_v2_seed44_stable/`
- `runs/ssl_repro_fd001_nopre_3seed/`
- `runs/ssl_repro_fd002_paperfull_v2/`
- `runs/ssl_repro_fd003_paperfull_v2/`
- `runs/ssl_repro_fd003_paperfull_v2_moreseeds/`
- `runs/ssl_repro_fd004_paperfull_v2/`

## 6. Recommended interpretation going forward

What should be treated as solid:

- the paper's same-subset self-supervised training pipeline is now **implemented**
- the code path is stable enough to reuse as a component in later studies
- FD001 is a strong positive control showing the implementation is not fundamentally wrong

What should **not** be over-claimed:

- we do **not** have a full strict four-subset reproduction package
- we do **not** currently claim paper-level matching on FD003/FD004

## 7. Practical next-use guidance

If this code is reused later, start from:

```bash
/home/shelterpl/miniconda3/envs/cd_mamba/bin/python train_self_supervised.py \
  --subset FD001 \
  --seeds 42 \
  --pretrain-epochs 50 \
  --finetune-epochs 50 \
  --pretrain-batch-size 64 \
  --finetune-batch-size 64 \
  --ssl-preset paper_full \
  --freeze-encoder \
  --val-all-windows \
  --ntuplet-mode flatten \
  --no-ntuplet-normalize
```

If reused for cross-domain work later, keep in mind:

- current SSL code is **same-subset** by design
- any cross-domain SSL use should be treated as a **new research branch**, not as a direct continuation of this paper-faithful reproduction
