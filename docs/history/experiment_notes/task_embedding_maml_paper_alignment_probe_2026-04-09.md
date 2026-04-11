# Task-Embedding MAML Paper-Alignment Probe (2026-04-09)

## Goal

Probe how `CD-MambAtt` behaves when the data protocol is pushed much closer to
the **Task-Embedding MAML** paper:

- paper:
  - *A cross-domain few-shot remaining useful life estimation framework based on model-agnostic meta-learning with task embeddings*
  - local file:
    `/home/shelterpl/cd_mambatt/docs/1-s2.0-S0166361525001617-main.pdf`

## Aligned settings used here

The paper settings that were matched as closely as our current pipeline allows:

- task: `FD001 → FD003`
- target shots: **`1`**
- window size: **`30`**
- sensors: **14-sensor subset**
  - preset: `paper14`
  - indices (1-based): `2,3,4,7,8,9,11,12,13,14,15,17,20,21`
- normalization: **Min-Max**
  - for multi-condition subsets, our code now also supports condition-wise
    Min-Max
- RUL cap: `125`

Important mismatches that still remain:

- the paper is **MAML + segmentation + task embeddings**
- our method is still **cross-domain adaptation**, not meta-learning
- the paper reports averages over **50 repetitions**
- our probe here is **seed 42 only**

## Code changes made for this probe

To support this comparison, the following capabilities were added:

- sensor subset preset support:
  - `cd_mambatt/data.py`
  - `train_cross_domain_baseline.py`
  - `train_cd_mambatt_v1.py`
  - `train_cd_mambatt_v3.py`
- Min-Max normalization support:
  - `cd_mambatt/data.py`
- fixed a shape bug triggered by paper14 input in v3 monotonic loader:
  - `train_cd_mambatt_v3.py`

## Experiments

### 1. Strict paper-aligned `CD-MambAtt v3` probe

Command family:

```bash
conda run -n cd_mamba python train_cd_mambatt_v3.py \
  --task FD001_TO_FD003 \
  --seeds 42 \
  --target-shots 1 \
  --target-val-units 1 \
  --window-size 30 \
  --sensor-subset paper14 \
  --normalization-mode minmax \
  --target-lr 1.5e-3 \
  --target-lr-scheduler cosine \
  --lambda-spec-domain 0.0 \
  --domain-feature-tap frontend_mean
```

Artifacts:

- run dir:
  `/home/shelterpl/cd_mambatt/runs/task_embed_paperalign_v3_strict_seed42/FD001_TO_FD003`
- log:
  `/home/shelterpl/cd_mambatt/runs/task_embed_paperalign_v3_strict_seed42.log`

Results:

- source test RMSE: **12.7394**
- direct target RMSE: **52.9791**
- adapted target RMSE: **46.3013**

Mechanism snapshot:

- gate mean: **~0.170**
- best pseudo acceptance ratio: **0.0680**
- best epoch: **6**
- domain accuracy: **1.0**

Interpretation:

- under strict `1-shot + W=30 + 14 sensors + minmax`, the model still learns
  the **source** subset well
- but the **cross-domain gap explodes**
- adaptation helps a little (`52.98 → 46.30`) but remains far from the paper's
  reported `24.34`

### 2. Stable-validation variant (`target_val_units = 10`)

Same as above, except:

- `--target-val-units 10`

Artifacts:

- run dir:
  `/home/shelterpl/cd_mambatt/runs/task_embed_paperalign_v3_val10_seed42/FD001_TO_FD003`
- log:
  `/home/shelterpl/cd_mambatt/runs/task_embed_paperalign_v3_val10_seed42.log`

Results:

- source test RMSE: **12.8178**
- direct target RMSE: **53.0644**
- adapted target RMSE: **46.3688**

Interpretation:

- increasing validation units does **not** solve the problem
- therefore the failure is **not mainly a model-selection noise issue**

### 3. Paper-aligned minimal baseline (`pretrain + full finetune`)

Command family:

```bash
conda run -n cd_mamba python train_cross_domain_baseline.py \
  --task FD001_TO_FD003 \
  --seeds 42 \
  --target-shots 1 \
  --target-val-units 10 \
  --finetune-mode full \
  --window-size 30 \
  --sensor-subset paper14 \
  --normalization-mode minmax
```

Artifacts:

- run dir:
  `/home/shelterpl/cd_mambatt/runs/task_embed_paperalign_baseline_val10_seed42/FD001_TO_FD003`
- log:
  `/home/shelterpl/cd_mambatt/runs/task_embed_paperalign_baseline_val10_seed42.log`

Results:

- source test RMSE: **13.0418**
- direct target RMSE: **53.3152**
- fine-tuned target RMSE: **56.3190**

Interpretation:

- plain `1-shot` fine-tuning **overfits and gets worse**
- our v3 adaptation scaffold is still substantially better than this baseline:
  - **56.32 → 46.37** (about **10 RMSE** gain)

## Comparison to Task-Embedding MAML paper

Paper number used:

- `FD001→FD003 = 24.34`

Local paper-aligned probe summary:

| Method | Setting | RMSE |
|---|---|---:|
| Task-Embedding MAML (paper) | reported | **24.34** |
| CD-MambAtt v3 | strict paper-aligned probe | **46.30** |
| CD-MambAtt v3 | val10 probe | **46.37** |
| Pretrain + full finetune baseline | val10 probe | **56.32** |

## Main conclusion

This probe gives a clean diagnostic result:

1. Our current encoder is **not the main bottleneck on source fitting** under
   the paper-aligned data settings:
   - source RMSE already reaches about **12.8**
2. The real failure is the **`K=1` cross-domain adaptation regime**
3. The paper's **meta-learning / task-embedding machinery is likely genuinely
   valuable** in this setting
4. Our current cross-domain DA-style training can still help relative to plain
   fine-tuning, but it is **not competitive with the paper in the strict 1-shot
   protocol**

## Recommended next action

If we keep following this comparison line, the next most meaningful direction is
**not** blind hyperparameter tuning, but:

- introduce a **true meta-learning / episode-based adaptation path**, or
- explicitly build a **low-dimensional task-adaptation module** for the `K=1`
  regime

In other words:

> this experiment supports the view that our current method is strong in the
> `5-shot DA` regime, but weak in the paper's `1-shot meta-learning` regime.
