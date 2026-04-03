# Comparison Fairness Audit (2026-04-02)

## Question

Was the previous `CD-MambAtt vs FOMLN` comparison really fair?

## Short answer

> **Not fully.**

More precisely:

- the old `CD-MambAtt 5-shot vs FOMLN 15-shot` comparison is **not fair**
- the newer `CD-MambAtt 15-shot vs FOMLN 15-shot` comparison is only
  **partially fair / matched-shot**, but **not strict apples-to-apples**

---

## 1. What is fair in the newer `15-shot vs 15-shot` comparison

These parts are aligned:

1. same transfer pairs
   - `FD001→FD003`
   - `FD003→FD001`
   - `FD002→FD004`
   - `FD001→FD004`
2. same number of seeds
   - `42,43,44,45,46`
3. same evaluation metrics
   - `RMSE`
   - `SCORE`
4. same nominal support-shot count
   - both runners use `15` target support units
5. both are in-house reruns on the same local environment

Therefore, calling it a **matched-shot comparison** is reasonable.

---

## 2. Why it is still not strictly fair

### 2.1 Extra labeled target validation data for CD-MambAtt

CD-MambAtt `15-shot` runs use:

- `15` labeled target support units
- **plus `10` labeled target validation units**

FOMLN formal reproduction uses:

- `15` labeled target support units
- **`0` target validation units**

So the total labeled target budget is not matched:

- CD-MambAtt: effectively `15 + 10`
- FOMLN: `15`

This clearly favors CD-MambAtt in model selection.

### 2.2 Unlabeled target-pool access is not matched

CD-MambAtt uses the remaining target-train units as an unlabeled pool for:

- MMD
- pseudo-labeling
- monotonicity construction

FOMLN reproduction does **not** use an analogous unlabeled target pool.

So the comparison is not matched in target-domain information access.

### 2.3 Preprocessing pipeline is not unified

CD-MambAtt and FOMLN are not evaluated under the same input pipeline:

- CD-MambAtt:
  - `21` sensors
  - window length `20`
- FOMLN reproduction:
  - paper sensor subset (`15` sensors)
  - window length `30`
  - condition-based standardization for multi-condition subsets

This is acceptable for a **paper-style reproduced baseline comparison**, but it
is not a same-pipeline controlled benchmark.

### 2.4 FOMLN reproduction is still marked `paper_alignment_in_progress`

The FOMLN runner itself still records:

- `implementation_stage = paper_alignment_in_progress`

Main unresolved gaps already documented:

- meta-task construction ambiguity
- some conformer/internal width choices inferred rather than explicitly stated
- no exact proof yet that the implementation fully matches the published paper

So the current FOMLN line is a strong **reproduced in-house baseline**, but not
yet an unquestionable exact-paper reproduction.

### 2.5 Model-selection protocol differs

CD-MambAtt:

- selects checkpoints by **target validation RMSE**

FOMLN reproduction:

- with `target_val_samples = 0`
- falls back to selecting the best adaptation step by **support RMSE**

This again means the two methods are not using the same model-selection rule.

---

## 3. Practical interpretation of the current result

The current `15-shot vs 15-shot` table should therefore be interpreted as:

> **matched-shot but not fully apples-to-apples**

This means:

- large wins are encouraging and probably meaningful
- small wins should be treated cautiously

### Margin audit

| Task | CD-MambAtt 15-shot | FOMLN 15-shot | Margin |
|---|---:|---:|---:|
| `FD001→FD003` | 20.1384 | 25.4748 | **5.3364** |
| `FD003→FD001` | 19.5788 | 20.1261 | **0.5474** |
| `FD002→FD004` | 23.2879 | 26.0206 | **2.7327** |
| `FD001→FD004` | 23.9469 | 25.0808 | **1.1340** |

Interpretation:

- `FD001→FD003`: strong enough to remain convincing
- `FD002→FD004`: still reasonably convincing
- `FD003→FD001` and `FD001→FD004`: too close to claim as fully decisive under
  the current protocol mismatch

---

## 4. Final verdict

### What we can honestly claim now

1. `CD-MambAtt 5-shot vs FOMLN 15-shot`
   - **not fair**
   - should only be used as a loose data-efficiency reference

2. `CD-MambAtt 15-shot vs FOMLN 15-shot`
   - **partially fair**
   - good as an in-house matched-shot comparison
   - **not yet strict publication-grade apples-to-apples evidence**

---

## 5. How to make the next comparison genuinely stricter

Recommended order:

1. rerun CD-MambAtt with **no extra target validation units**
   - or give FOMLN the same `+10` target validation units
2. clearly state whether the comparison allows **unlabeled target pools**
   - if yes, this is a semi-supervised few-shot setting
   - if no, CD-MambAtt needs a restricted comparison variant
3. if needed, run a same-pipeline control:
   - same sensor subset
   - same window length
   - same condition normalization
4. continue cleaning FOMLN until it is no longer marked
   `paper_alignment_in_progress`

---

## Bottom line

> The current comparison is useful and encouraging, but the word
> **“fair” was too strong**.

The correct wording should be:

> **matched-shot in-house comparison with remaining protocol mismatches**.
