# MetaDFKN Protocol-Cleaning Note (2026-04-02)

## Purpose

This note is the **next-step audit** after finishing the fair
`CD-MambAtt 15-shot vs FOMLN 15-shot` comparison.

The goal is not to implement MetaDFKN yet, but to answer:

1. what is explicit in the local MetaDFKN paper,
2. what is still ambiguous for fair reproduction,
3. whether MetaDFKN should be the next true baseline implementation target.

Local source used:

- `/home/shelterpl/cd_mambatt/docs/1-s2.0-S0951832024000036-main (1).pdf`
- extracted text:
  - `/home/shelterpl/cd_mambatt/docs/text/metadfkn_2024_ress.txt`

---

## 1. What is explicit from the local paper

### 1.1 Benchmark and task coverage

The paper explicitly uses:

- `C-MAPSS`
- all **12 cross-domain transfer scenarios**
- `RMSE` and `SCORE`

The extracted text also explicitly says:

> the training sets of `FD001/FD002/FD003/FD004` are used as source domains,
> and the test sets of the four datasets are used as target domains.

This is a major protocol difference from our current in-house few-shot setup,
where target support / validation / test are all sampled from the official
target-side train/test structure.

### 1.2 Input preprocessing choices

Explicit items found in the local text:

- keep **14 sensors**
- use segmented linear RUL with:
  - `RUL_max = 125`

The extracted text clearly states the 14 sensors are chosen because they are
highly correlated with degradation, but the exact sensor index list is not
cleanly shown in the extracted snippet.

### 1.3 Model-side choices

Explicit hyperparameter findings:

- flow kernel length:
  - **`K = 10`**
- TinyCMN hidden layers:
  - final choice = **single GRU layer**

### 1.4 Reported C-MAPSS table values

The local PDF text contains the paper's Table 6 values, e.g.:

- `FD001 → FD003 = 13.03`
- `FD003 → FD001 = 5.48`
- `FD002 → FD004 = 9.38`
- `FD001 → FD004 = 13.11`

These are numerically very strong and therefore remain a serious reported
baseline threat.

---

## 2. Main protocol ambiguities

### 2.1 The biggest ambiguity: what exactly is the target-domain few-shot protocol?

The paper contains two signals that do **not** align cleanly:

1. it says:
   - source = training sets
   - target = test sets

2. but in the comparison section it also says recent methods are:
   - trained on source-domain data
   - then fine-tuned with a **limited quantity of target-domain data**
   - described in the text as **"Target-Unsupervised"**

This leaves several unresolved questions:

- Is MetaDFKN itself using labeled target support from the target train split?
- Or is it using target test-domain data in some few-shot adaptation form?
- Does "few-shot" mean **few units**, **few windows**, or **few trajectories**?
- Is the protocol semi-supervised, supervised few-shot, or partially
  target-unlabeled with a small labeled support set?

Until this is cleaned up, the paper numbers should be treated as
**reported baselines**, not fair reproduced baselines.

### 2.2 C-MAPSS window construction is still unclear

The extracted text explicitly gives the PRONOSTIA sliding-window setting, but
the C-MAPSS window length / stride is not cleanly exposed in the extracted
segments I reviewed today.

This matters because C-MAPSS RUL performance can move noticeably with:

- window length
- stride
- sensor subset
- normalization scope

### 2.3 Training protocol is still under-specified from the extracted text

From the local extracted snippets reviewed so far, I do **not** yet have a
clean statement of:

- optimizer choice
- outer / inner learning rates
- batch size
- number of repetitions / seeds
- exact support/query construction for each meta-task

These may exist in the full PDF, but they are not yet in a clean enough form
for direct implementation.

---

## 3. Fairness judgment

## 3.1 What is fair to say now

It is fair to say:

- MetaDFKN is a **strong reported baseline**
- it is relevant to our problem
- it is likely a more dangerous published comparator than Task-Embedding MAML

## 3.2 What is not fair to say yet

It is **not** fair to say yet:

- that MetaDFKN is directly comparable to our current `5-shot` pipeline
- that its paper numbers are apples-to-apples with our current support/query
  protocol
- that we can implement it faithfully right now without first cleaning the
  target-domain setup

---

## 4. Decision after this audit pass

### Current decision

> **MetaDFKN should be the next protocol-cleaning target, but not yet the next
> immediate coding target until the target-domain setup is clarified.**

This is different from FOMLN:

- FOMLN had enough explicit structure to justify a real reproduction path
- MetaDFKN still has too much ambiguity in the few-shot target protocol

---

## 5. Recommended next action

### Option A — safest

Do one more **paper-only audit pass** on the local PDF and extract:

1. exact C-MAPSS window setting
2. exact sensor indices
3. exact few-shot target construction
4. explicit optimizer / learning-rate / repetition settings

### Option B — faster but riskier

Implement a **best-effort MetaDFKN-style proxy baseline** with clearly marked
deviations from the paper.

I do **not** recommend Option B yet.

---

## 6. Bottom line

Today’s conclusion is:

> MetaDFKN remains the next most important published baseline, but it still
> requires **protocol cleaning before reproduction**.

So the project status is now:

1. FOMLN: **formal reproduced baseline completed**
2. MetaDFKN: **paper audit in progress**
3. Task-Embedding MAML: **reported-baseline only for now**
