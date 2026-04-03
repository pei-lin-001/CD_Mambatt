# Published Baseline Paper Audit (2026-04-01)

## Purpose

This note audits the **actual local paper files** currently placed in `docs/`
before reproducing published baselines for `CD-MambAtt`.

The main goal is to answer:

1. Which published methods are **actually comparable** to our current setup?
2. Which methods are worth **true reproduction** first?
3. Which paper numbers should be treated only as **reported results** rather
   than fair head-to-head baselines?

---

## Audited paper files

### 1. FOMLN

- file:
  - `/home/shelterpl/cd_mambatt/docs/A first-order meta learning method for remaining useful life prediction of rotating machinery under limited samples.pdf`
- title:
  - *A first-order meta learning method for remaining useful life prediction of rotating machinery under limited samples*

### 2. MetaDFKN

- file:
  - `/home/shelterpl/cd_mambatt/docs/1-s2.0-S0951832024000036-main (1).pdf`
- title:
  - *Meta-learning with deep flow kernel network for few shot cross-domain remaining useful life prediction*

### 3. Task-Embedding MAML

- file:
  - `/home/shelterpl/cd_mambatt/docs/1-s2.0-S0166361525001617-main.pdf`
- title:
  - *A cross-domain few-shot remaining useful life estimation framework based on model-agnostic meta-learning with task embeddings*

---

## A. FOMLN audit

## A.1 What the paper actually uses

From the local PDF text:

- dataset:
  - `C-MAPSS`
- target sample regime:
  - **15-shot**
- input sensors:
  - **15 sensors**
  - sensor IDs:
    - `2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21`
- window length:
  - **30**
- RUL cap:
  - **125**
- multi-condition preprocessing:
  - **condition-based standardization (CS)** for `FD002/FD004`
- model:
  - **Conformer-based meta-learner**
- training:
  - inner-loop learning rate = **0.001**
  - outer-loop learning rate = **0.1**
  - inner loops = **10**
  - outer loops = **50**

## A.2 What the paper compares against

The cross-domain comparison table (`Table 6`) includes:

- `LSTM-DANN`
- `DIDRLSTM`
- `MAML based`
- `CADA`
- `VLSTM-LWSAN`
- `FOMLN`

Important detail from the paper text:

- the paper explicitly says several comparison methods use
  **large amounts of unlabeled target-domain samples** for domain
  adaptation
- FOMLN emphasizes that its own result is under a **15-shot** target setup

## A.3 What this means for us

### Comparable parts

- same `C-MAPSS` benchmark
- same 12 source→target transfer pairs
- same `RMSE` and `SCORE`
- same `RUL cap = 125`

### Non-comparable parts

- our current main protocol is **5-shot**
- FOMLN paper result is **15-shot**
- FOMLN also uses:
  - different sensor subset
  - different backbone
  - meta-learning protocol

## A.4 Reproduction value

**High.**

Why:

- it is clearly one of the strongest published methods against our current
  numbers
- the paper provides enough implementation detail to support a serious
  reimplementation
- unlike MetaDFKN, its target-shot setting is at least explicit

## A.5 Reproduction decision

> **FOMLN should be our first true published-baseline reproduction target.**

---

## B. MetaDFKN audit

## B.1 What the paper actually uses

From the local PDF text:

- dataset:
  - `C-MAPSS`
- sensors:
  - **14 sensors** with obvious degradation information
- target scenarios:
  - all **12 cross-domain transfer pairs**
- source/target split wording:
  - uses training sets as **source domains**
  - uses test sets as **target domains**
- RUL cap:
  - **125**
- key model choices:
  - flow kernel length finally chosen as **K = 10**
  - TinyCMN hidden layers finally chosen as **single GRU layer**

## B.2 Important ambiguity

The paper clearly reports strong cross-domain numbers, but the exact
**target-domain few-shot protocol** is not cleanly aligned with our current
pipeline from the extracted text.

The comparison section says the reported methods are first trained on source
domain data and then fine-tuned with a **limited quantity of target-domain
data**, described as `"Target-Unsupervised"` in the paper text.

At the same time, the C-MAPSS setup description mainly states:

- source = training sets
- target = test sets

This makes the exact fairness relationship with our current
`5-shot + target train split + target val split + resample` protocol
**uncertain**.

## B.3 What this means for us

### Comparable parts

- same benchmark family (`C-MAPSS`)
- same 12 transfer pairs
- same metrics

### Major risk

- the paper protocol is not yet audited enough to claim a fair apples-to-apples
  comparison with our `5-shot` setting

## B.4 Reproduction value

**Medium-high, but only after protocol audit.**

Why:

- MetaDFKN is numerically the strongest threat in reported tables
- but protocol mismatch risk is high
- reproducing it blindly could consume time before we even confirm the fair
  setup

## B.5 Reproduction decision

> **MetaDFKN should be audited first, then selectively reproduced.**

It should **not** be the first implementation target before protocol cleaning.

---

## C. Task-Embedding MAML audit

## C.1 What the paper actually uses

From the local PDF text:

- dataset:
  - `C-MAPSS`
- target sample regime:
  - **K = 1** in the reported C-MAPSS case-study setting
- source domain meta-task construction:
  - **30%** of base-domain training trajectories randomly selected
- target domain evaluation:
  - `K` run-to-failure trajectories selected from target training set as support
  - all truncation points in target test set used as query
- sensors:
  - **14 sensors**
- window length:
  - **30**
- optimizer:
  - `Adam`
- learning rates:
  - outer = **5e-5**
  - inner = **2e-5**
- repetitions:
  - reported results are averaged over **50 repetitions**

## C.2 Critical finding

The widely cited results such as:

- `FD001→FD003 = 24.34`
- `FD003→FD001 = 24.24`
- `FD002→FD004 = 26.96`
- `FD004→FD002 = 23.24`

come from a setting where the paper text states:

- **`K = 1`**

Therefore:

> these numbers are **not fair direct baselines** against our current
> `5-shot` results.

## C.3 Reproduction value

**Low for immediate reproduction.**

Why:

- the paper is useful as a literature comparison point
- but it is currently much less urgent than FOMLN
- and its reported numbers should not be over-interpreted against our
  `5-shot` pipeline

## C.4 Reproduction decision

> **Task-Embedding MAML should currently stay in the reported-baseline group,
> not the first reproduction group.**

---

## D. Final priority ranking

## D.1 True reproduction priority

1. **FOMLN**
2. **MetaDFKN** (after protocol audit)

## D.2 Reported-baseline priority

3. **Task-Embedding MAML**

---

## E. Immediate implementation plan

## E.1 First code target

Implement a new baseline:

- `train_fomln_baseline.py`

with a supporting model file such as:

- `cd_mambatt/models/fomln.py`

## E.2 Minimum reproduction scope for FOMLN

The first reproduction should align with the paper as closely as practical:

- `15-shot`
- selected 15 sensors
- window length `30`
- `RUL cap = 125`
- condition-based normalization for `FD002/FD004`
- same 12 transfer tasks
- report `RMSE` and `SCORE`

## E.3 Recommended reproduction phases

### Phase 1

Build a **minimal FOMLN-compatible data + model skeleton**

### Phase 2

Run one or two transfer pairs first:

- `FD001 -> FD003`
- `FD002 -> FD004`

### Phase 3

Expand to the full 12-task table if the partial runs are healthy

---

## F. Key conclusion

The most important finding from the actual local paper audit is:

> **The currently collected paper numbers are not equally comparable to our
> 5-shot protocol.**

More specifically:

- **FOMLN**: strong and worth reproducing first, but it is **15-shot**
- **MetaDFKN**: strong but protocol needs more auditing before fair comparison
- **Task-Embedding MAML**: the commonly cited table is effectively **1-shot**,
  so it should not be treated as a fair direct baseline against our current
  `5-shot` results
