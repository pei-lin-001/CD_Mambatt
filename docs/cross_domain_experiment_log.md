# Cross-Domain Experiment Log

Last updated: 2026-04-03

This file records the end-to-end cross-domain experiments for the
`CD-MambAtt` project. From now on, every completed run should be added
here with:

- task
- protocol
- command/output directory
- mean/std RMSE
- best RMSE
- short conclusion

---

## 1. Canonical Task

- task: `FD001 -> FD003`
- source subset: `FD001`
- target subset: `FD003`
- target-shot protocol: `5-shot`
- base seeds: `42,43,44`

This is currently the main debug and comparison task.

---

## 2. Baseline Progression

### 2.1 Direct transfer baseline

- date: 2026-03-31
- runner: `train_cross_domain_baseline.py`
- output:
  - `/home/shelterpl/cd_mambatt/runs/cross_domain_fd001_to_fd003_direct/FD001_TO_FD003`
- protocol:
  - source supervised only
  - no target fine-tuning
  - seeds = `42,43,44`
- results:
  - mean direct target RMSE = **33.1579**
  - std direct target RMSE = **7.9510**
  - best direct target RMSE = **25.7574**
- conclusion:
  - original MambAtt suffers large cross-domain degradation
  - this establishes the need for adaptation

### 2.2 5-shot head fine-tuning baseline

- date: 2026-03-31
- runner: `train_cross_domain_baseline.py`
- output:
  - `/home/shelterpl/cd_mambatt/runs/cross_domain_fd001_to_fd003_5shot_head/FD001_TO_FD003`
- protocol:
  - `5-shot`
  - target validation units = `10`
  - fixed target few-shot partition
  - fine-tune mode = `head`
  - seeds = `42,43,44`
- results:
  - mean fine-tune test RMSE = **30.0936**
  - std fine-tune test RMSE = **3.6504**
  - best fine-tune test RMSE = **24.9500**
- conclusion:
  - target few-shot supervision helps
  - head-only adaptation improves over direct transfer, but the gain is limited

### 2.3 5-shot full fine-tuning baseline

- date: 2026-03-31
- runner: `train_cross_domain_baseline.py`
- output:
  - `/home/shelterpl/cd_mambatt/runs/cross_domain_fd001_to_fd003_5shot_full/FD001_TO_FD003`
- protocol:
  - `5-shot`
  - target validation units = `10`
  - fixed target few-shot partition
  - fine-tune mode = `full`
  - seeds = `42,43,44`
- results:
  - mean fine-tune test RMSE = **21.8817**
  - std fine-tune test RMSE = **1.4535**
  - best fine-tune test RMSE = **19.9316**
- conclusion:
  - full fine-tuning is clearly stronger than head-only fine-tuning
  - this became the strongest baseline before adding alignment losses

### 2.4 5-shot full fine-tuning baseline with per-seed few-shot resampling

- date: 2026-03-31
- runner: `train_cross_domain_baseline.py`
- output:
  - `/home/shelterpl/cd_mambatt/runs/cross_domain_fd001_to_fd003_5shot_full_resample/FD001_TO_FD003`
- protocol:
  - `5-shot`
  - target validation units = `10`
  - resample target few-shot partition for each seed
  - fine-tune mode = `full`
  - seeds = `42,43,44`
- results:
  - mean fine-tune test RMSE = **22.7826**
  - std fine-tune test RMSE = **1.4936**
  - best fine-tune test RMSE = **23.3872**
- conclusion:
  - the stricter protocol is slightly harder than the fixed-partition protocol
  - this is the main strict baseline for later method comparisons

---

## 3. CD-MambAtt v1: MMD Alignment

### 3.1 First MMD version (`lambda_mmd = 0.1`)

- date: 2026-03-31
- runner: `train_cd_mambatt_v1.py`
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v1_fd001_to_fd003_mmd01_resample/FD001_TO_FD003`
- protocol:
  - source supervised pretraining
  - target few-shot supervised adaptation
  - target unlabeled pool participates through MMD
  - `lambda_mmd = 0.1`
  - `source_loss_weight = 1.0`
  - `target_loss_weight = 1.0`
  - `5-shot`
  - target validation units = `10`
  - resample target few-shot partition for each seed
  - seeds = `42,43,44`
- results:
  - mean direct target RMSE = **33.0534**
  - mean CD test RMSE = **21.8335**
  - std CD test RMSE = **1.7380**
  - best CD test RMSE by validation selection = **24.1041**
  - best observed CD test RMSE across seeds = **19.8832** (`seed = 44`)
- conclusion:
  - the minimal MMD version already beats the strict `5-shot full + resample` baseline
  - improvement over strict full fine-tuning baseline:
    - `22.7826 -> 21.8335`
    - gain ≈ **0.95 RMSE**

### 3.2 MMD weight sweep summary

- date: 2026-03-31
- runner: `train_cd_mambatt_v1.py`
- shared protocol:
  - `FD001 -> FD003`
  - `5-shot`
  - target validation units = `10`
  - resample target few-shot partition for each seed
  - seeds = `42,43,44`
  - `source_loss_weight = 1.0`
  - `target_loss_weight = 1.0`
- completed runs:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v1_fd001_to_fd003_mmd005_resample/FD001_TO_FD003`
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v1_fd001_to_fd003_mmd01_resample/FD001_TO_FD003`
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v1_fd001_to_fd003_mmd02_resample/FD001_TO_FD003`
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v1_fd001_to_fd003_mmd05_resample/FD001_TO_FD003`

| `lambda_mmd` | Mean CD RMSE | Std | Best test by val-selection | Best observed test |
|---:|---:|---:|---:|---:|
| 0.05 | 22.1885 | 1.4486 | 24.1420 | 20.6773 |
| 0.10 | **21.8335** | 1.7380 | 24.1041 | **19.8832** |
| 0.20 | 22.1105 | 1.4638 | 24.1202 | 20.6759 |
| 0.50 | 21.9006 | 1.7443 | 24.1469 | 19.8944 |

- conclusion:
  - `lambda_mmd = 0.1` is currently the best choice by **mean target RMSE**
  - `lambda_mmd = 0.5` is very close, but still slightly worse on the mean
  - the sweep suggests moderate alignment is beneficial, but increasing the
    MMD weight beyond `0.1` does not produce clear additional gains

### 3.3 CD-MambAtt v2: MMD + confidence-filtered pseudo-labeling

- date: 2026-03-31
- runner: `train_cd_mambatt_v2.py`
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v2_fd001_to_fd003_mmd01_pseudo11_resample_formal1/FD001_TO_FD003`
- protocol:
  - source supervised pretraining
  - target few-shot supervised adaptation
  - target unlabeled pool participates through:
    - `MMD`
    - source-stage classification
    - confidence-filtered target pseudo-stage classification
  - `lambda_mmd = 0.1`
  - `lambda_source_stage = 1.0`
  - `lambda_pseudo = 1.0`
  - `pseudo_num_stages = 3`
  - pseudo acceptance schedule quantile = `0.3 -> 0.7`
  - `5-shot`
  - target validation units = `10`
  - resample target few-shot partition for each seed
  - seeds = `42,43,44`
- results:
  - mean direct target RMSE = **33.1867**
  - mean CD test RMSE = **22.2637**
  - std CD test RMSE = **2.4844**
  - best CD test RMSE by validation selection = **25.4128**
  - best observed CD test RMSE across seeds = **19.3399** (`seed = 44`)
  - mean best-epoch pseudo acceptance ratio = **0.0673**
- per-seed snapshot:

| Seed | Best val RMSE | Test RMSE | Test SCORE | Best-epoch pseudo acceptance |
|---:|---:|---:|---:|---:|
| 42 | 11.4478 | 25.4128 | 1496.85 | 0.0630 |
| 43 | 18.3324 | 22.0382 | 2191.51 | 0.0261 |
| 44 | 19.9970 | **19.3399** | 906.05 | 0.1128 |

- conclusion:
  - the first formal pseudo-label version is **not yet better** than the best
    MMD-only v1 setting on mean RMSE
  - compared with v1 (`lambda_mmd = 0.1`), the mean worsened from
    **21.8335 -> 22.2637**
  - the acceptance ratio is low (**6.7%** on average), which strongly suggests
    that the current pseudo-label schedule is too conservative or too unstable
  - there is still upside because one seed reached **19.3399**, better than the
    best observed v1 seed-wise RMSE (**19.8832**), but the method is not yet
    stable enough

### 3.4 CD-MambAtt v2: relaxed pseudo schedule (`lambda_pseudo = 0.5`, `q: 0.5 -> 0.9`)

- date: 2026-03-31
- runner: `train_cd_mambatt_v2.py`
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v2_fd001_to_fd003_mmd01_pseudo05_q5090_resample_formal1/FD001_TO_FD003`
- protocol:
  - same as Section 3.3, except:
    - `lambda_pseudo = 0.5`
    - pseudo acceptance quantile schedule = `0.5 -> 0.9`
- results:
  - mean direct target RMSE = **33.0690**
  - mean CD test RMSE = **22.2134**
  - std CD test RMSE = **2.4842**
  - best CD test RMSE by validation selection = **25.3336**
  - best observed CD test RMSE across seeds = **19.2550** (`seed = 44`)
  - mean best-epoch pseudo acceptance ratio = **0.1409**
- per-seed snapshot:

| Seed | Best val RMSE | Test RMSE | Test SCORE | Best-epoch pseudo acceptance |
|---:|---:|---:|---:|---:|
| 42 | 11.4572 | 25.3336 | 1476.98 | 0.1376 |
| 43 | 18.2206 | 22.0518 | 2180.42 | 0.0766 |
| 44 | 19.8698 | **19.2550** | 901.62 | 0.2085 |

- conclusion:
  - relaxing the pseudo schedule roughly **doubled** acceptance
    (`0.0673 -> 0.1409`)
  - the mean RMSE improved slightly over the first v2 run
    (`22.2637 -> 22.2134`)
  - however, it still does **not** beat the best MMD-only v1 mean
    (**21.8335**)
  - this supports the next planned step from the PDF:
    add **cross-domain contrastive / N-tuplet alignment** on top of the
    stabilized pseudo-label pipeline

### 3.5 CD-MambAtt v2 + cross-domain contrastive (`lambda_contrastive = 0.1`)

- date: 2026-03-31
- runner: `train_cd_mambatt_v2.py`
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v2_fd001_to_fd003_mmd01_pseudo05_q5090_contrastive01_resample_formal1/FD001_TO_FD003`
- protocol:
  - same as Section 3.4, plus:
    - `lambda_contrastive = 0.1`
    - `contrastive_temperature = 0.1`
  - implementation detail:
    - source features are used as anchors
    - target labeled features + accepted target pseudo-labeled features form
      the cross-domain bank
    - same-stage pairs are positives; different-stage pairs are negatives
- results:
  - mean direct target RMSE = **33.1334**
  - mean CD test RMSE = **22.2511**
  - std CD test RMSE = **2.4527**
  - best CD test RMSE by validation selection = **25.3535**
  - best observed CD test RMSE across seeds = **19.3563** (`seed = 44`)
  - mean best-epoch pseudo acceptance ratio = **0.1380**
  - mean best-epoch contrastive valid-anchor ratio = **0.9969**
- per-seed snapshot:

| Seed | Best val RMSE | Test RMSE | Test SCORE | Pseudo acceptance | Contrastive valid-anchor |
|---:|---:|---:|---:|---:|---:|
| 42 | 11.4280 | 25.3535 | 1481.95 | 0.1346 | 1.0000 |
| 43 | 18.2125 | 22.0436 | 2148.51 | 0.0759 | 1.0000 |
| 44 | 19.9626 | **19.3563** | 912.63 | 0.2034 | 0.9906 |

- conclusion:
  - the contrastive term runs correctly and has near-complete valid-anchor
    coverage
  - however, with `lambda_contrastive = 0.1`, it does **not** improve the mean
    over the relaxed pseudo-only setting
    (`22.2134 -> 22.2511`)
  - at this point, the best mean result is still the simpler
    **v1 MMD-only** configuration (`21.8335`)
  - likely next moves:
    - try a **smaller** contrastive weight (e.g. `0.05`)
    - or implement the remaining PDF step: **local monotonicity loss**

### 3.6 CD-MambAtt v2 + cross-domain contrastive (`lambda_contrastive = 0.05`)

- date: 2026-03-31
- runner: `train_cd_mambatt_v2.py`
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v2_fd001_to_fd003_mmd01_pseudo05_q5090_contrastive005_resample_formal1/FD001_TO_FD003`
- protocol:
  - same as Section 3.4, plus:
    - `lambda_contrastive = 0.05`
    - `contrastive_temperature = 0.1`
- results:
  - mean direct target RMSE = **33.0628**
  - mean CD test RMSE = **22.2396**
  - std CD test RMSE = **2.5070**
  - best CD test RMSE by validation selection = **25.4181**
  - best observed CD test RMSE across seeds = **19.2901** (`seed = 44`)
  - mean best-epoch pseudo acceptance ratio = **0.1402**
  - mean best-epoch contrastive valid-anchor ratio = **0.9974**
- per-seed snapshot:

| Seed | Best val RMSE | Test RMSE | Test SCORE | Pseudo acceptance | Contrastive valid-anchor |
|---:|---:|---:|---:|---:|---:|
| 42 | 11.4579 | 25.4181 | 1495.81 | 0.1350 | 1.0000 |
| 43 | 18.3780 | 22.0106 | 2179.35 | 0.0774 | 1.0000 |
| 44 | 19.8813 | **19.2901** | 902.69 | 0.2083 | 0.9923 |

- conclusion:
  - reducing the contrastive weight from `0.1` to `0.05` gives a **small**
    mean improvement (`22.2511 -> 22.2396`)
  - but it still does **not** beat:
    - relaxed pseudo-only (`22.2134`)
    - best MMD-only v1 (`21.8335`)
  - therefore, the current evidence is:
    - the cross-domain contrastive term is implemented and functioning
    - but by itself it is **not yet the missing ingredient**
    - the most valuable next step is now the remaining PDF module:
      **local monotonicity loss**

### 3.7 CD-MambAtt v2 + local monotonicity (`lambda_monotonic = 0.1`)

- date: 2026-03-31
- runner: `train_cd_mambatt_v2.py`
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v2_fd001_to_fd003_mmd01_pseudo05_q5090_monotonic01_gap1s5_resample_formal1/FD001_TO_FD003`
- protocol:
  - base setting:
    - `lambda_mmd = 0.1`
    - `lambda_pseudo = 0.5`
    - pseudo quantile schedule = `0.5 -> 0.9`
  - monotonic setting:
    - `lambda_monotonic = 0.1`
    - `monotonic_pair_gap = 1`
    - `monotonic_pair_stride = 5`
    - monotonic pairs built from target labeled + unlabeled engines
  - no contrastive term in this run (`lambda_contrastive = 0.0`)
- results:
  - mean direct target RMSE = **33.0485**
  - mean CD test RMSE = **21.8611**
  - std CD test RMSE = **1.2578**
  - best CD test RMSE by validation selection = **23.3897**
  - best observed CD test RMSE across seeds = **20.3090** (`seed = 43`)
  - mean best-epoch pseudo acceptance ratio = **0.1686**
- per-seed snapshot:

| Seed | Best val RMSE | Test RMSE | Test SCORE | Pseudo acceptance | Monotonic pairs |
|---:|---:|---:|---:|---:|---:|
| 42 | 11.5034 | 23.3897 | 1283.30 | 0.0609 | 4177 |
| 43 | 19.9900 | **20.3090** | 1448.12 | 0.1439 | 4096 |
| 44 | 21.8045 | 21.8845 | 1423.39 | 0.3010 | 4125 |

- conclusion:
  - this is the **first extension that materially closes the gap** to the best
    v1 MMD-only mean:
    - `22.2134 -> 21.8611`
  - compared with relaxed pseudo-only, local monotonicity gives:
    - better **mean RMSE**
    - much lower **variance** (`2.4842 -> 1.2578`)
  - it still misses the current best mean (`21.8335`) by only about **0.03 RMSE**
  - this strongly suggests the PDF’s monotonicity idea is the most promising
    remaining component so far

### 3.8 CD-MambAtt v2 + local monotonicity (`lambda_monotonic = 0.05`)

- date: 2026-03-31
- runner: `train_cd_mambatt_v2.py`
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v2_fd001_to_fd003_mmd01_pseudo05_q5090_monotonic005_gap1s5_resample_formal1/FD001_TO_FD003`
- protocol:
  - base setting:
    - `lambda_mmd = 0.1`
    - `lambda_pseudo = 0.5`
    - pseudo quantile schedule = `0.5 -> 0.9`
  - monotonic setting:
    - `lambda_monotonic = 0.05`
    - `monotonic_pair_gap = 1`
    - `monotonic_pair_stride = 5`
  - no contrastive term in this run (`lambda_contrastive = 0.0`)
- results:
  - mean direct target RMSE = **33.1894**
  - mean CD test RMSE = **21.1291**
  - std CD test RMSE = **1.5928**
  - best CD test RMSE by validation selection = **23.3410**
  - best observed CD test RMSE across seeds = **19.6539** (`seed = 44`)
  - mean best-epoch pseudo acceptance ratio = **0.1309**
- per-seed snapshot:

| Seed | Best val RMSE | Test RMSE | Test SCORE | Pseudo acceptance | Monotonic pairs |
|---:|---:|---:|---:|---:|---:|
| 42 | 11.4779 | 23.3410 | 1273.27 | 0.0612 | 4177 |
| 43 | 19.6115 | 20.3926 | 1434.46 | 0.1384 | 4096 |
| 44 | 21.9240 | **19.6539** | 838.05 | 0.1932 | 4125 |

- conclusion:
  - lowering the monotonic weight from `0.1` to `0.05` yields a **large**
    additional gain on the mean:
    - `21.8611 -> 21.1291`
  - this is now the **best mean result across all completed experiments**
  - compared with the previous best (`v1 MMD-only, 21.8335`), the gain is:
    - `21.8335 -> 21.1291`
    - improvement ≈ **0.70 RMSE**
  - this is the strongest evidence so far that the PDF’s local monotonicity
    term is genuinely valuable in the cross-domain setting

### 3.9 Full combination test: monotonic (`0.05`) + contrastive (`0.05`)

- date: 2026-03-31
- runner: `train_cd_mambatt_v2.py`
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v2_fd001_to_fd003_mmd01_pseudo05_q5090_monotonic005_contrastive005_gap1s5_resample_formal1/FD001_TO_FD003`
- protocol:
  - current best monotonic setting from Section 3.8
  - plus:
    - `lambda_contrastive = 0.05`
    - `contrastive_temperature = 0.1`
- results:
  - mean direct target RMSE = **33.1458**
  - mean CD test RMSE = **22.3660**
  - std CD test RMSE = **0.7207**
  - best CD test RMSE by validation selection = **23.3823**
  - best observed CD test RMSE across seeds = **21.7909** (`seed = 43`)
  - mean best-epoch pseudo acceptance ratio = **0.2106**
  - mean best-epoch contrastive valid-anchor ratio = **0.9946**
- per-seed snapshot:

| Seed | Best val RMSE | Test RMSE | Test SCORE | Pseudo acceptance | Contrastive valid-anchor |
|---:|---:|---:|---:|---:|---:|
| 42 | 11.4942 | 23.3823 | 1280.52 | 0.0590 | 1.0000 |
| 43 | 19.6963 | **21.7909** | 2256.26 | 0.2752 | 1.0000 |
| 44 | 21.9190 | 21.9248 | 1443.16 | 0.2975 | 0.9838 |

- conclusion:
  - adding the contrastive term back on top of the best monotonic setting
    **significantly hurts** mean performance:
    - `21.1291 -> 22.3660`
  - although variance becomes very small, the whole run shifts to a worse
    accuracy region
  - practical conclusion:
    - for the current implementation and protocol, **monotonic helps**
    - **contrastive hurts**
    - the best known setting remains the simpler
      `MMD + pseudo + monotonic(0.05)` configuration

### 3.10 CD-MambAtt v2 + local monotonicity (`lambda_monotonic = 0.02`)

- date: 2026-03-31
- runner: `train_cd_mambatt_v2.py`
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v2_fd001_to_fd003_mmd01_pseudo05_q5090_monotonic002_gap1s5_resample_formal1/FD001_TO_FD003`
- protocol:
  - same as Section 3.8, except:
    - `lambda_monotonic = 0.02`
- results:
  - mean direct target RMSE = **33.2340**
  - mean CD test RMSE = **21.6954**
  - std CD test RMSE = **1.5317**
  - best CD test RMSE by validation selection = **23.3711**
  - best observed CD test RMSE across seeds = **19.6688** (`seed = 44`)
  - mean best-epoch pseudo acceptance ratio = **0.3161**
- per-seed snapshot:

| Seed | Best val RMSE | Test RMSE | Test SCORE | Pseudo acceptance | Monotonic pairs |
|---:|---:|---:|---:|---:|---:|
| 42 | 11.5008 | 23.3711 | 1279.00 | 0.0593 | 4177 |
| 43 | 19.0631 | 22.0462 | 2281.51 | 0.6879 | 4096 |
| 44 | 21.9510 | **19.6688** | 840.70 | 0.2011 | 4125 |

- conclusion:
  - lowering the monotonic weight further from `0.05` to `0.02` **hurts**
    the mean:
    - `21.1291 -> 21.6954`
  - however it is still better than:
    - MMD-only best (`21.8335`)
    - monotonic `0.1` (`21.8611`)
  - current monotonic tuning conclusion:
    - `0.05` is the best tested weight so far

---

## 4. Current Ranking on `FD001 -> FD003`

Sorted by mean target-side performance:

| Method | Protocol | Mean RMSE | Std | Best RMSE |
|---|---|---:|---:|---:|
| CD-MambAtt v2 + pseudo + monotonic (`0.05`) | 5-shot, resample | **21.1291** | 1.5928 | 19.6539 |
| CD-MambAtt v2 + pseudo + monotonic (`0.02`) | 5-shot, resample | 21.6954 | 1.5317 | 19.6688 |
| CD-MambAtt v1 + MMD (`lambda=0.1`) | 5-shot, resample | 21.8335 | 1.7380 | 19.8832* |
| CD-MambAtt v2 + pseudo + monotonic (`0.1`) | 5-shot, resample | 21.8611 | **1.2578** | 20.3090 |
| CD-MambAtt v1 + MMD (`lambda=0.5`) | 5-shot, resample | 21.9006 | 1.7443 | 19.8944 |
| Full fine-tune baseline | 5-shot, fixed partition | 21.8817 | 1.4535 | 19.9316 |
| CD-MambAtt v2 + pseudo (`0.5`, `q:0.5->0.9`) | 5-shot, resample | 22.2134 | 2.4842 | **19.2550** |
| CD-MambAtt v2 + pseudo + monotonic (`0.05`) + contrastive (`0.05`) | 5-shot, resample | 22.3660 | 0.7207 | 21.7909 |
| CD-MambAtt v2 + pseudo + contrastive (`0.05`) | 5-shot, resample | 22.2396 | 2.5070 | 19.2901 |
| CD-MambAtt v2 + pseudo + contrastive (`0.1`) | 5-shot, resample | 22.2511 | 2.4527 | 19.3563 |
| CD-MambAtt v2 + MMD + pseudo (`1.0/1.0`) | 5-shot, resample | 22.2637 | 2.4844 | **19.3399** |
| CD-MambAtt v1 + MMD (`lambda=0.2`) | 5-shot, resample | 22.1105 | 1.4638 | 20.6759 |
| CD-MambAtt v1 + MMD (`lambda=0.05`) | 5-shot, resample | 22.1885 | 1.4486 | 20.6773 |
| Full fine-tune baseline | 5-shot, resample | 22.7826 | 1.4936 | 23.3872 |
| Head fine-tune baseline | 5-shot, fixed partition | 30.0936 | 3.6504 | 24.9500 |
| Direct transfer | no fine-tune | 33.1579 | 7.9510 | 25.7574 |

\* `19.8832` is the best observed seed-wise test RMSE. The validation-selected
best seed for the current best-mean MMD run gave `24.1041`, which indicates
the current few-shot validation split is still noisy.

---

## 5. Current Takeaway

At the current stage, the most reliable statement is:

- the original MambAtt direct transfer degrades heavily on `FD001 -> FD003`
- `5-shot` target fine-tuning recovers a large part of the lost performance
- adding MMD on top of the `5-shot` setup gives a further consistent gain
- the first pseudo-label extension is promising in **best-case** seed
  performance, but currently loses to MMD-only on the **mean**
- relaxing pseudo acceptance improves stability a little, but not enough;
  the next missing ingredient from the PDF is the cross-domain
  contrastive/N-tuplet term
- the cross-domain contrastive implementation is now in place and functioning,
  but neither `lambda=0.1` nor `lambda=0.05` improved the mean result
- combining contrastive with the best monotonic setting also hurts, so the
  current evidence is quite strong that the present contrastive formulation is
  not beneficial
- the local monotonicity loss is the first new module that clearly helps on the
  **mean** while also improving **stability**
- within the tested monotonic weights, `0.05` is the current best
- the best mean result so far is:
  - `CD-MambAtt v2 + pseudo + monotonic`
  - `lambda_mmd = 0.1`
  - `lambda_pseudo = 0.5`
  - `lambda_monotonic = 0.05`
  - mean RMSE = **21.1291**

## 6. Recommended Next Step

The next reasonable extension is now:

1. stabilize pseudo-label coverage
   - relax the pseudo acceptance schedule (higher quantiles)
   - and/or reduce `lambda_pseudo`
2. improve target-side selection stability
   - enlarge target validation units
   - or repeat with more seeds
3. optionally test neighboring monotonic weights around the current best
   - e.g. `0.03` or `0.04`

---

## 7. Submission-stage Generalization Runs (`5 seeds`)

After the first `FD001 -> FD003` module validation stage, the current best
configuration was frozen and evaluated on additional transfer pairs.

### 7.1 Fixed best configuration

```bash
--lambda-mmd 0.1
--lambda-source-stage 1.0
--lambda-pseudo 0.5
--pseudo-start-quantile 0.5
--pseudo-end-quantile 0.9
--lambda-monotonic 0.05
--monotonic-pair-gap 1
--monotonic-pair-stride 5
--lambda-contrastive 0.0
--target-shots 5
--target-val-units 10
--seeds 42,43,44,45,46
```

### 7.2 Important implementation fix before multi-condition runs

- date: 2026-04-01
- file:
  - `/home/shelterpl/cd_mambatt/cd_mambatt/data.py`
- issue:
  - the multi-condition normalization path used raw floating-point operating
    settings as condition keys
  - this created many fake conditions on `FD002/FD004`, and earlier validation
    metrics could explode to extremely large values
- fix:
  - operating settings are now rounded to integer-level condition IDs before
    grouping
- consequence:
  - old exploding `FD002/FD004` outputs should be treated as invalid
  - the re-run results below are the valid ones

### 7.3 `FD003 -> FD001`

- date: 2026-04-01
- runner: `train_cd_mambatt_v2.py`
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_bestcfg_fd003_to_fd001_5seeds/FD003_TO_FD001`
- results:
  - mean direct target RMSE = **24.6059**
  - std direct target RMSE = **2.6170**
  - mean CD test RMSE = **19.8137**
  - std CD test RMSE = **1.2628**
  - best CD test RMSE by validation selection = **19.8305**
  - improvement over direct transfer = **4.7922 RMSE** (**19.5%**)
  - mean best-epoch pseudo acceptance ratio = **0.1074**
- conclusion:
  - reverse transfer also benefits clearly
  - this shows the method is not only effective in the original
    `FD001 -> FD003` direction

### 7.4 `FD002 -> FD004` (multi-condition, fixed normalization)

- date: 2026-04-01
- runner: `train_cd_mambatt_v2.py`
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_bestcfg_fd002_to_fd004_5seeds_fixcond/FD002_TO_FD004`
- results:
  - mean direct target RMSE = **29.5028**
  - std direct target RMSE = **1.1257**
  - mean CD test RMSE = **22.1776**
  - std CD test RMSE = **2.1685**
  - best CD test RMSE by validation selection = **20.6206**
  - improvement over direct transfer = **7.3252 RMSE** (**24.8%**)
  - mean best-epoch pseudo acceptance ratio = **0.0384**
- conclusion:
  - after fixing condition normalization, the method remains stable and gives a
    large gain on a multi-condition transfer task
  - this is an important generalization result beyond the single-condition case

### 7.5 `FD001 -> FD004` (hard transfer, fixed normalization)

- date: 2026-04-01
- runner: `train_cd_mambatt_v2.py`
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_bestcfg_fd001_to_fd004_5seeds_fixcond/FD001_TO_FD004`
- results:
  - mean direct target RMSE = **32.3820**
  - std direct target RMSE = **5.2311**
  - mean CD test RMSE = **24.3449**
  - std CD test RMSE = **2.4762**
  - best CD test RMSE by validation selection = **21.2799**
  - improvement over direct transfer = **8.0371 RMSE** (**24.8%**)
  - mean best-epoch pseudo acceptance ratio = **0.1206**
- conclusion:
  - the current method still gives a strong improvement on a harder transfer
    pair
  - this substantially strengthens the claim that the method is not just
    overfitted to the easy canonical task

### 7.6 `FD004 -> FD002` (multi-condition reverse transfer, fixed normalization)

- date: 2026-04-01
- runner: `train_cd_mambatt_v2.py`
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_bestcfg_fd004_to_fd002_5seeds_fixcond/FD004_TO_FD002`
- results:
  - mean direct target RMSE = **21.3666**
  - std direct target RMSE = **0.6723**
  - mean CD test RMSE = **19.5311**
  - std CD test RMSE = **1.0414**
  - best CD test RMSE by validation selection = **19.3835**
  - improvement over direct transfer = **1.8354 RMSE** (**8.6%**)
  - mean best-epoch pseudo acceptance ratio = **0.0412**
- conclusion:
  - this task is noticeably harder to improve because the direct-transfer
    baseline is already relatively strong
  - even so, the current method still gives a consistent positive gain
  - this is useful evidence that the method is not dependent on only one
    transfer direction inside the multi-condition setting

### 7.7 `FD001 -> FD002` (single-condition to multi-condition)

- date: 2026-04-01
- runner: `train_cd_mambatt_v2.py`
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_bestcfg_fd001_to_fd002_5seeds/FD001_TO_FD002`
- results:
  - mean direct target RMSE = **31.1029**
  - std direct target RMSE = **5.9483**
  - mean CD test RMSE = **22.9457**
  - std CD test RMSE = **2.8311**
  - best CD test RMSE by validation selection = **20.0879**
  - improvement over direct transfer = **8.1572 RMSE** (**26.2%**)
  - mean best-epoch pseudo acceptance ratio = **0.1156**
- conclusion:
  - this is a strong stage-2 transfer result, because the target subset is
    multi-condition while the source subset is single-condition
  - the method again shows a large improvement, which strengthens the claim
    that the current recipe generalizes beyond the original stage-1 tasks

### 7.8 Multi-task summary table

| Task | Direct RMSE | CD RMSE | Gain | Gain % | Best CD RMSE |
|---|---:|---:|---:|---:|---:|
| `FD001 -> FD002` | 31.1029 | **22.9457** | 8.1572 | 26.2% | 20.0879 |
| `FD003 -> FD001` | 24.6059 | **19.8137** | 4.7922 | 19.5% | 19.8305 |
| `FD002 -> FD004` | 29.5028 | **22.1776** | 7.3252 | 24.8% | 20.6206 |
| `FD001 -> FD004` | 32.3820 | **24.3449** | 8.0371 | 24.8% | 21.2799 |
| `FD004 -> FD002` | 21.3666 | **19.5311** | 1.8354 | 8.6% | 19.3835 |

---

## 8. Current Overall Takeaway

The evidence is now stronger than it was during the first
`FD001 -> FD003`-only stage:

- the current working CD-MambAtt configuration improves over direct transfer on
  **multiple task pairs**
- the gains hold for:
  - single-condition to multi-condition transfer
  - reverse single-condition transfer
  - multi-condition transfer
  - a harder cross-subset transfer
  - reverse multi-condition transfer
- the most reliable working recipe is still:
  - **MMD + confidence-filtered pseudo-labeling + local monotonicity**
- the current contrastive formulation remains unsupported by the experiments

This means the project has moved from:

- “single-task proof of concept”

to:

- “multi-task evidence that the method generalizes on C-MAPSS”

---

## 9. Canonical task formal re-check (`FD001 -> FD003`, `5 seeds`)

The earlier best canonical result for the working CD-MambAtt recipe,
`21.1291`, came from a `3-seed` run. To make the canonical task directly
comparable with the new `5-seed` multi-task suite, we reran it with the same
formal protocol.

### 9.1 Canonical `5-seed` rerun with the cross-task default (`lambda_monotonic = 0.05`)

- date: 2026-04-01
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_bestcfg_fd001_to_fd003_5seeds/FD001_TO_FD003`
- results:
  - mean direct target RMSE = **34.7179**
  - mean CD test RMSE = **22.0342**
  - std CD test RMSE = **1.4408**
  - best CD test RMSE by validation selection = **23.3685**
  - improvement over direct transfer = **12.6837 RMSE** (**36.5%**)
- conclusion:
  - the method still strongly beats direct transfer on the canonical task
  - however, the `5-seed` estimate is weaker than the earlier `3-seed`
    estimate, which means we should treat the earlier `21.1291` as somewhat
    optimistic

### 9.2 Canonical `5-seed` stability check (`lambda_monotonic = 0.1`)

- date: 2026-04-01
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_formal_fd001_to_fd003_monotonic01_5seeds/FD001_TO_FD003`
- results:
  - mean direct target RMSE = **34.8204**
  - mean CD test RMSE = **21.9608**
  - std CD test RMSE = **1.4574**
  - best CD test RMSE by validation selection = **23.3842**
  - improvement over direct transfer = **12.8595 RMSE** (**36.9%**)
- conclusion:
  - on the canonical `5-seed` rerun, monotonic `0.1` is only **slightly**
    better than `0.05`
  - the gap is small enough that the two settings should currently be treated
    as a near-tie under the formal protocol

### 9.3 Canonical formal comparison table

| Canonical setting | Seeds | Mean CD RMSE | Std | Best CD RMSE |
|---|---:|---:|---:|---:|
| monotonic `0.05` | 5 | 22.0342 | 1.4408 | 23.3685 |
| monotonic `0.1` | 5 | **21.9608** | 1.4574 | 23.3842 |
| monotonic `0.05` | 3 | **21.1291** | 1.5928 | 23.3410 |

### 9.4 Refined interpretation

- the cross-task evidence for the method remains strong
- the canonical task still shows a clear improvement over direct transfer
- but the exact best monotonic weight is **not yet sharply separated** under
  `5-seed` evaluation
- practically:
  - `0.05` remains the current cross-task default because the new multi-task
    suite was run with it
  - `0.1` is now a serious candidate for the canonical task because it is
    marginally better in the new `5-seed` rerun

---

## 10. Published baseline reproduction track: FOMLN

### 10.1 Status reset

- date: 2026-04-01
- related files:
  - `/home/shelterpl/cd_mambatt/train_fomln_baseline.py`
  - `/home/shelterpl/cd_mambatt/cd_mambatt/models/fomln.py`
  - `/home/shelterpl/cd_mambatt/docs/fomln_baseline_progress_2026-04-01.md`
- conclusion:
  - the initial FOMLN implementation was **only a runnable skeleton**
  - it is **not** accepted as a formal published-baseline reproduction
  - all early `runs/fomln_smoke*` and `runs/fomln_gpu_stage*` directories
    should be treated as:
    - environment / smoke / debugging artifacts

### 10.2 Reproduction upgrade now completed

- date: 2026-04-01
- current changes:
  - custom attention implementation with:
    - `h = 8`
    - `d_k = d_v = 64`
  - working conformer width default is now:
    - `d_model = 512`
  - rationale:
    - Table 1 reports `h = 8`, `d_k = d_v = 64`
    - a same-domain alignment probe strongly favored `512` over `10`
  - front-end conv now follows the paper more directly:
    - in-channels `15`
    - out-channels `10`
    - kernel `10`
    - stride `1`
  - source training now uses explicit source meta-tasks
  - target adaptation now uses a fixed `15-shot` support set
  - CUDA is mandatory for this runner
- important note:
  - the implementation is now **paper-alignment-in-progress**
  - but it is **still not yet claimed as a final exact reproduction**
  - the main remaining ambiguity is the paper's lack of a precise
    source meta-task construction rule

### 10.3 Post-reset CUDA runtime validation

- date: 2026-04-01
- output:
  - `/home/shelterpl/cd_mambatt/runs/fomln_repro_cuda_smoke_v3/FD001_TO_FD003`
- protocol:
  - CUDA-only runtime validation
  - `FD001 -> FD003`
  - `15-shot`
  - fixed partition of source windows into meta-tasks
  - `source-task-size = 15`
  - `outer-loops = 1`
  - `meta-batch-tasks = 1`
  - `inner-steps = 1`
  - `adapt-steps = 1`
  - `target-scale = 125`
- results:
  - source meta-task count = **9**
  - mean direct target RMSE = **83.4471**
  - mean adapted target RMSE = **82.7856**
- conclusion:
  - the revised FOMLN code path now runs end-to-end on **CUDA**
  - this run is a **post-refactor implementation check**
  - it should not yet be cited as a formal FOMLN baseline result

### 10.4 Revised FOMLN candidate stability run

- date: 2026-04-01
- output:
  - `/home/shelterpl/cd_mambatt/runs/fomln_repro_cuda_candidate1/FD001_TO_FD003`
- protocol:
  - CUDA-only reproduction-development run
  - `FD001 -> FD003`
  - `15-shot`
  - fixed partition of source windows into meta-tasks
  - `source-task-size = 15`
  - `outer-loops = 10`
  - `meta-batch-tasks = 4`
  - `inner-steps = 10`
  - `adapt-steps = 10`
  - `target-scale = 125`
  - `grad-clip-norm = 1.0`
- results:
  - source meta-task count = **137**
  - mean direct target RMSE = **71.5558**
  - mean adapted target RMSE = **69.1342**
- conclusion:
  - the revised code path remains stable under a much stronger CUDA setting
  - target adaptation still improves over direct transfer
  - but the absolute error is still far from the published FOMLN result, so
    this remains a **reproduction-development checkpoint**, not a formal
    comparison line for the paper table

### 10.5 Same-domain width ambiguity probe (`FD001`, `15-shot`)

- date: 2026-04-01
- outputs:
  - `/home/shelterpl/cd_mambatt/runs/fomln_same_fd001_dmodel10_probe/FD001_TO_FD003`
  - `/home/shelterpl/cd_mambatt/runs/fomln_same_fd001_dmodel512_probe/FD001_TO_FD003`
- protocol:
  - same-domain reproduction-development probe
  - subset = `FD001`
  - `15-shot`
  - `outer-loops = 10`
  - `meta-batch-tasks = 4`
  - `source-task-size = 15`
  - `inner-steps = 10`
  - `adapt-steps = 10`
- results:
  - `d_model = 10`
    - direct RMSE = **72.6529**
    - adapted RMSE = **66.1854**
  - `d_model = 512`
    - direct RMSE = **53.3864**
    - adapted RMSE = **38.1539**
- conclusion:
- the paper does not explicitly print the final conformer hidden width
- but this probe strongly suggests that using:
  - **`d_model = 512`**
    is more plausible than using `10`
- the FOMLN runner default has therefore been moved to `512`

### 10.6 Paper-aligned meta-update correction and optimizer audit

- date: 2026-04-01
- related files:
  - `/home/shelterpl/cd_mambatt/train_fomln_baseline.py`
- new implementation changes:
  - source meta-update corrected from an averaged Reptile-style update to the
    paper-like **sequential interpolation**
    - `Φ ← (1 - α) Φ + α Φ_j`
  - source inner-loop optimizer default moved to:
    - **`Adam`**
  - target adaptation optimizer can now be configured separately
- conclusion:
  - the FOMLN runner is now more closely aligned with the algorithm text
    extracted from the local paper PDF
  - but the target adaptation optimizer is still treated as an implementation
    ambiguity that must be validated empirically

### 10.7 All-Adam instability probe

- date: 2026-04-01
- outputs:
  - `/home/shelterpl/cd_mambatt/runs/fomln_seqadam_same_fd001_probe/FD001_TO_FD003`
  - `/home/shelterpl/cd_mambatt/runs/fomln_seqadam_fd001_to_fd003_probe/FD001_TO_FD003`
- protocol:
  - `d_model = 512`
  - sequential meta-update
  - source task level = `windows`
  - target shot level = `windows`
  - source inner optimizer = `Adam`
  - target adaptation optimizer = `Adam`
  - `outer-loops = 50`
  - `inner-steps = 10`
  - `adapt-steps = 10`
  - `target-scale = 125`
  - `grad-clip-norm = 1.0`
- results:
  - same-domain `FD001 -> FD001`
    - direct RMSE = **25.1680**
    - adapted RMSE = **61.5891**
  - cross-domain `FD001 -> FD003`
    - direct RMSE = **36.9894**
    - adapted RMSE = **107.2370**
- conclusion:
  - the corrected sequential meta-update itself is not the problem
  - the severe failure appears during the **target adaptation stage** when
    using `Adam` directly
  - therefore target adaptation optimizer was decoupled from source inner-loop
    optimizer in the runner

### 10.8 Decoupled optimizer probe: source `Adam`, target `SGD`

- date: 2026-04-01
- outputs:
  - `/home/shelterpl/cd_mambatt/runs/fomln_seqadam_meta_sgd_adapt_same_fd001_probe/FD001_TO_FD003`
  - `/home/shelterpl/cd_mambatt/runs/fomln_seqadam_meta_sgd_adapt_fd001_to_fd003_probe/FD001_TO_FD003`
- protocol:
  - same as Section 10.7, except:
    - target adaptation optimizer = `SGD`
- results:
  - same-domain `FD001 -> FD001`
    - direct RMSE = **25.1680**
    - adapted RMSE = **21.6903**
  - cross-domain `FD001 -> FD003`
    - direct RMSE = **36.9894**
    - adapted RMSE = **34.9648**
- conclusion:
  - replacing target-side `Adam` with `SGD` removes the numerical blow-up
  - however, pure window-level `15-shot` remains too weak to explain the
    paper's reported performance

### 10.9 `15-shot` interpretation probe: windows vs units

- date: 2026-04-01
- output:
  - `/home/shelterpl/cd_mambatt/runs/fomln_seqadam_meta_sgd_adapt_same_fd001_unitshot_probe/FD001_TO_FD003`
- protocol:
  - source task level = `windows`
  - target shot level = **`units`**
  - source inner optimizer = `Adam`
  - target adaptation optimizer = `SGD`
  - other settings unchanged from Section 10.8
- results:
  - same-domain `FD001 -> FD001`
    - direct RMSE = **25.1680**
    - adapted RMSE = **19.6785**
- conclusion:
  - simply interpreting the paper's `15-shot` as **15 units / trajectories**
    instead of 15 random sliding windows yields a clear improvement
  - this is the strongest current clue that the paper's support protocol is
    trajectory-level in practice

### 10.10 Current strongest FOMLN reproduction-development candidate

- date: 2026-04-01
- outputs:
  - `/home/shelterpl/cd_mambatt/runs/fomln_seqadam_metaunits_sgd_adapt_same_fd001_unitshot_probe/FD001_TO_FD003`
  - `/home/shelterpl/cd_mambatt/runs/fomln_seqadam_metaunits_sgd_adapt_fd001_to_fd003_probe/FD001_TO_FD003`
- protocol:
  - source task level = **`units`**
  - source task size = `15`
  - target shot level = **`units`**
  - target shots = `15`
  - source inner optimizer = `Adam`
  - target adaptation optimizer = `SGD`
  - `d_model = 512`
  - `outer-loops = 50`
  - `meta-batch-tasks = 4`
  - `inner-steps = 10`
  - `adapt-steps = 10`
  - `target-scale = 125`
  - `grad-clip-norm = 1.0`
- results:
  - same-domain `FD001 -> FD001`
    - direct RMSE = **26.3622**
    - adapted RMSE = **14.9430**
  - cross-domain `FD001 -> FD003`
    - direct RMSE = **32.2795**
    - adapted RMSE = **21.4792**
- conclusion:
  - this is the best FOMLN reproduction-development configuration obtained so
    far
  - compared with the earlier strong window-level candidate
    (`FD001 -> FD003`, adapted RMSE **33.0446**), the new interpretation
    improves to **21.4792**
  - practical working hypothesis:
    - the paper's notion of "sample" is being operationalized at the
      **unit / trajectory level**
  - next step:
    - continue this configuration on additional transfer pairs

### 10.11 Additional FOMLN transfer validation: `FD003 -> FD001`

- date: 2026-04-01
- output:
  - `/home/shelterpl/cd_mambatt/runs/fomln_seqadam_metaunits_sgd_adapt_fd003_to_fd001_probe/FD003_TO_FD001`
- protocol:
  - current best reproduction-development candidate from Section 10.10
  - source task level = `units`
  - target shot level = `units`
  - source inner optimizer = `Adam`
  - target adaptation optimizer = `SGD`
  - `outer-loops = 50`
  - `inner-steps = 10`
  - `adapt-steps = 10`
- results:
  - direct RMSE = **17.6081**
  - adapted RMSE = **16.7696**
  - improvement = **0.8385**
- conclusion:
  - the unit-level interpretation generalizes beyond the canonical direction
  - adaptation gain is smaller here because direct transfer is already
    relatively strong on this simpler same-condition reverse task

### 10.12 Additional FOMLN transfer validation: `FD001 -> FD004`

- date: 2026-04-01
- output:
  - `/home/shelterpl/cd_mambatt/runs/fomln_seqadam_metaunits_sgd_adapt_fd001_to_fd004_probe/FD001_TO_FD004`
- protocol:
  - same as Section 10.11
- results:
  - direct RMSE = **34.4970**
  - adapted RMSE = **23.3202**
  - improvement = **11.1768**
- conclusion:
  - the current FOMLN reproduction-development candidate shows a strong gain
    on the harder `FD001 -> FD004` transfer
  - this is important because `FD004` is one of the more challenging target
    domains in the benchmark

### 10.13 Additional FOMLN transfer validation: `FD002 -> FD004`

- date: 2026-04-01
- output:
  - `/home/shelterpl/cd_mambatt/runs/fomln_seqadam_metaunits_sgd_adapt_fd002_to_fd004_probe/FD002_TO_FD004`
- protocol:
  - same as Section 10.11
- results:
  - direct RMSE = **29.0239**
  - adapted RMSE = **22.0029**
  - improvement = **7.0211**
- conclusion:
  - the current candidate also works on the multi-condition transfer
    `FD002 -> FD004`
  - across the newly tested tasks, adaptation improves over direct transfer in
    all cases

### 10.14 FOMLN reproduction-development snapshot after unit-level reinterpretation

- date: 2026-04-01
- current best single-seed results under the working candidate:

| Task | Direct RMSE | Adapted RMSE |
|---|---:|---:|
| `FD001 -> FD003` | 32.2795 | **21.4792** |
| `FD003 -> FD001` | 17.6081 | **16.7696** |
| `FD001 -> FD004` | 34.4970 | **23.3202** |
| `FD002 -> FD004` | 29.0239 | **22.0029** |

- conclusion:
  - the unit-level `15-shot` interpretation is now the strongest working
    explanation of the paper protocol
  - the next two most important steps are:
    1. test whether using all source unit-level meta-tasks per outer loop helps
    2. move this candidate to multi-seed evaluation

### 10.15 FOMLN source-task coverage probe: full unit-task meta-batch

- date: 2026-04-01
- output:
  - `/home/shelterpl/cd_mambatt/runs/fomln_seqadam_metaunitsFULL_sgd_adapt_fd001_to_fd003_probe/FD001_TO_FD003`
- protocol:
  - same as the current working candidate, except:
    - `meta-batch-tasks = 7`
    - this equals the full source unit-task count for `FD001`
- results:
  - direct RMSE = **33.6033**
  - adapted RMSE = **24.1990**
- conclusion:
  - using **all** source unit-level meta-tasks in every outer step is worse
    than sampling a subset
  - compared with the current candidate on the same task:
    - sampled unit-task meta-batch (`4`) -> **21.4792**
    - full unit-task meta-batch (`7`) -> **24.1990**
  - practical decision:
    - keep `meta-batch-tasks = 4` as the working default for the next round
      of multi-seed evaluation

### 10.16 FOMLN memory-stability correction: mini-batch inner loops

- date: 2026-04-01
- related file:
  - `/home/shelterpl/cd_mambatt/train_fomln_baseline.py`
- problem:
  - even with single-seed / single-process execution, the unit-level protocol
    could still push the RTX 4060 laptop into shared GPU memory spillover
- diagnosis:
  - the previous code moved the full source task and full target support pool
    onto CUDA for every inner/adaptation step
  - under unit-level `15-shot`, this means thousands of windows can hit the GPU
    at once
  - the paper's weighted loss is also written in a **batch-based** form
- code change:
  - source inner loop now samples a mini-batch per step
  - target adaptation loop now also samples a mini-batch per step
  - new flags:
    - `--inner-batch-size`
    - `--adapt-batch-size`
  - current stable setting:
    - `inner_batch_size = 256`
    - `adapt_batch_size = 256`
- conclusion:
  - this is both a memory fix and a more defensible interpretation of the
    paper's batch-based weighted loss

### 10.17 Mini-batch stability verification

- date: 2026-04-01
- output:
  - `/home/shelterpl/cd_mambatt/runs/fomln_minibatch_verify_fd003_to_fd001/FD003_TO_FD001`
- protocol:
  - current unit-level candidate + mini-batch inner loops (`256 / 256`)
- result:
  - `FD003 -> FD001`, `seed = 44`
    - direct RMSE = **24.5753**
    - adapted RMSE = **19.0092**
- conclusion:
  - the mini-batch version runs stably on CUDA without reproducing the
    previous VRAM/shared-memory problem

### 10.18 Formal FOMLN baseline: `FD003 -> FD001` (5 seeds, mini-batch protocol)

- date: 2026-04-01
- output:
  - `/home/shelterpl/cd_mambatt/runs/fomln_formal_mb256_unit15_5seeds_fd003_to_fd001/FD003_TO_FD001/summary_aggregated.json`
- protocol:
  - unit-level `15-shot`
  - source inner optimizer = `Adam`
  - target adaptation optimizer = `SGD`
  - `meta-batch-tasks = 4`
  - `inner/adapt batch size = 256 / 256`
  - seeds = `42,43,44,45,46`
- results:
  - mean direct RMSE = **25.0772**
  - mean adapted RMSE = **20.1261**
  - std = **1.3369**
  - best adapted RMSE = **18.0928**
- conclusion:
  - this is now the first completed **formal FOMLN 5-seed result** under the
    stable mini-batch protocol

### 10.19 Formal FOMLN baseline: `FD001 -> FD003` (5 seeds, mini-batch protocol)

- date: 2026-04-01
- output:
  - `/home/shelterpl/cd_mambatt/runs/fomln_formal_mb256_unit15_5seeds_fd001_to_fd003/FD001_TO_FD003/summary_aggregated.json`
- results:
  - mean direct RMSE = **34.0750**
  - mean adapted RMSE = **25.4748**
  - std = **1.9631**
  - best adapted RMSE = **22.2528**
- conclusion:
  - adaptation still clearly beats direct transfer
  - but this direction remains harder than `FD003 -> FD001`

### 10.20 Formal FOMLN baseline: `FD002 -> FD004` (5 seeds, mini-batch protocol)

- date: 2026-04-01
- output:
  - `/home/shelterpl/cd_mambatt/runs/fomln_formal_mb256_unit15_5seeds_fd002_to_fd004/FD002_TO_FD004/summary_aggregated.json`
- results:
  - mean direct RMSE = **31.1871**
  - mean adapted RMSE = **26.0206**
  - std = **2.7396**
  - best adapted RMSE = **22.6677**
- conclusion:
  - the stable formal baseline is now also available on a multi-condition task

### 10.21 Formal FOMLN baseline: `FD001 -> FD004` (5 seeds, mini-batch protocol)

- date: 2026-04-01
- output:
  - `/home/shelterpl/cd_mambatt/runs/fomln_formal_mb256_unit15_5seeds_fd001_to_fd004/FD001_TO_FD004/summary_aggregated.json`
- results:
  - mean direct RMSE = **30.9944**
  - mean adapted RMSE = **25.0808**
  - std = **1.4670**
  - best adapted RMSE = **23.6054**
- conclusion:
  - adaptation again improves over direct transfer across all seeds

### 10.22 Current formal FOMLN baseline table (stable mini-batch protocol)

- date: 2026-04-01

| Task | Mean direct RMSE | Mean adapted RMSE | Std | Best adapted RMSE |
|---|---:|---:|---:|---:|
| `FD001 -> FD003` | 34.0750 | **25.4748** | 1.9631 | 22.2528 |
| `FD003 -> FD001` | 25.0772 | **20.1261** | 1.3369 | 18.0928 |
| `FD002 -> FD004` | 31.1871 | **26.0206** | 2.7396 | 22.6677 |
| `FD001 -> FD004` | 30.9944 | **25.0808** | 1.4670 | 23.6054 |

- conclusion:
  - we now have a **stable, CUDA-only, 5-seed FOMLN baseline package**
    on four important transfer pairs
  - every task shows a clear gain from target adaptation over direct transfer
  - this table is the current published-baseline comparison anchor for the
    project until a stronger reproduction variant is found

### 10.23 Fair-comparison rerun: CD-MambAtt `15-shot` on `FD001 -> FD003`

- date: 2026-04-02
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_bestcfg_fd001_to_fd003_15shot_5seeds/FD001_TO_FD003/summary.json`
- protocol:
  - same best CD-MambAtt configuration used in the `5-shot` study
  - target shots increased from `5` to `15` for direct comparison with FOMLN
  - seeds = `42,43,44,45,46`
- results:
  - mean direct RMSE = **34.8152**
  - mean adapted RMSE = **20.1384**
  - std = **3.5060**
  - best adapted RMSE = **24.2581** (best-by-validation seed selection)
- conclusion:
  - under the same `15-shot` budget, CD-MambAtt still clearly beats the
    reproduced FOMLN baseline on the canonical task
  - fair head-to-head delta vs FOMLN:
    - `25.4748 -> 20.1384`
    - gain ≈ **5.34 RMSE**

### 10.24 Fair-comparison rerun: CD-MambAtt `15-shot` on `FD003 -> FD001`

- date: 2026-04-02
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_bestcfg_fd003_to_fd001_15shot_5seeds/FD003_TO_FD001/summary.json`
- results:
  - mean direct RMSE = **24.5970**
  - mean adapted RMSE = **19.5788**
  - std = **2.1954**
  - best adapted RMSE = **19.3647**
- conclusion:
  - CD-MambAtt also remains slightly better than the reproduced FOMLN baseline
    on the reverse transfer direction
  - fair head-to-head delta vs FOMLN:
    - `20.1261 -> 19.5788`
    - gain ≈ **0.55 RMSE**

### 10.25 Fair-comparison rerun: CD-MambAtt `15-shot` on `FD002 -> FD004`

- date: 2026-04-02
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_bestcfg_fd002_to_fd004_15shot_5seeds/FD002_TO_FD004/summary.json`
- results:
  - mean direct RMSE = **30.9390**
  - mean adapted RMSE = **23.2879**
  - std = **1.6107**
  - best adapted RMSE = **24.1634** (best-by-validation seed selection)
- conclusion:
  - the harder multi-condition transfer pair is now also completed under the
    matched `15-shot` protocol
  - CD-MambAtt still beats the reproduced FOMLN baseline:
    - `26.0206 -> 23.2879`
    - gain ≈ **2.73 RMSE**

### 10.26 Runtime incident note: no active stale training process, but WSL CUDA path temporarily lost

- date: 2026-04-02
- inspection:
  - `pgrep -af 'train_cd_mambatt_v2.py|train_fomln_baseline.py|python .*cd_mambatt'`
    returned no active training worker
  - WSL `nvidia-smi` failed with:
    - `Failed to initialize NVML: GPU access blocked by the operating system`
  - `conda run -n cd_mamba python -c "import torch; print(torch.cuda.is_available())"`
    returned `False`
  - Windows host `nvidia-smi` still works and sees the RTX 4060 normally
- conclusion:
  - the apparent `4.5h` process was not an actually running Python training
    job at inspection time
  - the current blocker is specifically the **WSL GPU bridge**, not the Windows
    NVIDIA driver itself
  - the remaining fair-comparison task `FD001 -> FD004` must wait until WSL
    CUDA access is restored

### 10.27 Fair-comparison rerun: CD-MambAtt `15-shot` on `FD001 -> FD004`

- date: 2026-04-02
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_bestcfg_fd001_to_fd004_15shot_5seeds/FD001_TO_FD004/summary.json`
- results:
  - mean direct RMSE = **34.3142**
  - mean adapted RMSE = **23.9469**
  - std = **2.9165**
  - best adapted RMSE = **22.3123** (best-by-validation seed selection)
- execution note:
  - the first background launch completed seeds `42,43,44` and left a partial
    `seed_45`
  - the missing `45,46` seeds were then resumed in a second invocation
  - the final `summary.json` was manually re-aggregated across all 5 seeds
    after the resumed run completed
- conclusion:
  - the full fair `15-shot` table is now complete on all four overlapping
    tasks
  - CD-MambAtt still beats the reproduced FOMLN baseline on this final task:
    - `25.0808 -> 23.9469`
    - gain ≈ **1.13 RMSE**

### 10.28 Final fair table: CD-MambAtt `15-shot` vs reproduced FOMLN `15-shot`

- date: 2026-04-02

| Task | CD-MambAtt 15-shot RMSE | FOMLN 15-shot RMSE | Δ (CD - FOMLN) |
|---|---:|---:|---:|
| `FD001 -> FD003` | **20.1384** | 25.4748 | -5.3364 |
| `FD003 -> FD001` | **19.5788** | 20.1261 | -0.5473 |
| `FD002 -> FD004` | **23.2879** | 26.0206 | -2.7327 |
| `FD001 -> FD004` | **23.9469** | 25.0808 | -1.1339 |

- conclusion:
  - we now have a completed **fair head-to-head package** on the four main
    overlapping transfer pairs
  - CD-MambAtt beats the reproduced FOMLN baseline on **4 / 4** tasks under
    the matched `15-shot` target-shot budget

## 11. SPD / DD-Mamba implementation smoke

### 11.1 CUDA forward validation for the new `dd_spd` block

- date: 2026-04-02
- purpose:
  - verify that the newly added `DDMambaBlock` can be instantiated inside
    `MambAttRegressor` and run on CUDA
- command summary:
  - instantiate `MambAttRegressor(mamba_block_mode='dd_spd')`
  - run `forward_features_with_aux` on a random CUDA tensor with shape
    `(2, 20, 21)`
- observed:
  - CUDA visible
  - prediction shape = `(2,)`
  - domain feature shape = `(2, 21)`
  - gate mean ≈ `0.1192`
- conclusion:
  - the SPD-style DD-Mamba block is wired into the main model correctly
  - auxiliary outputs needed for domain-adversarial training are available

### 11.2 `train_cd_mambatt_v3.py` 1-epoch smoke run

- date: 2026-04-02
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v3_smoke/FD001_TO_FD003/summary.json`
- protocol:
  - task = `FD001 -> FD003`
  - seeds = `42`
  - source epochs = `1`
  - target epochs = `1`
  - source max batches = `1`
  - target max batches = `1`
  - `mamba_block_mode = dd_spd`
  - `lambda_domain_adv = 0.1`
- key smoke observations:
  - source stage completed
  - adaptation stage completed
  - domain-adversarial loss produced finite values:
    - `train_domain_adv_loss = 0.7089`
    - `domain_accuracy = 0.3333`
  - gate statistics were emitted:
    - `gate_mean = 0.1192`
- result note:
  - the RMSE values are **not meaningful benchmark numbers** because this was
    only a 1-batch / 1-epoch smoke run
- conclusion:
  - the new SPD v0 training pipeline is now end-to-end runnable on CUDA
  - the next step should be a real multi-epoch canonical run on
    `FD001 -> FD003`

### 11.3 SPD branch fix: dead-gate gradient resolved

- date: 2026-04-02
- change:
  - `x_proj_spec` / `dt_proj_spec` initialization changed from exact zero to
    near-zero random init (`std = 1e-4`)
- reason:
  - with exact zero initialization, the gate branch receives zero gradient at
    the first step because the specific-path contribution is identically zero
- external verification summary:
  - DD vs original output difference:
    - from `0` to about `2e-7`
  - `gate_proj` gradient:
    - from `0.0` to non-zero (`~1e-6`)
- conclusion:
  - the SPD gate is no longer dead at initialization
  - the block still starts numerically extremely close to original Mamba

### 11.4 Canonical SPD pilot: `FD001 -> FD003`, seed `42`, short schedule

- date: 2026-04-02
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v3_fd001_to_fd003_seed42_pilot/FD001_TO_FD003/summary.json`
- protocol:
  - source epochs = `10`
  - target epochs = `5`
  - seed = `42`
  - `mamba_block_mode = dd_spd`
  - `lambda_domain_adv = 0.1`
- results:
  - source test RMSE = **16.3048**
  - direct target RMSE = **42.4490**
  - adapted target RMSE = **21.7000**
  - best adaptation epoch by validation = `3`
  - gate mean at best epoch ≈ **0.1750**
  - best-epoch domain accuracy ≈ **0.9983**
- interpretation:
  - SPD v0 can already reach the same RMSE scale as the current v2 method on a
    shortened schedule
  - but the very high domain-classification accuracy means the current
    invariant-path adversarial alignment is still weak

### 11.5 Canonical SPD full single-seed run: `FD001 -> FD003`, seed `42`

- date: 2026-04-02
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v3_fd001_to_fd003_seed42_full1/FD001_TO_FD003/summary.json`
- protocol:
  - source epochs = `50`
  - target epochs = `20`
  - seed = `42`
  - `mamba_block_mode = dd_spd`
  - `lambda_domain_adv = 0.1`
- results:
  - source test RMSE = **14.9012**
  - direct target RMSE = **41.2828**
  - adapted target RMSE = **21.8929**
  - best adaptation epoch by validation = `1`
  - best-epoch gate mean ≈ **0.1829**
  - best-epoch domain accuracy ≈ **0.9037**
- current interpretation:
  - SPD v0 is numerically stable under the full single-seed schedule
  - single-seed target-side RMSE is already in the **~21.9** range, close to
    the current v2 canonical scale
  - however, the domain classifier is still too successful; this suggests the
    present adversarial pressure is not yet making the invariant path truly
    domain-invariant
  - next work should focus on:
    - stronger / scheduled `lambda_domain_adv`
    - alternative domain-feature tap point
    - then multi-seed evaluation

### 11.6 SPD tuned single-seed run: `FD001 -> FD003`, seed `42`, stronger adversarial setup

- date: 2026-04-02
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v3_fd001_to_fd003_seed42_adv05_prelast_h16/FD001_TO_FD003/summary.json`
- protocol:
  - source epochs = `50`
  - target epochs = `20`
  - seed = `42`
  - `mamba_block_mode = dd_spd`
  - `lambda_domain_adv = 0.5`
  - `grl_lambda = 1.0`
  - `grl_warmup_epochs = 5`
  - `domain_feature_tap = pre_transformer_last`
  - `domain_adv_hidden_dim = 16`
- results:
  - source test RMSE = **14.9339**
  - direct target RMSE = **41.4872**
  - adapted target RMSE = **21.2435**
  - best adaptation epoch by validation = `1`
  - best-epoch domain accuracy ≈ **0.7351**
  - best-epoch gate mean ≈ **0.1834**
- interpretation:
  - compared with the earlier SPD full run (`21.8929`), the stronger
    adversarial setup clearly improved the single-seed target RMSE
  - the domain classifier became less dominant at the best epoch
    (`0.9037 -> 0.7351`), so the change is not a no-op
  - however, best epoch still occurs immediately at adaptation epoch `1`,
    meaning later epochs still tend to overfit or lose invariance

### 11.7 SPD 3-seed validation: current best tuned setup is **not** stable yet

- date: 2026-04-02
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v3_fd001_to_fd003_3seeds_adv05_prelast_h16/FD001_TO_FD003/summary.json`
- protocol:
  - task = `FD001 -> FD003`
  - seeds = `42, 43, 44`
  - source epochs = `50`
  - target epochs = `20`
  - `mamba_block_mode = dd_spd`
  - `lambda_domain_adv = 0.5`
  - `grl_lambda = 1.0`
  - `grl_warmup_epochs = 5`
  - `domain_feature_tap = pre_transformer_last`
  - `domain_adv_hidden_dim = 16`
- per-seed results:
  - seed `42`:
    - direct RMSE = **41.6009**
    - adapted RMSE = **21.1727**
    - best epoch = `1`
    - best-epoch domain accuracy = **0.7366**
  - seed `43`:
    - direct RMSE = **43.3332**
    - adapted RMSE = **26.9824**
    - best epoch = `16`
    - best-epoch domain accuracy = **0.9982**
  - seed `44`:
    - direct RMSE = **33.0671**
    - adapted RMSE = **23.9256**
    - best epoch = `18`
    - best-epoch domain accuracy = **0.9992**
- summary:
  - mean direct RMSE = **39.3337 ± 4.4873**
  - mean adapted RMSE = **24.0269 ± 2.3729**
  - mean best-epoch domain accuracy = **0.9114**
- comparison against prior references on the same canonical task:
  - `CD-MambAtt v2` matched `3-seed` reference:
    - mean RMSE = **21.1291**
  - `CD-MambAtt v2` matched `5-seed` reference:
    - mean RMSE = **22.0342**
- interpretation:
  - the current SPD v0 setup does **not** beat the existing v2 baseline under
    multi-seed evaluation
  - the main failure mode is cross-seed instability: once the selected epoch
    drifts to later adaptation stages, domain accuracy collapses back toward
    `1.0` and test RMSE worsens sharply
  - this strongly suggests the present domain-adversarial signal is still too
    weak or too easy to bypass

### 11.8 SPD follow-up: `concat_inv_pre` feature tap

- date: 2026-04-02
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v3_fd001_to_fd003_seed42_adv05_concat_h16/FD001_TO_FD003/summary.json`
- protocol:
  - seed = `42`
  - same as Section 11.6 except:
    - `domain_feature_tap = concat_inv_pre`
- results:
  - source test RMSE = **14.9303**
  - direct target RMSE = **41.6914**
  - adapted target RMSE = **21.1674**
  - best adaptation epoch = `1`
  - best-epoch domain accuracy = **0.8759**
  - best-epoch gate mean ≈ **0.1826**
- interpretation:
  - concatenating invariant-path and pre-transformer features produced only a
    tiny single-seed RMSE gain over `pre_transformer_last`
    (`21.2435 -> 21.1674`)
  - but the domain classifier became **easier**, not harder, to optimize
    (`0.7351 -> 0.8759`)
  - therefore `concat_inv_pre` is not currently a compelling direction

### 11.9 SPD follow-up on failed seed: stronger adversarial weight did not rescue seed `43`

- date: 2026-04-02
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v3_fd001_to_fd003_seed43_adv10_prelast_h16/FD001_TO_FD003/summary.json`
- protocol:
  - seed = `43`
  - same as Section 11.6 except:
    - `lambda_domain_adv = 1.0`
- results:
  - source test RMSE = **17.9920**
  - direct target RMSE = **43.3331**
  - adapted target RMSE = **27.1231**
  - best adaptation epoch = `16`
  - best-epoch domain accuracy = **0.9979**
  - best-epoch gate mean ≈ **0.1302**
- interpretation:
  - increasing `lambda_domain_adv` from `0.5` to `1.0` did **not** fix the bad
    seed; performance slightly worsened (`26.9824 -> 27.1231`)
  - the early adaptation epochs briefly showed healthier domain accuracy
    (epoch `1` around `0.61`), but the selected checkpoint still moved to a late
    epoch where the discriminator again became nearly perfect
  - current evidence indicates the issue is not solved by simply increasing the
    adversarial coefficient

### 11.10 Current SPD v0 conclusion after follow-up runs

- date: 2026-04-02
- status:
  - **implementation status**: working on CUDA, reproducible, and not a no-op
  - **single-seed best**:
    - `FD001 -> FD003`, seed `42`
    - RMSE = **21.1674** (`concat_inv_pre`) or **21.2435**
      (`pre_transformer_last`)
  - **current multi-seed status**:
    - mean RMSE = **24.0269** on seeds `42,43,44`
- current judgment:
  - SPD v0 is scientifically interesting because it can outperform bare v3 on
    a favorable seed and its gate branch is active
  - however, the present domain-adversarial realization is still not robust
    enough to replace the existing `CD-MambAtt v2` baseline
  - the immediate next step should **not** be larger seed sweeps of the same
    configuration; instead it should target the instability mechanism itself,
    e.g.:
    - checkpoint selection that penalizes high domain accuracy
    - stronger isolation of invariant vs specific channels
    - stage-wise or partially frozen adaptation to prevent late-epoch drift

### 11.11 Code update: SPD invariant alignment now supports `GRL` or direct `MMD`

- date: 2026-04-03
- file:
  - `/home/shelterpl/cd_mambatt/train_cd_mambatt_v3.py`
- change:
  - added `--inv-alignment-mode {grl,mmd,none}`
  - added `--lambda-inv-mmd`
  - preserved the previous `GRL`-based domain discriminator path as an ablation
  - added direct invariant-path MMD alignment as a new option
- motivation:
  - the earlier SPD v0 runs showed that GRL often became unstable on
    `FD001 -> FD003`, especially once training selected late adaptation epochs
    with domain accuracy near `1.0`
  - the new variant directly aligns the SPD invariant path with MMD instead of
    using a min-max discriminator game

### 11.12 SPD follow-up: replace GRL with invariant-path MMD (`inv_mean`)

- date: 2026-04-03
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v3_fd001_to_fd003_3seeds_invmmd01_invmean/FD001_TO_FD003/summary.json`
- protocol:
  - task = `FD001 -> FD003`
  - seeds = `42, 43, 44`
  - source epochs = `50`
  - target epochs = `20`
  - `mamba_block_mode = dd_spd`
  - `inv_alignment_mode = mmd`
  - `lambda_inv_mmd = 0.1`
  - `lambda_domain_adv = 0.0`
  - `domain_feature_tap = inv_mean`
- per-seed results:
  - seed `42`:
    - direct RMSE = **41.4420**
    - adapted RMSE = **20.8094**
    - best epoch = `1`
  - seed `43`:
    - direct RMSE = **43.3332**
    - adapted RMSE = **26.4091**
    - best epoch = `17`
  - seed `44`:
    - direct RMSE = **33.0312**
    - adapted RMSE = **22.8309**
    - best epoch = `13`
- summary:
  - mean direct RMSE = **39.2688 ± 4.4777**
  - mean adapted RMSE = **23.3498 ± 2.3153**
- comparison against prior SPD-GRL run on the same three seeds:
  - SPD + GRL:
    - mean RMSE = **24.0269**
    - std = **2.3729**
  - SPD + invariant-path MMD:
    - mean RMSE = **23.3498**
    - std = **2.3153**
- comparison against current `CD-MambAtt v2` three-seed reference:
  - mean RMSE = **21.1291**
- interpretation:
  - replacing GRL with direct invariant-path MMD is a **real improvement**
    over the previous SPD v0 adversarial branch
  - the gains are consistent across all three seeds:
    - seed `42`: `21.1727 -> 20.8094`
    - seed `43`: `26.9824 -> 26.4091`
    - seed `44`: `23.9256 -> 22.8309`
  - however, the current SPD branch still does **not** beat the existing
    `CD-MambAtt v2` reference under matched three-seed evaluation
  - one likely reason is that the base scaffold still applies the original
    global MMD on the combined features, so the spec path may not yet be fully
    acting as a clean domain-specific escape channel

### 11.13 SPD follow-up: scanning the original global MMD weight after adding inv-MMD

- date: 2026-04-03
- motivation:
  - after Section 11.12, the main question was whether the original global MMD
    on the combined features was interfering with the intended SPD
    inv/spec separation
- pilot setting:
  - task = `FD001 -> FD003`
  - seeds = `42, 43`
  - `inv_alignment_mode = mmd`
  - `lambda_inv_mmd = 0.1`
  - `domain_feature_tap = inv_mean`
- runs:
  - `lambda_mmd = 0.0`
    - output:
      - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v3_fd001_to_fd003_2seeds_invmmd01_globalmmd0_invmean/FD001_TO_FD003/summary.json`
    - results:
      - seed `42`: **20.8490**
      - seed `43`: **26.4351**
      - mean: **23.6421**
  - `lambda_mmd = 0.05`
    - output:
      - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v3_fd001_to_fd003_2seeds_invmmd01_globalmmd005_invmean/FD001_TO_FD003/summary.json`
    - results:
      - seed `42`: **22.5427**
      - seed `43`: **26.3376**
      - mean: **24.4401**
  - reference (`lambda_mmd = 0.1`)
    - seed `42`: **20.8094**
    - seed `43`: **26.4091**
    - mean: **23.6093**
- interpretation:
  - removing global MMD did **not** improve the two-seed pilot
  - weakening global MMD to `0.05` was clearly worse because seed `42`
    collapsed to a late-epoch checkpoint
  - therefore, among the tested values, the best current choice remains:
    - `lambda_mmd = 0.1`
    - `lambda_inv_mmd = 0.1`

### 11.14 SPD follow-up: stronger invariant-path MMD weight (`lambda_inv_mmd = 0.2`)

- date: 2026-04-03
- motivation:
  - since direct inv-MMD was stable, we next tested whether a stronger
    invariant-path MMD coefficient could further improve the bad seeds
- 2-seed pilot:
  - output:
    - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v3_fd001_to_fd003_2seeds_invmmd02_globalmmd01_invmean/FD001_TO_FD003/summary.json`
  - protocol:
    - seeds = `42, 43`
    - `lambda_mmd = 0.1`
    - `lambda_inv_mmd = 0.2`
  - results:
    - seed `42`: **20.7451**
    - seed `43`: **26.3556**
    - mean: **23.5504**
  - interpretation:
    - the two pilot seeds improved slightly over `lambda_inv_mmd = 0.1`, so
      a full `3-seed` rerun was justified

### 11.15 SPD follow-up: full 3-seed check for `lambda_inv_mmd = 0.2`

- date: 2026-04-03
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v3_fd001_to_fd003_3seeds_invmmd02_globalmmd01_invmean/FD001_TO_FD003/summary.json`
- protocol:
  - task = `FD001 -> FD003`
  - seeds = `42, 43, 44`
  - `inv_alignment_mode = mmd`
  - `lambda_mmd = 0.1`
  - `lambda_inv_mmd = 0.2`
  - `domain_feature_tap = inv_mean`
- per-seed results:
  - seed `42`: **20.8015**
  - seed `43`: **26.4627**
  - seed `44`: **22.8536**
- summary:
  - mean RMSE = **23.3726 ± 2.3401**
- comparison against the earlier `lambda_inv_mmd = 0.1` full run:
  - `lambda_inv_mmd = 0.1`:
    - mean RMSE = **23.3498 ± 2.3153**
  - `lambda_inv_mmd = 0.2`:
    - mean RMSE = **23.3726 ± 2.3401**
- interpretation:
  - stronger inv-MMD produced a tiny gain on seed `42`, but slight regressions
    on seeds `43` and `44`
  - under full `3-seed` evaluation, `lambda_inv_mmd = 0.2` is **not** better
    than `0.1`
  - the current best SPD configuration therefore remains:
    - `inv_alignment_mode = mmd`
    - `lambda_inv_mmd = 0.1`
    - `lambda_mmd = 0.1`

### 11.16 Current checkpoint after the latest SPD ablations

- date: 2026-04-03
- current best SPD setting:
  - output:
    - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v3_fd001_to_fd003_3seeds_invmmd01_invmean/FD001_TO_FD003/summary.json`
  - setting:
    - `inv_alignment_mode = mmd`
    - `lambda_inv_mmd = 0.1`
    - `lambda_mmd = 0.1`
    - `domain_feature_tap = inv_mean`
  - result:
    - mean RMSE = **23.3498 ± 2.3153**
- current conclusion:
  - switching from GRL to direct invariant-path MMD was the correct move
  - but neither weakening the old global MMD nor strengthening inv-MMD was
    enough to close the remaining gap to `CD-MambAtt v2`
  - the next SPD iteration should focus less on scalar loss weights and more
    on **architectural separation / optimization protocol**, e.g.:
    - selectively freezing parts of the shared backbone during adaptation
    - letting only the spec/gate-related parameters absorb domain-specific shift
    - or explicitly reducing the supervision route through the invariant path

### 11.17 SPD selective-freezing pilot: `spec_gate_head`

- date: 2026-04-03
- output:
  - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v3_fd001_to_fd003_2seeds_invmmd01_freeze_specgatehead/FD001_TO_FD003/summary.json`
- protocol:
  - task = `FD001 -> FD003`
  - seeds = `42, 43`
  - `mamba_block_mode = dd_spd`
  - `inv_alignment_mode = mmd`
  - `lambda_inv_mmd = 0.1`
  - `lambda_mmd = 0.1`
  - `adaptation_freeze_mode = spec_gate_head`
- summary:
  - mean RMSE = **27.0818 ± 1.7057**
- interpretation:
  - this freeze mode is too restrictive
  - only letting the spec path, gate, and regression head adapt causes a strong
    accuracy drop relative to the non-freeze inv-MMD reference
  - the gate often grows too aggressively while the frozen shared backbone
    cannot compensate

### 11.18 SPD selective-freezing follow-up: `spec_gate_transformer_head`

- date: 2026-04-03
- code update:
  - `train_cd_mambatt_v3.py` now supports:
    - `--adaptation-freeze-mode spec_gate_transformer_head`
  - trainable parameters in this mode:
    - `x_proj_spec`
    - `dt_proj_spec`
    - `gate_proj`
    - all `transformer_blocks.*`
    - `head.*`
  - initial trainable parameter count:
    - **18,272 / 23,564** (`77.54%`)
- 2-seed pilot:
  - output:
    - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v3_fd001_to_fd003_2seeds_invmmd01_freeze_specgatetransformerhead/FD001_TO_FD003/summary.json`
  - per-seed RMSE:
    - seed `42`: **25.4497**
    - seed `43`: **20.5555**
  - mean RMSE:
    - **23.0026 ± 2.4471**
  - comparison against the matched non-freeze `42,43` reference:
    - non-freeze inv-MMD (`42,43`): **23.6093**
    - frozen pilot (`42,43`): **23.0026**
- 3-seed full check:
  - output:
    - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v3_fd001_to_fd003_3seeds_invmmd01_freeze_specgatetransformerhead/FD001_TO_FD003/summary.json`
  - per-seed RMSE:
    - seed `42`: **26.2902**
    - seed `43`: **23.3387**
    - seed `44`: **22.7875**
  - mean RMSE:
    - **24.1388 ± 1.5378**
- interpretation:
  - the first `2-seed` pilot looked mildly promising, but the full `3-seed`
    rerun removed that gain
  - compared with the current non-freeze inv-MMD `3-seed` reference
    (**23.3498 ± 2.3153**), the full frozen result is worse
  - freezing the shared Mamba / invariant path does reduce variance, but it
    also hurts the mean performance
  - this suggests the model still needs some shared-path adaptation capacity

### 11.19 SPD freeze-then-unfreeze pilot: warm-start with `spec_gate_transformer_head`

- date: 2026-04-03
- code update:
  - `train_cd_mambatt_v3.py` now supports:
    - `--adaptation-freeze-epochs`
  - current behavior:
    - keep the requested freeze mode for the first `N` adaptation epochs
    - then switch to full-model adaptation (`freeze_mode = none`)
    - an explicit JSON log line is emitted at the switch epoch
- pilot run:
  - output:
    - `/home/shelterpl/cd_mambatt/runs/cd_mambatt_v3_fd001_to_fd003_2seeds_invmmd01_freeze5_then_unfreeze_specgatetransformerhead/FD001_TO_FD003/summary.json`
  - protocol:
    - seeds = `42, 43`
    - `adaptation_freeze_mode = spec_gate_transformer_head`
    - `adaptation_freeze_epochs = 5`
    - epoch `6` switches to full unfreeze
  - per-seed RMSE:
    - seed `42`: **25.8292**
    - seed `43`: **21.4375**
  - mean RMSE:
    - **23.6333 ± 2.1958**
- interpretation:
  - staged unfreezing partially recovers the severe degradation of the static
    freeze, but it still does **not** beat the non-freeze inv-MMD baseline
  - compared with the matched non-freeze `42,43` reference (**23.6093**),
    the freeze-then-unfreeze pilot is essentially tied but slightly worse
  - current conclusion for this branch:
    - simple selective freezing is **not enough** to improve SPD
    - the next SPD iteration should move toward stronger structural separation
      or more informative auxiliary objectives, rather than another small
      freeze-pattern sweep
