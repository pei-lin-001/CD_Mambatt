# CD-MambAtt vs FOMLN Formal Comparison (2026-04-02)
## Purpose
This note compares the current **formal FOMLN reproduction baseline** against the current best **CD-MambAtt cross-domain results** on the overlapping C-MAPSS transfer pairs.
## Important fairness note
- Current **CD-MambAtt** results use **5-shot** target supervision.
- Current **FOMLN reproduction baseline** uses **15-shot** target supervision, following the FOMLN paper setting.
- Therefore this is **not yet a perfectly matched head-to-head protocol**.
- However, it is still highly informative because if CD-MambAtt remains better even under a stricter `5-shot` setting, that is a strong signal.
## Protocol snapshot
### CD-MambAtt current best config
- target shots: `5`
- seeds: `42,43,44,45,46`
- core setting: `lambda_mmd=0.1`, `lambda_pseudo=0.5`, `lambda_monotonic=0.05`
### FOMLN formal baseline current stable config
- target shots: `15`
- seeds: `42,43,44,45,46`
- source-task-level: `units`
- shot-level: `units`
- source inner optimizer: `Adam`
- target adaptation optimizer: `SGD`
- inner/adapt batch size: `256 / 256`
## Mean RMSE comparison
| Task | CD-MambAtt shots | CD-MambAtt mean RMSE | FOMLN shots | FOMLN mean RMSE | Δ (CD - FOMLN) | Published FOMLN paper | Reproduced FOMLN gap vs paper |
|---|---:|---:|---:|---:|---:|---:|---:|
| `FD001 → FD003` | 5 | **22.0342** | 15 | 25.4748 | -3.4406 | 14.34 | 11.1348 |
| `FD003 → FD001` | 5 | **19.8137** | 15 | 20.1261 | -0.3124 | 13.23 | 6.8961 |
| `FD002 → FD004` | 5 | **22.1776** | 15 | 26.0206 | -3.8430 | 18.55 | 7.4706 |
| `FD001 → FD004` | 5 | **24.3449** | 15 | 25.0808 | -0.7359 | 18.16 | 6.9208 |

Interpretation of `Δ (CD - FOMLN)`: negative means CD-MambAtt is better.
## Direct-transfer improvement comparison
| Task | CD direct RMSE | CD adapted RMSE | CD gain | FOMLN direct RMSE | FOMLN adapted RMSE | FOMLN gain | Gain gap (CD - FOMLN) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `FD001 → FD003` | 34.7179 | 22.0342 | 12.6837 | 34.0750 | 25.4748 | 8.6002 | 4.0834 |
| `FD003 → FD001` | 24.6059 | 19.8137 | 4.7922 | 25.0772 | 20.1261 | 4.9511 | -0.1589 |
| `FD002 → FD004` | 29.5028 | 22.1776 | 7.3252 | 31.1871 | 26.0206 | 5.1665 | 2.1587 |
| `FD001 → FD004` | 32.3820 | 24.3449 | 8.0371 | 30.9944 | 25.0808 | 5.9136 | 2.1235 |

## Key observations
1. **CD-MambAtt currently beats the reproduced FOMLN mean RMSE on all 4 overlapping tasks**: `True`.
2. This is encouraging because CD-MambAtt is currently using the stricter **5-shot** setting, while FOMLN uses **15-shot**.
3. The biggest mean-RMSE margins in favor of CD-MambAtt are on:
   - `FD002 → FD004`: -3.8430
   - `FD001 → FD003`: -3.4406
   - `FD001 → FD004`: -0.7359
   - `FD003 → FD001`: -0.3124
4. The reproduced FOMLN baseline is still noticeably worse than the **published FOMLN paper numbers**, which means the current FOMLN line should be treated as a **formal reproduced baseline**, not as a claim that we have perfectly matched the paper.
5. Among the four tasks, the reproduced FOMLN is closest to CD-MambAtt on:
   - `FD003 → FD001` with mean-RMSE gap -0.3124.
## Current scientific interpretation
- If the goal is a **reproducible in-house baseline**, we now have it.
- If the goal is a **strictly fair published comparison**, the next missing piece is to rerun CD-MambAtt under the same `15-shot` protocol.
- That experiment would answer the strongest reviewer question: **does CD-MambAtt still beat FOMLN when both methods are given the same target-shot budget?**
## Recommended next action
1. Run **CD-MambAtt (best config)** on the same four tasks with **`target_shots = 15`** and `5 seeds`.
2. Build the first fair comparison table: `CD-MambAtt 15-shot vs FOMLN 15-shot`.
3. Keep the current `5-shot` CD-MambAtt table as the stronger data-efficiency result.

## Update: first fair `15-shot` head-to-head results now available

After the first version of this note, I started rerunning CD-MambAtt under the
same `15-shot` target-shot budget used by FOMLN.

Completed fair-comparison tasks so far:

- `/home/shelterpl/cd_mambatt/runs/cd_mambatt_bestcfg_fd001_to_fd003_15shot_5seeds/FD001_TO_FD003/summary.json`
- `/home/shelterpl/cd_mambatt/runs/cd_mambatt_bestcfg_fd003_to_fd001_15shot_5seeds/FD003_TO_FD001/summary.json`
- `/home/shelterpl/cd_mambatt/runs/cd_mambatt_bestcfg_fd002_to_fd004_15shot_5seeds/FD002_TO_FD004/summary.json`

### Fair comparison table (`15-shot` vs `15-shot`)

| Task | CD-MambAtt 15-shot mean RMSE | FOMLN 15-shot mean RMSE | Δ (CD - FOMLN) |
|---|---:|---:|---:|
| `FD001 → FD003` | **20.1384** | 25.4748 | -5.3364 |
| `FD003 → FD001` | **19.5788** | 20.1261 | -0.5474 |
| `FD002 → FD004` | **23.2879** | 26.0206 | -2.7327 |
| `FD001 → FD004` | **23.9469** | 25.0808 | -1.1339 |

Interpretation:

- once CD-MambAtt is also given `15-shot` target supervision, it still beats
  the reproduced FOMLN baseline on all four overlapping tasks
- the largest margin is still on `FD001 → FD003`
- `FD002 → FD004` remains a meaningful win even on the harder multi-condition
  transfer pair
- `FD001 → FD004` is also now completed and remains a positive fair
  head-to-head win, although with a smaller margin than the other two
  multi-condition directions

### Final completed row

- `/home/shelterpl/cd_mambatt/runs/cd_mambatt_bestcfg_fd001_to_fd004_15shot_5seeds/FD001_TO_FD004/summary.json`

Key numbers:

- mean direct RMSE = **34.3142**
- mean adapted RMSE = **23.9469**
- std = **2.9165**
- best adapted RMSE (best-by-validation seed) = **22.3123**
- note:
  - this summary was aggregated from an initial `42-44` run plus a resumed
    `45-46` run, because the first background launch stopped after partial
    completion
  - the final `summary.json` in that directory is the corrected full 5-seed
    aggregation

### Final fair head-to-head conclusion

Across all four overlapping transfer pairs, the fair `15-shot vs 15-shot`
comparison now says:

1. **CD-MambAtt beats the reproduced FOMLN baseline on 4/4 tasks**
2. the largest gain is on `FD001 → FD003` (**-5.3364 RMSE**)
3. the smallest but still positive gain is on `FD003 → FD001`
   (**-0.5474 RMSE**)
4. the multi-condition tasks are also positive overall:
   - `FD002 → FD004`: **-2.7327**
   - `FD001 → FD004`: **-1.1339**

This establishes the first completed fair comparison package for the project:
**CD-MambAtt 15-shot vs reproduced FOMLN 15-shot** on all four overlapping
C-MAPSS transfer pairs.
