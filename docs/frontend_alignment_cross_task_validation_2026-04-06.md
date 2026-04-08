# Front-end alignment cross-task validation (2026-04-06)

这轮实验的目的不是继续猜结构，而是检验 **`frontend_mean` 这个“更早对齐”结论是否具有跨任务普适性**。

## 本轮做了什么

- 为 `scripts/run_targeted_adaptation_experiment.py` 增加了 **manual source-checkpoint 模式**，可以直接用当前 SPD checkpoint + 已有 split 复跑 adaptation。
- 对当前可用任务做了 3-seed 对照：
  - `exp_inv_mean_full`: 当前 SPD 基线（`domain_feature_tap=inv_mean`）
  - `exp_frontend_mean_full`: 早期前端对齐（`domain_feature_tap=frontend_mean`）
- `FD001 -> FD003` 继续沿用此前已经完成的 v3 SPD fullmatch / frontend 对照。

## 结果汇总

### FD001_TO_FD003

- 基线（已有 v3 SPD run）: **21.0859 ± 0.9814**
- `frontend_mean`: **20.3882 ± 0.7798**
- `frontend_mean - old_reference`: **-0.6978 RMSE**
- 旧存档参考: **21.0859 ± 0.9814**

### FD003_TO_FD001

- 当前 `inv_mean`: **19.9104 ± 1.0008**
- `frontend_mean`: **19.9144 ± 0.9976**
- `frontend_mean - exp_inv_mean_full`: **+0.0040 RMSE**
- 旧存档参考: **20.3806 ± 1.3164**

### FD001_TO_FD004

- 当前 `inv_mean`: **23.5773 ± 2.6271**
- `frontend_mean`: **23.6285 ± 2.5528**
- `frontend_mean - exp_inv_mean_full`: **+0.0512 RMSE**
- 旧存档参考: **23.9372 ± 2.8138**

## 结论

1. **`frontend_mean` 不是普适增益。**
   - `FD001 -> FD003`: 明显有效，`21.0859 -> 20.3882`。
   - `FD003 -> FD001`: 基本无变化，`19.9104 -> 19.9144`。
   - `FD001 -> FD004`: 基本无变化，`23.5773 -> 23.6285`。

2. **因此“把 invariant 对齐前移到前端”目前只能算 task-conditional trick，不是统一机制答案。**
   它说明 `FD001 -> FD003` 的主导域差异更早进入共享前端；但在更难迁移里，瓶颈不只是前端均值漂移。

3. **当前 SPD 基线本身是健康的。**
   本轮用当前 SPD source checkpoint 复跑后，`FD003 -> FD001` 与 `FD001 -> FD004` 的 `inv_mean` 结果都略优于旧的 v2 存档参考。

## 下一步建议

- 不要把 `frontend_mean` 直接升级成默认设置。
- 后续更值得做的是：
  1. 先区分 **前端主导漂移** vs **时序/工况主导漂移**；
  2. 对难任务优先检查 **conditional alignment、pseudo-label reliability、multi-condition normalization**。
