# SPD 前端对齐高学习率实验（2026-04-08）

这轮夜间实验最终打通的是一条**更简单**、但更有效的路线：

- 保留 `SPD(dd_spd)` 主体
- 保留 `spec-domain-predictive` 辅助监督
- 保留 `cosine` 调度
- **把 invariant 对齐位置前移到 `frontend_features`**
- 同时把 adaptation learning rate 从旧 best 的 `2e-3` 下调到 **`1.5e-3`**

对应脚本：

- `/home/shelterpl/cd_mambatt/experiments/ablations/spd_highlr_frontend_3seed.py`

结果快照：

- `/home/shelterpl/cd_mambatt/docs/generated/spd_highlr_frontend_lr15_summary_2026-04-08.json`
- 日志：`/home/shelterpl/cd_mambatt/runs/logs/spd_highlr_frontend_3seed_lr15_20260408.log`

---

## 配置

任务：

- `FD001 -> FD003`

协议：

- 5-shot
- target val units = 10
- source split seed = 42
- `resample_few_shot_per_seed = True`
- `source_val_all_windows = True`
- `target_val_all_windows = True`

损失：

- source MSE = `1.0`
- target MSE = `1.0`
- global MMD = `0.1`
- source stage CE = `1.0`
- **frontend invariant MMD = `0.1`**
- **spec-domain-predictive = `0.1`**

优化：

- Adam
- adaptation LR = **`1.5e-3`**
- scheduler = `CosineAnnealingLR(T_max=20, eta_min=1e-5)`

---

## 结果

| Seed | Direct RMSE | Adapted RMSE | Best Epoch |
|---|---:|---:|---:|
| 42 | 42.6474 | 20.9774 | 3 |
| 43 | 51.2006 | 19.9270 | 1 |
| 44 | 35.9336 | 19.1410 | 10 |
| **Mean ± Std** |  | **20.0151 ± 0.7523** |  |

---

## 对比

相对之前文档里记录的旧 best：

- 旧 best：`20.37`
- 新结果：**`20.0151`**
- 改善：**`-0.355 RMSE`**

这说明：

1. **“更早的前端对齐”在简单 recipe 里是有效的**。
2. `2e-3` 对这条路线偏猛，`1.5e-3` 更稳。
3. 当前更好的故事不是继续堆复杂 conditional loss，而是：
   - 前端早对齐
   - 让 spec 分支明确承担 domain 信息
   - 用更合适的 adaptation 优化强度

---

## 当前判断

对 `FD001 -> FD003` 来说，当前最值得继续沿着走的主线是：

> **simple SPD recipe + frontend alignment + tuned cosine LR**

而不是继续在 `conditional alignment` 上做小修小补。
