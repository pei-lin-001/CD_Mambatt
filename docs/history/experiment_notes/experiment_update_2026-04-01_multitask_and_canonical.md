# CD-MambAtt 实验更新报告（2026-04-01）

## 1. 本次新增完成的实验

今天新增完成了 4 组关键实验：

1. `FD004 -> FD002`，`5 seeds`
2. `FD001 -> FD002`，`5 seeds`
3. `FD001 -> FD003` 正式 `5 seeds` 复跑（`lambda_monotonic = 0.05`）
4. `FD001 -> FD003` 正式 `5 seeds` 稳定性复核（`lambda_monotonic = 0.1`）

所有实验均基于当前工作版 CD-MambAtt：

- MMD
- confidence-filtered pseudo-labeling
- local monotonicity
- no contrastive term

---

## 2. 多任务新增结果

| Task | Direct RMSE | CD RMSE | Gain | Gain % | Output |
|---|---:|---:|---:|---:|---|
| `FD004 -> FD002` | 21.3666 | **19.5311** | 1.8354 | 8.6% | `runs/cd_mambatt_bestcfg_fd004_to_fd002_5seeds_fixcond` |
| `FD001 -> FD002` | 31.1029 | **22.9457** | 8.1572 | 26.2% | `runs/cd_mambatt_bestcfg_fd001_to_fd002_5seeds` |

### 解释

- `FD004 -> FD002`：
  - 提升幅度较小，但仍是**稳定正增益**
  - 原因很可能是 direct baseline 本身已经较强，改进空间更小

- `FD001 -> FD002`：
  - 是非常关键的 **single-condition → multi-condition** 结果
  - 增益较大，说明当前方法并不只适用于 stage-1 任务

---

## 3. Canonical 任务正式 `5 seeds` 复核

### 3.1 主配置复跑：`lambda_monotonic = 0.05`

- task: `FD001 -> FD003`
- direct mean RMSE = **34.7179**
- CD mean RMSE = **22.0342**
- std = **1.4408**
- output:
  - `runs/cd_mambatt_bestcfg_fd001_to_fd003_5seeds`

### 3.2 稳定性复核：`lambda_monotonic = 0.1`

- task: `FD001 -> FD003`
- direct mean RMSE = **34.8204**
- CD mean RMSE = **21.9608**
- std = **1.4574**
- output:
  - `runs/cd_mambatt_formal_fd001_to_fd003_monotonic01_5seeds`

### 3.3 解释

这一步带来两个重要信息：

1. 先前常引用的 `21.1291` 来自 `3 seeds`，现在看偏乐观
2. 在正式 `5 seeds` 下，`monotonic = 0.05` 与 `0.1` **几乎打平**

所以：

> 当前不能再把 `0.05` 写成“已经被完全证明的全局最优”；更严谨的说法是：`0.05` 是当前多任务主配置，而 `0.1` 是 canonical 正式协议下略优的近邻候选。

---

## 4. 当前整体判断

截至本次更新，可以较稳妥地下结论：

### 已成立

- 方法在多个 C-MAPSS 迁移任务上都优于 direct transfer
- 结果已经覆盖：
  - reverse single-condition
  - multi-condition
  - reverse multi-condition
  - single-condition → multi-condition
  - harder transfer

### 仍需谨慎

- canonical 任务上的“最佳 monotonic 权重”还没有完全拉开
- 因此，后续论文写作里应避免把某个单一权重说得过于绝对

---

## 5. 建议的下一步

当前更值得做的，不再是继续无边界扫参，而是：

1. 补 published baselines
2. 准备第二数据集 `XJTU-SY`
3. 如需进一步完善对称性，再补 `FD002 -> FD001`
4. 若临近定稿，再决定是否在更多任务上比较 monotonic `0.05` vs `0.1`

---

## 6. 一句话总结

> 今天的结果进一步证明：CD-MambAtt 的跨域思路是成立的，而且已经有多任务证据支撑；但在更严格的 `5 seeds` canonical 协议下，monotonic 权重的最佳点仍需要用更谨慎的口径来表述。
