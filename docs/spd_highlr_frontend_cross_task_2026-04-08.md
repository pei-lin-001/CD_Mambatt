# SPD 前端对齐高学习率配方：跨任务验证（2026-04-08）

沿用在 `FD001 -> FD003` 上打出的新 best 配方：

- SPD (`dd_spd`)
- global feature MMD
- source stage CE
- spec-domain-predictive
- cosine scheduler
- target LR = `1.5e-3`

并比较两种 invariant 对齐位置：

- `frontend`：对齐 `frontend_features`
- `inv`：对齐原来的 `domain_features / inv_mean`

结果快照：

- `docs/generated/spd_highlr_frontend_cross_task_2026-04-08.json`

---

## 1. FD001 -> FD004

### frontend 对齐

- seed42: `23.6176`
- seed43: `25.3548`
- seed44: `21.4423`
- **mean ± std = `23.4716 ± 1.6006`**

### inv 对齐

- seed42: `23.6342`
- seed43: `25.7419`
- seed44: `21.4503`
- **mean ± std = `23.6088 ± 1.7521`**

### 结论

- `frontend` 比 `inv` **略好**
- 也比之前内部记录的 SPD best（约 `23.72 ± 2.38`）**略好**
- 说明这条新配方对 `FD001 -> FD004` **有一定可迁移性**

---

## 2. FD003 -> FD001

### frontend 对齐

- seed42: `20.4722`
- seed43: `19.4026`
- seed44: `23.5308`
- **mean ± std = `21.1352 ± 1.7493`**

### inv 对齐

- seed42: `21.0633`
- seed43: `19.1718`
- seed44: `23.5536`
- **mean ± std = `21.2629 ± 1.7945`**

### 结论

- `frontend` 仍比 `inv` **略好**
- 但仍然**没有打穿**这个任务
- 相比之前内部最好 SPD（约 `21.03 ± 1.70`）没有实质提升
- 相比 v2 参考（约 `19.81`）仍然落后

---

## 3. 综合判断

这条新主线的跨任务结论很清楚：

1. **`FD001 -> FD003`：明显成功**
   - 新 best：`20.0151 ± 0.7523`

2. **`FD001 -> FD004`：小幅成功**
   - 比旧内部 SPD best 略好
   - frontend tap 比 inv tap 更优

3. **`FD003 -> FD001`：仍未解决**
   - frontend tap 只是轻微优于 inv tap
   - 但没有突破历史最好结果

所以目前最合理的判断是：

> **frontend alignment + tuned cosine LR 是一条真实有效的主线，但它不是“所有任务统一奏效”的万能解。**

尤其：

> `FD003 -> FD001` 仍然是下一步必须单独攻克的难点任务。
