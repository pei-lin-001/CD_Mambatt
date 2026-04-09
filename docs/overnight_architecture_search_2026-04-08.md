# 夜间架构探索记录（2026-04-08）

## 本轮原则

按照最新约束：

- **先诊断失败原因，再做下一步修改**
- 先避免继续陷入纯超参数调优
- 优先尝试**架构层面的改动**
- 每条新路都尽量配套“它是否真的被模型用上”的诊断

---

## 为什么没有继续沿着 DCG 往下调

对 `domain-conditioned gate (DCG)` 的补跑与诊断已经说明：

- `domain_gate_shift` 学到的量很小；
- 强制切换 `domain_label=0/1` 后，gate 和预测变化都很弱；
- target test RMSE 基本不变。

所以当前证据不支持：

> “只要显式告诉 gate 当前是 source/target，模型就会真正用起 spec 路径”

更准确的结论是：

> **标量 gate bias 太弱，不足以在结构上改变模型行为。**

---

## 本轮新假设

已知机制证据显示：

1. 域偏移在共享前端很早就出现；
2. few-shot 梯度主要打在 frontend / transformer / head；
3. 只改 gate 的 routing 力度不够。

因此本轮尝试把改动前移到 `conv1d` 之后，但**不污染 invariant 路径**：

### 新架构：Frontend Adapter

在 `DDMambaBlock` 中：

- 先得到共享前端特征 `x_shared = act(conv1d(...))`
- 再根据目标域标签生成一个 **target-conditioned frontend adapter**
- 令：
  - invariant 路径继续只看 `x_shared`
  - combined/spec 路径看 `x_combined = x_shared + adapter(...)`

这样做的目的：

- 把 target-specific 修正前移到 `conv1d` 后；
- 但 invariant 路径仍保持纯净；
- 同时让 `frontend_mean` 对齐继续只约束共享前端。

---

## 已实现的两类 adapter

### 1. `target_affine`

对 `x_shared` 做 target-only 的逐通道仿射调制：

- scale
- bias

### 2. `target_residual`

对 `x_shared` 做 target-only 的 1x1 Conv residual：

- `x_shared + adapter(x_shared)`

### 3. `target_residual + dual_state`

额外和 `dual_state` 组合，测试：

- 早期 target adapter
- + scan 物理隔离

是否可以形成互补。

---

## 当前已启动的夜间实验

统一基线配置：

- task: `FD001_TO_FD003`
- shots: `5-shot`
- seeds: `42,43,44`
- backbone: `dd_spd`
- inv alignment tap: `frontend_mean`
- `lambda_inv_mmd = 0.1`
- `lambda_spec_domain = 0.1`
- `target_lr = 1.5e-3`
- cosine scheduler

### 运行脚本

- `scripts/run_overnight_architecture_search.sh`

### 日志

- 主日志：
  - `runs/logs/overnight_architecture_search_20260408.log`
- 摘要：
  - `runs/logs/overnight_architecture_search_20260408.summary.txt`

### 计划跑的三个实验

1. `runs/cd_mambatt_v3_frontaffine_20260408`
2. `runs/cd_mambatt_v3_frontresidual_20260408`
3. `runs/cd_mambatt_v3_frontresidual_dualstate_20260408`

---

## 配套诊断

每个实验跑完后，夜间脚本会自动调用：

- `scripts/analyze_conditioning_effect.py`

输出到：

- `runs/logs/<run_name>.conditioning.json`

核心检查：

- 强制 `domain_label=0/1` 时：
  - source / target RMSE 是否明显变化
  - gate 是否明显变化
  - prediction 是否明显变化
  - adapter / shift 参数是否学到了非平凡量

也就是说，明早不仅看“结果好不好”，还看：

> **新结构到底有没有真的改变模型行为。**

---

## 明早优先查看

1. `runs/logs/overnight_architecture_search_20260408.summary.txt`
2. `runs/cd_mambatt_v3_frontaffine_20260408/FD001_TO_FD003/summary.json`
3. `runs/cd_mambatt_v3_frontresidual_20260408/FD001_TO_FD003/summary.json`
4. `runs/cd_mambatt_v3_frontresidual_dualstate_20260408/FD001_TO_FD003/summary.json`
5. 对应的 `*.conditioning.json`

