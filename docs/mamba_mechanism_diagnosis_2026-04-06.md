# Mamba 机制诊断（2026-04-06）

对应脚本：

- `scripts/mamba_mechanism_diagnosis.py`
- 结果快照：`docs/generated/mamba_mechanism_diagnosis_seed42.json`

诊断对象：

- 基准 run：`runs/cd_mambatt_v3_spd_fullmatch/FD001_TO_FD003`
- seed：`42`
- source ckpt：同 run 的 `source/best.pt`
- adapted ckpt：同 run 的 `cd_stage/best.pt`

---

## 六个问题的当前答案（seed42 机制体检版）

### Q1. 域偏移最早从哪里进来？

结论：**非常早，在共享前端就已经很强**，不是等到 SPD 分支后才出现。

关键证据（source/test vs target/test 的线性域分类准确率）：

- `x_conv`: **1.00**
- `z_raw`: **1.00**
- `D_x`: **1.00**
- `B_inv`: **0.995**
- `C_inv`: **1.00**
- `dt_inv`: **0.975**
- `features`: **0.89**

解释：

- `conv1d` 后的共享 `x` 已经几乎可被线性完美分域。
- 所以只在 `dt/B/C` 处做分解，天然是在“后补救”，不是在最早入口截断域偏移。

### Q2. few-shot 目标监督主要把梯度打到哪里？

结论：**先打到 head / transformer / shared frontend，spec gate 最弱。**

target supervised loss 的梯度范数（source-init）：

- `mamba_frontend`: **5517**
- `head`: **3023**
- `mamba_out_proj`: **1919**
- `transformer`: **1175**
- `mamba_inv_selectivity`: **538**
- `mamba_dynamics`: **151**
- `mamba_spec_gate`: **87**

结论含义：

- few-shot 监督一开始并不是优先“唤醒 spec 分支”。
- 它更像先做**共享表征校准 + decoder 校准**。

### Q3. 跨域差异主要更像 `dt` 问题还是 `B/C` 问题？

结论：**当前证据更偏向 `B/C` 更“带域”，但对最终预测的实际贡献非常小。**

域分类准确率：

- `dt_inv`: **0.975**
- `B_inv`: **0.995**
- `C_inv`: **1.00**

但做推理期 ablation 时：

- target test `full`: **22.40**
- target test `inv_only`: **22.35**
- target test `dt_only`: **22.34**
- target test `bc_only`: **22.41**

解释：

- `B/C` 表征本身更容易分域；
- 但目前训练到的 spec 注入量极小，以至于无论只留 `dt` 还是只留 `B/C`，对最终 RMSE 几乎都没影响。

### Q4. 59× hidden drift 的根因是什么？

结论：**scan 的递推确实在放大 drift，但放大的源头来自共享前端，而不是 gate 本身。**

step20 / step1 MMD 比例：

- `x_conv`: **1.10x**
- `D_x`: **1.01x**
- `inv_state`: **7.09x**
- `mixed_state`: **6.63x**

解释：

- 输入侧域差异在时间上并不会自然爆炸；
- 进入 SSM 递推后，状态 drift 明显累计放大；
- 但 `mixed_state` 没有比 `inv_state` 更糟很多，说明**“是否混入 spec”不是当前主矛盾**；
- 当前更像是**共享前端已经带偏 + SSM 递推负责累积放大**。

### Q5. spec 分支到底有没有在 target calibration 中起作用？

结论：**目前几乎没有真正承担主要校准角色。**

使用度证据：

- source gate mean: **0.184**
- target gate mean: **0.148**
- source spec/feature norm ratio: **0.016**
- target spec/feature norm ratio: **0.035**

并且 ablation 后 target RMSE 几乎不变：

- `full`: **22.40**
- `inv_only`: **22.35**

解释：

- target 上 gate 甚至比 source 更低；
- spec feature 只占很小残差；
- 拿掉 spec 几乎不伤性能；
- 所以当前 spec 分支更多像“弱扰动”，不是 few-shot target 的主要适配通路。

### Q6. 瓶颈更像在 Mamba 还是 Transformer / decoder？

结论：**快速 few-shot 适配更依赖 Transformer/head，而不是 spec-gate 单独调参。**

6 epoch quick adaptation（同 source ckpt 起跑）：

- `head_only`: **37.73**
- `transformer_head`: **20.43**
- `spec_gate_head`: **29.91**
- `spec_gate_transformer_head`: **20.49**
- `full`: **21.85**

解释：

- 只调 head 几乎不够；
- 一旦允许 `transformer + head`，效果立刻大幅上升；
- 只调 `spec_gate + head` 明显不够；
- 加上 transformer 后才接近 full；
- 说明**few-shot 适配的关键通路更像 decoder / sequence aggregation 校准，而不是单靠 SPD spec 支路。**

---

## 对后续结构设计的直接含义

当前最重要的认识不是“再继续拆 dt/B/C”，而是：

1. **域偏移过早进入了共享前端**（`conv1d` 后就很强）。
2. **SSM 负责把已有偏移递推放大**，但不是唯一源头。
3. **spec 分支目前几乎没有承载 target calibration。**
4. **few-shot 适配真正有效的短路径更像 Transformer / decoder 校准。**

所以后续研究优先级应从：

- “继续在 scan 内部做更复杂拆分”

转向：

- “前端共享表示如何减域偏”
- “few-shot 梯度如何更稳定地路由到 decoder / aggregation”
- “是否需要更早的 invariant/specific 切分，而不是只在 `dt/B/C` 处切”

