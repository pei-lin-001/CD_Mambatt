# CD-MambAtt 跨域阶段性判断报告

日期：2026-03-31  
项目：`CD-MambAtt`  
主任务：`FD001 -> FD003`，`5-shot`，target few-shot per-seed resample

---

## 1. 这份报告回答什么问题

本报告用于回答：

1. 结合 `cd_mambatt` 方案文档（PDF 转文本）与当前实验，跨域思路是否成立？
2. 哪些模块已经被实验支持？
3. 哪些模块目前不支持，或者至少在当前实现下没有收益？
4. 当前最合理的“工作版 CD-MambAtt”应该长什么样？

---

## 2. 总体结论

结论可以直接概括为：

> **CD-MambAtt 这条跨域路线是成立的，而且已经有实验结果支撑。**

但同时也要更精确地说：

> **PDF 中“所有模块都叠加后一定更强”的版本，目前没有被实验支持。**

当前实验更支持的版本是：

> **MMD 粗对齐 + 置信伪标签 + 局部单调性损失**

而不是：

> **MMD + 伪标签 + 跨域 contrastive + 单调性 全部一起上**

---

## 3. 与 PDF 方案的逐点对照结论

### 3.1 已被实验支持的部分

#### A. 跨域动机成立

原始单域 MambAtt 直接跨域效果很差：

- Direct transfer mean RMSE = **33.1579**

这说明 PDF 的核心出发点是对的：

- 原始 MambAtt 在跨域场景下会显著退化
- 必须加入跨域适配机制

#### B. Step 1：粗对齐（MMD）有效

最强 MMD-only 结果：

- `lambda_mmd = 0.1`
- mean RMSE = **21.8335**

这比严格 `5-shot` full fine-tune baseline 更好：

- strict full fine-tune baseline = **22.7826**

说明：

- PDF 中“先做粗对齐”的思路是有效的

#### C. Step 2：置信伪标签有效，但不够稳定

较优 pseudo 版本：

- `lambda_pseudo = 0.5`
- pseudo quantile = `0.5 -> 0.9`
- mean RMSE = **22.2134**

说明：

- 特征空间伪标签不是无效想法
- 但它单独还不足以成为当前最强模块

#### D. Step 4：局部单调性损失最有价值

这是目前最强阳性结果：

- `lambda_monotonic = 0.05`
- mean RMSE = **21.1291**
- std = **1.5928**

相较于当前最佳 MMD-only：

- `21.8335 -> 21.1291`
- 提升约 **0.70 RMSE**

这说明 PDF 中“局部单调性损失”不是装饰项，而是目前最关键的新模块。

---

### 3.2 当前未被支持的部分

#### Step 3：跨域 contrastive / N-tuplet

当前实现下，contrastive 没有带来收益：

- pseudo + contrastive `0.1`：**22.2511**
- pseudo + contrastive `0.05`：**22.2396**
- monotonic `0.05` + contrastive `0.05`：**22.3660**

而不加 contrastive 的最优 monotonic 版本是：

- **21.1291**

因此当前最稳妥的判断是：

> **cross-domain contrastive 在当前实现/当前协议下并没有帮助，反而拖后腿。**

这不等于理论永久错误，但至少说明：

- 它不是当前性能提升的关键来源
- 如果继续保留，必须重新设计，而不是继续机械扫权重

---

## 4. 关键实验结果表

下面表格按“当前对我们最有解释力”的顺序列出。

| 方法 | 说明 | Mean RMSE | Std | 备注 |
|---|---|---:|---:|---|
| Direct transfer | 原始单域模型直接跨域 | 33.1579 | 7.9510 | 明显失效 |
| Full fine-tune baseline | 5-shot, resample | 22.7826 | 1.4936 | 严格 baseline |
| CD-MambAtt v1 + MMD | `lambda_mmd=0.1` | 21.8335 | 1.7380 | 当前旧最好 |
| v2 + pseudo | `lambda_pseudo=0.5`, `q:0.5->0.9` | 22.2134 | 2.4842 | 有用但不稳 |
| v2 + pseudo + contrastive | `lambda_contrastive=0.05` | 22.2396 | 2.5070 | 无收益 |
| v2 + pseudo + monotonic | `lambda_monotonic=0.1` | 21.8611 | 1.2578 | 明显改善稳定性 |
| **v2 + pseudo + monotonic** | **`lambda_monotonic=0.05`** | **21.1291** | **1.5928** | **当前最好均值** |
| v2 + pseudo + monotonic + contrastive | `0.05 + 0.05` | 22.3660 | 0.7207 | 整体变差 |

---

## 5. 当前最合理的方法定义

如果现在就要给出一个“基于实验而不是基于设想”的方法定义，那么最合理的是：

### 当前工作版 CD-MambAtt

1. **共享编码器**
2. **MMD 粗对齐**
3. **置信过滤伪标签**
4. **目标域局部单调性损失**

更简洁地写：

> **CD-MambAtt (working version) = MMD + confidence-filtered pseudo-labeling + local monotonicity**

而不建议当前把 contrastive 写成核心贡献模块。

---

## 6. 对研究价值的判断

就目前结果而言，这个方向已经不再是“看起来可能行”，而是：

> **已经有了明确实验支撑、并且跑出了优于已有最好 baseline 的结果。**

这意味着：

- 方向是对的
- 论文主线是能成立的
- 但方法表述需要收缩成“被实验验证过的版本”

换句话说：

### 现在最值得坚持的叙事

1. 原始 MambAtt 直接跨域失效  
2. MMD 粗对齐可以恢复一部分性能  
3. 置信伪标签可以辅助，但稳定性有限  
4. 局部单调性约束显著提升跨域 few-shot RUL 性能  
5. 因而，跨域场景下最关键的并不一定是更复杂的对比项，而是更符合退化先验的结构性约束

这个叙事是完整的，而且目前是被结果支持的。

---

## 7. 当前最重要的研究判断

### 判断 1

> **跨域版本值得继续做。**

因为已经有清晰增益：

- best previous mean = **21.8335**
- current best mean = **21.1291**

### 判断 2

> **PDF 方案中真正最有价值的新增点，目前是 monotonicity，不是 contrastive。**

### 判断 3

> **后续方法定义应该以实验结果为准，而不是强行保留所有最初设想。**

---

## 8. 接下来最合理的实验顺序

### 第一优先级

继续围绕 monotonic-only 最优版本微调：

- 已启动：`lambda_monotonic = 0.02`

### 第二优先级

增强结论可信度，而不是盲目加新模块：

- 增加 seeds
- 或提高 target validation stability

### 第三优先级

再考虑是否重做 contrastive，而不是继续当前版本扫参：

- 当前 contrastive 结论已经比较负面

---

## 9. 当前未完成但正在进行的事项

截至当前更新，`lambda_monotonic = 0.02` 也已经完成。

结果是：

- mean RMSE = **21.6954**

因此：

- 它仍然优于 MMD-only best (`21.8335`)
- 但不如当前最佳 `lambda_monotonic = 0.05` (`21.1291`)

这进一步加强了当前判断：

> **局部单调性确实有效，但过小的权重也会削弱收益；当前最优仍是 `lambda_monotonic = 0.05`。**

---

## 10. 最终一句话总结

> **CD-MambAtt 的跨域思路已经被证明有效；当前最成功的版本不是“全模块叠加”，而是“粗对齐 + 置信伪标签 + 局部单调性”。**

---

## 11. 2026-04-01 补充：多任务 `5 seeds` 泛化验证

在上一阶段，我们的结论主要建立在：

- `FD001 -> FD003`
- `3 seeds`

这足以说明方向成立，但还不足以证明结论具有跨任务泛化性。

因此，后续按照固定最优配置，做了额外的多任务验证。

### 固定配置

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

---

## 12. 一个关键工程修复：multi-condition 归一化条件键错误

在 `FD002/FD004` 上，之前出现过验证值异常爆炸的问题。

根因不是模型本身，而是数据归一化逻辑：

- 多工况数据原本需要按 operating condition 分组归一化
- 但旧逻辑直接使用高精度浮点 operating settings 作为 condition key
- 这会把本来有限个工况错误拆成大量“伪工况”
- 最终导致归一化严重失真，验证 RMSE 可能出现极端异常值

修复后：

- 在 `cd_mambatt/data.py` 中将 condition key 离散化到整数级
- `FD002/FD004`、`FD001/FD004` 的结果恢复到正常数量级

因此：

> **修复后的 multi-condition 结果才是有效结果。**

---

## 13. 新结果：方法已经不只在单一任务上成立

### 13.1 `FD003 -> FD001`

- direct mean RMSE = **24.6059**
- CD mean RMSE = **19.8137**
- gain = **4.7922 RMSE**（约 **19.5%**）
- std = **1.2628**

判断：

> 反向单工况迁移同样明显收益，说明方法并不依赖固定迁移方向。

### 13.2 `FD002 -> FD004`（multi-condition）

- direct mean RMSE = **29.5028**
- CD mean RMSE = **22.1776**
- gain = **7.3252 RMSE**（约 **24.8%**）
- std = **2.1685**

判断：

> 修复归一化后，方法在多工况任务上依然成立，这是当前最重要的新证据之一。

### 13.3 `FD001 -> FD004`（更难迁移）

- direct mean RMSE = **32.3820**
- CD mean RMSE = **24.3449**
- gain = **8.0371 RMSE**（约 **24.8%**）
- std = **2.4762**

判断：

> 在更难的迁移对上仍有大幅改进，说明当前方法不是只对“简单 pair”有效。

### 13.4 `FD004 -> FD002`（multi-condition 反向迁移）

- direct mean RMSE = **21.3666**
- CD mean RMSE = **19.5311**
- gain = **1.8354 RMSE**（约 **8.6%**）
- std = **1.0414**

判断：

> 这个任务上的 direct baseline 本身已经相对较强，因此可提升空间更小；但即便如此，当前方法依然取得了稳定正增益。

### 13.5 `FD001 -> FD002`（single-condition → multi-condition）

- direct mean RMSE = **31.1029**
- CD mean RMSE = **22.9457**
- gain = **8.1572 RMSE**（约 **26.2%**）
- std = **2.8311**

判断：

> 这是一个很关键的 stage-2 结果，因为它同时跨越了工况复杂度差异；当前方法依然给出了大幅改进。

### 13.6 汇总表

| 任务 | Direct RMSE | CD RMSE | Gain | Gain % |
|---|---:|---:|---:|---:|
| `FD001 -> FD002` | 31.1029 | **22.9457** | 8.1572 | 26.2% |
| `FD003 -> FD001` | 24.6059 | **19.8137** | 4.7922 | 19.5% |
| `FD002 -> FD004` | 29.5028 | **22.1776** | 7.3252 | 24.8% |
| `FD001 -> FD004` | 32.3820 | **24.3449** | 8.0371 | 24.8% |
| `FD004 -> FD002` | 21.3666 | **19.5311** | 1.8354 | 8.6% |

---

## 14. 对当前研究阶段的更新判断

如果只看 3 月 31 日晚上的状态，我们最多只能说：

> “方向被验证了，单任务雏形跑通了。”

但加入 4 月 1 日这批结果后，更准确的判断已经变成：

> **方法已经在多个 C-MAPSS 迁移对上表现出一致的正向增益。**

也就是说，当前项目阶段已经从：

- 单任务 proof-of-concept

推进到：

- 多任务泛化证据初步成立
- 并且已经覆盖了 multi-condition 的双向迁移
- 以及 single-condition → multi-condition 的 stage-2 迁移

这比之前更接近“可投稿实验包”的标准。

---

## 15. 当前仍然成立的核心结论

这些新结果并没有推翻旧判断，反而加强了它们：

### 结论 1

> **跨域版本值得继续做。**

现在这个结论不再只依赖 `FD001 -> FD003`。

### 结论 2

> **当前真正稳定有效的核心模块仍然是：MMD + 置信伪标签 + 局部单调性。**

### 结论 3

> **当前 contrastive 版本依然不应被视为核心贡献。**

因为在主任务上的实验依旧没有支持它。

---

## 16. 下一阶段最合理的动作

既然现在已经有：

- 多任务
- `5 seeds`
- multi-condition 修复后的有效结果

那么下一阶段最重要的事情就不再是继续小范围扫权重，而是：

1. **补 published baseline**
   - DPMA
   - deep flow kernel / meta-learning 类方法
2. **补更多迁移对**
   - 如 `FD004 -> FD002`
3. **补第二数据集**
   - `XJTU-SY`
4. **整理投稿版叙事**
   - 强调 monotonic prior 比当前 contrastive 实现更关键

---

## 17. 更新后的一句话总结

> **CD-MambAtt 当前“真正被实验支持”的版本，已经在多个 C-MAPSS 迁移任务上稳定优于 direct transfer；最关键的有效增益来自粗对齐、置信伪标签和局部单调性，而不是当前的 contrastive 实现。**

---

## 18. 对 canonical 任务的进一步校正：`FD001 -> FD003` 正式 `5 seeds`

前面我们最常引用的 canonical 最优结果是：

- `lambda_monotonic = 0.05`
- `3 seeds`
- mean RMSE = **21.1291**

但为了和后续多任务 `5 seeds` 结果统一口径，我们又做了正式复跑。

### 18.1 `lambda_monotonic = 0.05`

- direct mean RMSE = **34.7179**
- CD mean RMSE = **22.0342**
- std = **1.4408**

### 18.2 `lambda_monotonic = 0.1`

- direct mean RMSE = **34.8204**
- CD mean RMSE = **21.9608**
- std = **1.4574**

### 18.3 这意味着什么

这说明：

1. `FD001 -> FD003` 上，当前方法仍然**明显优于 direct transfer**
2. 但 `3 seeds` 的 **21.1291** 比正式 `5 seeds` 结果更乐观
3. 在正式 `5 seeds` 下，`monotonic = 0.05` 与 `0.1` 已经变成**近乎打平**

所以更严谨的表述应该改成：

> **当前方法的跨任务泛化性已经成立，但 canonical 任务上的“最优单点超参数”还没有在 5-seed 协议下完全拉开差距。**

这不会推翻主结论，但会影响我们后续写论文时对“最佳配置”的表述方式。

### 18.4 更新后的稳妥说法

目前最稳妥的说法不是：

> “`lambda_monotonic = 0.05` 已经被完全证明是全局最优。”

而是：

> “`lambda_monotonic = 0.05` 是当前多任务验证所采用的主配置；在 canonical 任务的正式 5-seed 复跑中，`0.1` 也表现出几乎等价、且略优的结果。”
