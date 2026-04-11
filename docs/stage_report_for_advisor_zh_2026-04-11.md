# CD-MambAtt 项目阶段总结报告（供导师汇报）

最后更新：`2026-04-11`

作者说明：

- 本文档面向**阶段性汇报**使用，重点不是记录所有零散试验细节，而是把当前项目的**目标、方法演进、关键结果、核心诊断、现有问题与下一步计划**系统整理清楚。
- 更细的逐次实验记录请查看：
  - [`cross_domain_experiment_log.md`](./cross_domain_experiment_log.md)
  - [`project_status.md`](./project_status.md)
  - [`project_overview_zh_2026-04-08.md`](./project_overview_zh_2026-04-08.md)

---

## 一、项目目标与研究背景

本项目围绕 **MambAtt（Mamba + Attention）** 展开，研究任务是：

> **跨域 few-shot 剩余寿命预测（RUL）**  
> 即在一套 C-MAPSS 子数据集上学习退化规律，再迁移到另一套只有少量标注样本的新子数据集上。

从问题定义上看，本项目与原始 MambAtt 论文并不完全相同：

- 原始论文重点是：
  - 单域监督预测；
  - 同子集 self-supervised 预训练 + one-shot 预测。
- 我们当前研究重点是：
  - **跨域 few-shot 迁移**；
  - 特别关注**分布偏移、目标域标注极少、不同工况/故障模式迁移**这类实际更困难的情形。

因此，本项目的总体路线可以概括为三步：

1. **先确认骨干是否可靠**  
   复现 MambAtt 的监督与自监督部分。
2. **再建立跨域基线是否成立**  
   证明 MambAtt 确实可以做跨域 few-shot RUL，而不是只适用于单域。
3. **最后在 Mamba 内部做结构创新**  
   尝试从 SSM 内部动态机制出发，提升跨域泛化能力。

---

## 二、当前项目整体状态

截至目前，项目已经不再处于“环境搭建/代码跑通”阶段，而处于：

> **基础复现完成 + 跨域方法建立 + 内部创新探索与机制诊断阶段**

可以用一句话概括当前进展：

> 我们已经完成了 MambAtt 的监督与自监督主路径复现，证明了基于 MambAtt 的跨域 few-shot RUL 方法是可行的，并构建出一条稳定的跨域主线；同时我们已经实现了 Mamba 内部的 SPD/DD-SSM 创新，并通过一系列机制实验发现当前真正的主矛盾更偏向“前端域偏移 + invariant path 稳定性”，而不只是 SSM 内部参数解耦本身。

---

## 三、数据集、任务设置与评价协议

## 3.1 数据集

主要使用 NASA **C-MAPSS** 数据集，包含：

- `FD001`
- `FD002`
- `FD003`
- `FD004`

这些子数据集之间在以下方面存在差异：

- 工作条件（single-condition / multi-condition）
- 故障模式（单故障 / 多故障）
- 数据分布与退化轨迹形态

因此非常适合作为跨域 RUL 迁移的 benchmark。

## 3.2 当前主要跨域任务

项目中已经重点实验过的 transfer pair 包括：

- `FD001 → FD003`
- `FD003 → FD001`
- `FD002 → FD004`
- `FD001 → FD004`
- `FD004 → FD002`
- `FD001 → FD002`

其中：

- `FD001 → FD003` 是当前 canonical 主任务；
- `FD003 → FD001` 是当前最关键的**难点诊断任务**；
- `FD002/FD004` 方向反映多工况迁移难度；
- `FD001 → FD004` 反映更强分布偏移下的迁移性能。

## 3.3 当前主要协议

项目里出现过多种 protocol，但当前最重要的有三类：

### A. 单域监督复现协议

- 输入特征：21 sensors
- window size：20
- stride：1
- RUL cap：125
- engine-level 80/20 split

### B. 跨域 few-shot 主线协议

- 默认 target few-shot：**5-shot**
- target validation units：通常为 `10`
- 使用 target train 剩余部分作为 unlabeled pool
- 评价指标：
  - `RMSE`
  - `SCORE`

### C. 论文 Task-Embedding MAML 对齐协议（paper-aligned probe）

为了和特定论文做更接近的对比，我们还增加了：

- `target_shots = 1`
- `window_size = 30`
- `sensor_subset = paper14`
- `normalization_mode = minmax`

这套协议是后期为了诊断 `K=1` 场景专门补充的。

---

## 四、代码与工程完成情况

## 4.1 环境与训练基础设施

已完成：

- WSL2 + CUDA 训练环境
- conda 环境：`cd_mamba`
- C-MAPSS 数据准备与加载流程
- 多脚本训练/实验框架
- 结果文档化与实验归档

## 4.2 当前主要训练入口

| 文件 | 作用 |
|---|---|
| `train_supervised.py` | 单域监督训练，用于骨干复现 |
| `train_self_supervised.py` | 原论文 same-subset 自监督复现 |
| `train_cross_domain_baseline.py` | direct transfer / few-shot fine-tune 基线 |
| `train_cd_mambatt_v2.py` | 稳定跨域主线：MMD + pseudo + monotonic |
| `train_cd_mambatt_v3.py` | SPD/DD-SSM 与机制实验主入口 |
| `train_fomln_baseline.py` | FOMLN published baseline 复现 |

## 4.3 当前主要模型/功能文件

| 文件 | 作用 |
|---|---|
| `cd_mambatt/models/mambatt.py` | 主模型定义，已扩展支持 SPD 和多种输出辅助接口 |
| `cd_mambatt/models/dd_mamba.py` | DDMambaBlock，SPD / DD-SSM 核心实现 |
| `cd_mambatt/data.py` | 数据读取、滑窗、normalization、sensor subset |
| `cd_mambatt/self_supervised.py` | 论文自监督三种任务实现 |
| `cd_mambatt/pseudo_labeling.py` | 伪标签与阶段统计 |
| `cd_mambatt/losses/mmd.py` | MMD 与 conditional MMD |
| `cd_mambatt/losses/domain_adversarial.py` | GRL / 域判别模块 |

## 4.4 已完成的重要工程性补充

后期为了更严谨的对比和机制分析，又补了几项关键能力：

1. **paper14 sensor subset 支持**
2. **Min-Max normalization 支持**
3. **多类机制诊断脚本**
4. **实验脚本统一整理到 `experiments/`**
5. **文档整理为主文档 + history + generated 结构**

这意味着项目现在已经具备：

> **持续研究、结果对比、机制诊断、文档汇报**的完整工程基础。

---

## 五、第一部分：MambAtt 监督复现结果

## 5.1 目标

先验证原始 MambAtt 骨干在单域监督场景下是否被正确实现。

## 5.2 论文参考结果

- 论文 Table 3：
  - `FD001 RMSE = 11.46`
- 论文 Transformer-decoder mean：
  - `12.34`

## 5.3 当前本地最好结果

当前最好监督线：

- `transformer_norm_mode = post`
- `dim_feedforward = 84`
- `val_all_windows = True`
- `lr = 5e-4`
- `weight_decay = 1e-4`
- seeds：`42,43,44`

结果：

- fixed split mean test RMSE = **15.0937**
- resampled split mean test RMSE = **15.3024**

## 5.4 已验证的重要经验

监督复现过程中已经做过多组控制实验，当前结论包括：

- `d_ff = 84` 比更大的 `128 / 256 / 512 / 2048` 更好
- `val_all_windows` 比 last-window-only validation 更有利于模型选择
- 额外包裹 Mamba 的 residual block 无明显收益，反而变差
- PyTorch 原生 Transformer 实现不如当前 custom 版本
- 输出端 dropout 保留，Transformer 内部 dropout 过大通常更差

## 5.5 阶段结论

监督骨干的结论非常明确：

> **我们已经完成了结构级别的复现，但尚未达到论文数值级别的精确复现。**

当前判断是：

- 骨干实现大方向是对的；
- 剩余 gap 更像是隐藏协议、训练细节或论文未明示的实现选择导致；
- 因此监督部分目前适合作为**可靠骨干**，但不再值得继续大量投入精力去追 11.46。

---

## 六、第二部分：原论文自监督复现结果

## 6.1 目标

复现原论文的 same-subset self-supervised 训练流程，确认我们对原论文的方法理解是否完整。

## 6.2 已实现的自监督任务

已经实现：

- temporal ordering loss
- N-tuplet loss
- pseudo-label loss

以及完整的 same-subset 训练流程：

- 同子集 SSL 预训练
- encoder 导出
- one-shot fine-tune / eval

## 6.3 当前结果

| 子集 | 论文 Table 5 | 当前本地结果 | 结论 |
|---|---:|---:|---|
| FD001 | 31.7270 | **31.6116 ± 2.5595** | 基本贴近论文 |
| FD002 | 30.3269 | **23.5244** | seed42 only，偏乐观 |
| FD003 | 32.3329 | **35.5230 ± 2.6474** | 仍高于论文 |
| FD004 | 32.6633 | **38.8006** | 偏差较大 |

## 6.4 额外 sanity check

在 FD001 上，还验证了：

| 设置 | Mean test RMSE |
|---|---:|
| no pretrain | 32.8112 |
| paper-full SSL pretrain | **31.6116** |

以及 loss ablation：

| 设置 | Test RMSE |
|---|---:|
| temporal only | 32.1312 |
| temporal + N-tuplet | 26.0186 |
| full | **24.9943** |

这些趋势与论文叙事基本一致。

## 6.5 阶段结论

这条线的当前定位是：

> **实现完成，可复用，但当前暂停。**

原因：

- FD001 已经说明实现逻辑基本正确；
- 但这条线本质上是**同子集**训练，不直接解决跨域问题；
- 因此它的最大价值是：
  - 为 later reuse 提供模块；
  - 为理解原论文提供支撑；
  - 为后续 union SSL 等跨域思路提供参考。

---

## 七、第三部分：跨域基线与 v2 主线结果

## 7.1 目标

在 MambAtt 骨干基础上建立一个**稳定成立**的跨域 few-shot 方法主线。

## 7.2 基线演进

我们先后做过：

1. direct transfer
2. 5-shot head fine-tune
3. 5-shot full fine-tune
4. MMD-only
5. MMD + pseudo-labeling
6. MMD + pseudo + monotonicity

最终形成当前稳定主线：

> **CD-MambAtt v2 = MMD + pseudo-labeling + monotonicity**

## 7.3 5-shot 多任务结果（5 seeds）

| 任务 | Direct transfer | CD-MambAtt v2 | 提升 |
|---|---:|---:|---:|
| `FD001 → FD003` | 34.8204 | **21.9608** | 12.8595 |
| `FD003 → FD001` | 24.6059 | **19.8137** | 4.7922 |
| `FD002 → FD004` | 29.5028 | **22.1776** | 7.3252 |
| `FD001 → FD004` | 32.3820 | **24.3449** | 8.0371 |
| `FD004 → FD002` | 21.3666 | **19.5311** | 1.8354 |
| `FD001 → FD002` | 31.1029 | **22.9457** | 8.1572 |

## 7.4 阶段结论

这是当前项目最根本的成果之一：

> **跨域 few-shot 这条主线已经成立。**

含义是：

- 我们不再处于“方法是否有效”的不确定阶段；
- 问题已经转向：
  - 怎样进一步把结果做得更强；
  - 怎样把创新点做得更有说服力；
  - 怎样解释为什么有效。

---

## 八、第四部分：published baseline 对比情况

## 8.1 FOMLN 复现与对比

我们已经在本地环境中复现了 FOMLN，并完成了若干 `15-shot` 对比。

### 当前 matched-shot in-house comparison

| 任务 | CD-MambAtt 15-shot | reproduced FOMLN 15-shot | Δ (CD - FOMLN) |
|---|---:|---:|---:|
| `FD001 → FD003` | **20.1384** | 25.4748 | -5.3364 |
| `FD003 → FD001` | **19.5788** | 20.1261 | -0.5474 |
| `FD002 → FD004` | **23.2879** | 26.0206 | -2.7327 |
| `FD001 → FD004` | **23.9469** | 25.0808 | -1.1340 |

### 应当如何表述

这组结果**可以说是积极的**，但不能过度表述为：

- “严格公平打败了 FOMLN 论文”

更稳妥的说法是：

> **在当前 matched-shot 的本地复现实验中，CD-MambAtt 优于复现的 FOMLN。**

原因是仍有若干协议差异：

- target validation 规则不一致
- unlabeled target pool 使用方式不一致
- 预处理 pipeline 不完全统一
- reproduced FOMLN 仍在 paper-alignment-in-progress 阶段

## 8.2 Task-Embedding MAML 文献对比

这篇论文的重要点在于：

- 同样是 C-MAPSS 跨域 few-shot
- 但核心协议是：
  - **`K = 1`**
  - `window = 30`
  - `14 sensors`
  - `minmax`
  - `50 repetitions`

它的 paper 数值如：

- `FD001→FD003 = 24.34`
- `FD003→FD001 = 24.24`
- `FD002→FD004 = 26.96`
- `FD004→FD002 = 23.24`

### 注意

这些数值**不能直接和我们 5-shot 主线比较**。

因为：

- 我们主线多是 `5-shot`
- 它是 `1-shot`

## 8.3 MetaDFKN

MetaDFKN paper 报告数值很强，例如：

- `FD001 → FD003 = 13.03`
- `FD003 → FD001 = 5.48`
- `FD002 → FD004 = 9.38`
- `FD001 → FD004 = 13.11`

但目前该论文的 target-domain protocol 仍不清晰，因此现在只能将其定位为：

> **强 reported baseline，但还不是严格 fair comparator。**

---

## 九、第五部分：SPD / DD-SSM 创新线

## 9.1 初衷

原始 v2 主线虽然有效，但创新更多停留在：

- feature-level loss
- pseudo-labeling
- monotonic regularization

因此我们尝试把创新推进到 Mamba 内部，形成：

> **SPD / DD-SSM（Selectivity Projection Disentanglement）**

其核心想法是：

- 在 Mamba 内部生成选择性参数 `Δ / B / C` 的地方做 invariant / specific 分拆；
- 希望从动态机制层面增强跨域泛化。

## 9.2 最好的非 SSL SPD 结果

### A. SPD + inv-MMD + spec-domain + LR2e-3 + cosine

canonical `FD001→FD003`：

- **20.37 ± 0.47**

相对 v2 参考：

- v2：21.96
- SPD：20.37

### B. SPD + frontend alignment + tuned cosine LR

进一步引入：

- invariant 对齐前移到 `frontend_features`
- `target_lr` 从 `2e-3` 调到 `1.5e-3`

结果：

- `FD001→FD003 = **20.0151 ± 0.7523**`

这是当前**最好非 SSL canonical 结果**。

## 9.3 跨任务表现

| 任务 | 结果 | 结论 |
|---|---:|---|
| `FD001→FD003` | **20.0151 / 20.37** | 明显优于 v2 |
| `FD001→FD004` | **23.47 左右** | 小幅优于 v2 |
| `FD003→FD001` | **21.03~21.14** | 仍落后于 v2 |

## 9.4 阶段结论

SPD 的当前定位不能简单说“失败”或“成功”，更准确应是：

> **SPD 在 canonical 任务上已经证明有真实收益，但目前还没有形成跨任务稳定成立的统一机制方案。**

这也是为什么项目后期越来越重视**机制诊断**而不是继续盲目堆结构。

---

## 十、第六部分：机制诊断与研究认识

这是目前项目最有研究价值的部分之一。

## 10.1 诊断 1：Mamba 隐状态 drift 严重

通过机制探针发现：

- Mamba 隐状态的域偏差会从时间步 1 递推放大到时间步 20
- 放大量级约为 **59 倍**

这说明：

> SSM 递推确实会累积和放大域偏差。

这也是 SPD/DD-SSM 研究的最初理论动机之一。

## 10.2 诊断 2：SPD 当前的解耦效果偏弱

关键观察：

- gate 只从 **0.12 → 0.18**
- inv vs combined 域分类准确率差异仅 **0.3%**
- spec 分支贡献约 **15%**

结论：

> 当前 SPD 虽然在结构上“拆开了”，但功能上并未形成强烈、清晰的 invariant / specific 分工。

这也解释了为什么很多继续强化 spec 分支的尝试收益有限。

## 10.3 诊断 3：域偏移很早就进入共享前端

进一步的机制实验发现：

- 域信息在共享前端（如 conv1d 之后）就已经很强；
- 因此只在后面的 inv_mean / domain_features 上对齐，往往偏晚。

这直接催生了：

> **frontend alignment**  
> 即把 invariant 对齐位置前移到更早的表征层。

并且在实验中它确实比后部对齐更有效。

## 10.4 诊断 4：union SSL 的核心作用不是打开 spec gate，而是强化 invariant path

在 `union SSL + no-spec adaptation` 的机制诊断中发现：

- `x_conv` MMD：`0.286 → 0.213`
- combined-core MMD：`0.638 → 0.181`
- invariant drift ratio：`7.73 → 1.98`
- mixed drift ratio：`6.73 → 1.46`
- target gate mean：`0.119 → 0.058`
- target `inv_only` ablation RMSE：`19.80 → 18.03`

结论：

> union SSL 的主要增益并不是“让模型更多依赖 spec 支路”，  
> 而是**稳定 Mamba 的状态动力学，并增强 invariant path**。

这个认识非常关键，因为它意味着：

- 我们过去对“spec 分支/门控”的关注可能过重；
- 当前更值得深入的是：
  - **前端域去偏**
  - **invariant 主路稳定性**
  - **few-shot 适配效率**

---

## 十一、第七部分：当前最好结果与其意义

当前 canonical 任务 `FD001→FD003` 的最好结果不是纯 SPD，而是：

> **source train ∪ target train union SSL 预训练 + no-spec adaptation**

结果（3 seeds）：

- **19.8483 ± 1.1460**

对比：

- baseline no-spec：`20.6220 ± 1.1758`
- previous SPD best：`20.0151`

## 阶段意义

这是当前项目最强的单任务结果，但它的意义不仅是“数值更低”，更重要的是：

1. 它表明**跨域 SSL 确实是有用的**
2. 它进一步支持：
   - 增益主要来自 invariant backbone 强化
   - 而不是 spec path 激活
3. 它把当前研究重心从“继续强化 spec 分支”转向：
   - **更强的 invariant 表征学习**
   - **更早的域偏移抑制**
   - **few-shot 适配机制**

---

## 十二、第八部分：Task-Embedding MAML 对齐实验给出的新结论

后期为了回答一个关键问题：

> “我们当前方法为什么看起来比某些 5-shot 文献强，但和某些 1-shot meta-learning 文献又对不上？”

我们做了 paper-aligned probe。

## 12.1 对齐设置

- `K = 1`
- `window_size = 30`
- `sensor_subset = paper14`
- `normalization_mode = minmax`

## 12.2 结果

| 方法 | `FD001→FD003` |
|---|---:|
| Task-Embedding MAML paper | **24.34** |
| CD-MambAtt v3 strict probe | **46.30** |
| CD-MambAtt v3 stable-val probe | **46.37** |
| plain 1-shot full finetune baseline | **56.32** |

## 12.3 结论

这项实验非常重要，因为它回答了一个方向性问题：

> **我们当前方法在 5-shot DA 场景下很强，但在 1-shot meta-learning 场景下明显不够强。**

并且原因大概率不是 source encoder 不够强，因为：

- source RMSE 已经约 **12.8**

因此真正的差距更可能在于：

- 我们当前缺少**适合 K=1 的任务级快速适配机制**
- 也说明 Task-Embedding MAML 这类方法中的：
  - task embeddings
  - episode-based adaptation
  - low-dimensional task-specific update
  
在 `1-shot` 场景下确实可能是关键。

---

## 十三、第九部分：已经验证有效与无效的方向

## 13.1 当前已验证相对有效的方向

1. **MMD + pseudo + monotonic**
   - 当前最稳定、最广泛成立的跨域主线
2. **spec-domain-predictive**
   - SPD 体系中最有效的辅助损失
3. **frontend alignment**
   - 符合机制诊断，且在实验中有稳定正收益
4. **合适的 adaptation LR + cosine**
   - 在 canonical 任务上提升明显
5. **union SSL**
   - 当前 canonical best 来源

## 13.2 当前已验证效果弱或优先级较低的方向

1. **SSDA / hidden-state alignment**
   - 迅速饱和，对超参数不敏感
2. **inv/spec 正交约束**
   - 效果弱，甚至可能有害
3. **decoder-priority**
   - 完整训练协议下没有兑现收益
4. **hard semantic-SPD**
   - 过于激进，数值稳定性不足
5. **简单增大外部容量**
   - 未能解决核心问题

---

## 十四、第十部分：容量实验的结论

为了回答“是不是模型太小所以性能卡住”的问题，我们做过受控容量实验。

结果示例：

| 配置 | seed42 test RMSE |
|---|---:|
| `d21 / m1 / t3 / ff84` | **14.7311** |
| `d42 / m1 / t3 / ff168` | 15.8325 |
| `d42 / m2 / t4 / ff168` | 15.8250 |
| `d84 / m2 / t4 / ff336` | 28.3166 |

## 结论

> **简单放大外部模型宽度，不是当前问题的有效解。**

这说明当前更大的瓶颈在：

- 协议差异
- 迁移机制
- 前端域偏移
- few-shot 适配方式

而不是“模型太小”本身。

---

## 十五、当前可以明确对外汇报的结论

## 15.1 可以明确说的

1. **MambAtt 监督骨干已经完成结构复现**
2. **原论文 same-subset 自监督流程已经完成实现**
3. **跨域 few-shot CD-MambAtt 主线已经成立，并在多个任务上明显优于 direct transfer**
4. **SPD / DD-SSM 已完成实装，并在 canonical 任务上带来了真实收益**
5. **当前 canonical 最好结果达到 `19.8483`**
6. **机制分析已经表明：当前主矛盾更偏向“前端域偏移 + invariant path 稳定性”，而非单纯 spec 分支解耦**

## 15.2 不应过度说的

1. 不能说已经**精确复现原论文监督数字**
2. 不能说已经**严格公平地超过 FOMLN 论文**
3. 不能说 SPD 已经形成了**跨任务稳定成立的统一结构创新**
4. 不能说 union SSL 最好结果已经在多任务上全部验证完毕

---

## 十六、当前项目面临的核心问题

结合所有实验结果，当前项目的困难主要有三个：

## 16.1 canonical 任务已经做得比较强，但难任务仍然没打穿

- `FD001→FD003` 已经推进到 `19.85`
- 但 `FD003→FD001` 仍未解决

这意味着：

> 现在最需要的不是继续在 easiest task 上微调，而是找出为什么某些 transfer direction 仍然失败。

## 16.2 SPD 的“创新叙事”还不够硬

虽然 SPD 已经带来收益，但当前证据链中仍有薄弱环节：

- gate 变化太小
- inv/spec 分支功能差异不够大
- 跨任务稳定性不足

因此如果未来要写成论文，当前版本的叙事仍需要更扎实的机制支持。

## 16.3 当前最好结果更依赖 invariant 主路，而不是 specific 分支

这意味着：

- 过去很多围绕 spec 分支展开的结构想法，可能并不是当前最有信息增益的方向；
- 后续更值得投入的是：
  - 更早的前端去域偏；
  - 更强的 invariant 表征；
  - 更适合 few-shot 的 task-level adaptation 机制。

---

## 十七、下一阶段建议

结合当前结果，我建议下一阶段按以下优先级推进：

## 17.1 第一优先级：围绕 `FD003→FD001` 做关键诊断

原因：

- 这是 SPD 当前最明显的失败方向；
- 如果这个方向迟迟打不通，SPD 很难作为“稳定创新线”成立。

## 17.2 第二优先级：继续强化“前端域偏移”视角

重点不应再只是深挖 `Δ/B/C` 解耦，而应更多关注：

- 更早位置的域对齐；
- shared frontend 的 domain bias；
- 早期表征的 invariant 化。

## 17.3 第三优先级：验证 union SSL 的跨任务稳定性

当前 union SSL + no-spec 只在 `FD001→FD003` 上证据最强，后续应补：

- `FD003→FD001`
- 至少再补 1~2 个任务
- 再补 5-seed 验证

## 17.4 第四优先级：如果继续跟 Task-Embedding MAML 对比，应转向真正的 meta-style adaptation

因为 paper-aligned probe 已经说明：

- 我们当前 DA-style 方法不适合 `K=1`
- 继续调常规超参意义不大

如果要继续这条线，应该考虑：

- episode-based meta-learning
- low-dimensional task adaptation
- task embedding / adapter 风格的 few-shot 机制

## 17.5 第五优先级：项目进一步收束

当前项目已经积累了较多脚本、结果和诊断。

下一阶段非常重要的一件事是：

> **把“哪些方向已经基本证伪、哪些方向是当前主线、哪些方向是下一步关键问题”进一步收束清楚。**

---

## 十八、总结性判断

如果把整个阶段的结论压缩成一句话，我认为最准确的表述是：

> **本项目已经完成了 MambAtt 的监督与自监督基础复现，并成功建立了一条在多任务上有效的跨域 few-shot RUL 主线；在此基础上，我们实现了 Mamba 内部的 SPD/DD-SSM 创新并完成了系统机制诊断。当前最强结果达到 `FD001→FD003 = 19.85`，但进一步研究表明，当前真正的主问题并不只是 SSM 内部 drift，而更偏向共享前端域偏移与 invariant path 稳定性。因此，项目已经从“方法是否成立”阶段进入到了“如何让创新更稳定、更有机制说服力”阶段。**

---

## 十九、建议汇报时优先展示的内容

如果用于导师汇报，建议优先展示以下 5 组内容：

1. **项目总体目标与三条主线**
   - 监督复现
   - 自监督复现
   - 跨域创新

2. **跨域主线 v2 的多任务结果**
   - 证明方法已经成立

3. **SPD / frontend alignment / union SSL 的结果演进**
   - 展示创新确实带来了 canonical 增益

4. **机制诊断结论**
   - 59x drift
   - 前端域偏移
   - spec 分支过弱
   - union SSL 强化 invariant path

5. **当前问题与下一步计划**
   - `FD003→FD001`
   - 前端去域偏
   - `K=1` 任务级快速适配

---

## 二十、关联文档

建议配合本报告一起查看的主文档：

- [`project_status.md`](./project_status.md)
- [`project_overview_zh_2026-04-08.md`](./project_overview_zh_2026-04-08.md)
- [`supervised_reproduction.md`](./supervised_reproduction.md)
- [`self_supervised_reproduction.md`](./self_supervised_reproduction.md)
- [`published_baselines.md`](./published_baselines.md)
- [`cross_domain_experiment_log.md`](./cross_domain_experiment_log.md)
- [`history/experiment_notes/task_embedding_maml_paper_alignment_probe_2026-04-09.md`](./history/experiment_notes/task_embedding_maml_paper_alignment_probe_2026-04-09.md)

