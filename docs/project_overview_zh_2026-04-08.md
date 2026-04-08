# CD-MambAtt 项目中文总览

最后更新：`2026-04-08`

---

## 1. 这个项目到底在做什么

一句话概括：

> 在 `MambAtt` 骨干上做 **跨域 few-shot RUL 预测**，目标是在一套 C-MAPSS 子数据集上学到退化规律，再迁移到另一套只有少量标注样本的新子数据集上。

项目现在有三条主线：

1. **监督复现线**
   - 复现原始 `MambAtt` 论文的单域监督结果，确认骨干是否可靠。
2. **自监督复现线**
   - 复现论文里的同子集 self-supervised 训练流程，确认我们对原论文的理解和实现是否完整。
3. **跨域创新线**
   - 在 `MambAtt` 基础上做跨域 few-shot RUL。
   - 当前已经做出一条稳定可用的 `CD-MambAtt v2` 主线。
   - 之后又进一步尝试了 `SPD / DD-SSM` 这条 Mamba 内部创新线。

当前最准确的项目定位不是“还在搭环境”，而是：

> **基础复现已经完成，跨域方法已经成立，但更进一步的内部创新目前卡在局部最优。**

---

## 2. 当前代码主干怎么理解

### 2.1 核心代码目录

主包在：

- `cd_mambatt/`

其中最重要的文件是：

| 文件 | 作用 |
|---|---|
| `cd_mambatt/models/mambatt.py` | 主模型定义，包含原始 `MambAttRegressor` 以及对 SPD 的支持 |
| `cd_mambatt/models/dd_mamba.py` | `DDMambaBlock`，也就是 SPD / DD-SSM 的核心实现 |
| `cd_mambatt/data.py` | C-MAPSS 读取、滑窗、数据集封装 |
| `cd_mambatt/pseudo_labeling.py` | 伪标签和阶段统计 |
| `cd_mambatt/self_supervised.py` | 论文自监督三种训练任务的实现 |
| `cd_mambatt/losses/mmd.py` | 全局 MMD、conditional MMD |
| `cd_mambatt/losses/monotonic.py` | 单调性损失 |
| `cd_mambatt/losses/domain_adversarial.py` | GRL 域对抗模块 |

### 2.2 主要训练入口

当前真正有用的训练脚本：

| 文件 | 用途 |
|---|---|
| `train_supervised.py` | 单域监督训练，主要用于论文监督复现和骨干体检 |
| `train_self_supervised.py` | 论文 same-subset 自监督复现 |
| `train_cross_domain_baseline.py` | direct transfer / few-shot fine-tune 基线 |
| `train_cd_mambatt_v2.py` | 稳定的跨域主线：`MMD + pseudo + monotonic` |
| `train_cd_mambatt_v3.py` | SPD / DD-SSM 以及后续机制实验主入口 |
| `train_fomln_baseline.py` | 复现 `FOMLN` published baseline |

### 2.3 哪些脚本是“实验脚手架”

项目中的独立实验脚本现在统一收在：

- `experiments/`

并进一步分成四类：

- `experiments/ablations/`：相对正式的 3-seed / 多任务验证
- `experiments/diagnostics/`：gate、hidden state、drift 等机制探针
- `experiments/prototypes/`：替代架构原型
- `experiments/quick_tests/`：快速 A/B 和临时验证脚本

例如：

- `experiments/ablations/spd_highlr_frontend_3seed.py`
- `experiments/ablations/spd_highlr_3seed.py`
- `experiments/ablations/spd_best_fd001_fd004.py`
- `experiments/prototypes/dualpath_3seed.py`
- `experiments/prototypes/multi_idea_test.py`
- `experiments/diagnostics/ssda_probe.py`
- `experiments/quick_tests/tmp_dual_state_quick_ab.py`

这些脚本的作用是：

- 快速验证某个想法；
- 在完整训练框架之外做定向小实验；
- 为后续是否值得并入主线提供证据。

它们不是项目最干净的主干，但保留价值很高，因为很多关键结论就是从这些小脚本跑出来的。

---

## 3. 目前已经做成了什么

### 3.1 监督复现：结构上对齐，数值上仍有差距

论文监督参考：

- `FD001` Table 3: `11.46`
- Transformer-decoder 平均：`12.34`

当前本地最好监督线：

- 配置：`post + d_ff=84 + val_all_windows + lr=5e-4 + wd=1e-4`
- `42,43,44` 固定 split 平均：
  - `15.0937`

当前判断：

- 监督骨干已经**结构复现完成**；
- 但和论文数字之间仍然有明显 gap；
- 这个 gap 更像是隐藏训练协议或实现细节，不像缺了某个大模块。

### 3.2 自监督复现：实现完成，但当前暂停

论文里三种 self-supervised 任务已经实现，主代码在：

- `cd_mambatt/self_supervised.py`
- `train_self_supervised.py`

最关键结果：

- `FD001` one-shot SSL 已经基本贴近论文：
  - 论文：`31.7270`
  - 当前：`31.6116 ± 2.5595`

但其他子集没有全部贴齐，所以当前状态是：

> **这条线已经能证明“实现是通的”，但不是当前优先级。**

### 3.3 跨域主线 v2：已经证明“方向成立”

当前最稳定、最可信的跨域方法仍然是：

> **`MMD + pseudo-labeling + monotonic`**

`5-shot` 多任务结果：

| 任务 | Direct transfer | CD-MambAtt v2 |
|---|---:|---:|
| `FD001 -> FD003` | 34.8204 | **21.9608** |
| `FD003 -> FD001` | 24.6059 | **19.8137** |
| `FD002 -> FD004` | 29.5028 | **22.1776** |
| `FD001 -> FD004` | 32.3820 | **24.3449** |
| `FD004 -> FD002` | 21.3666 | **19.5311** |
| `FD001 -> FD002` | 31.1029 | **22.9457** |

这说明：

- 我们的跨域方法不是只在单一任务上偶然有效；
- 这条路线已经是一个真正成立的方法；
- 当前问题不在“有没有方法”，而在“怎么继续把它做得更好、更有创新性”。

### 3.4 和 published baseline 的关系

我们本地还复现了 `FOMLN`。

在当前的 `15-shot` matched-shot in-house 对比里，`CD-MambAtt` 在 4 个重叠任务上都赢了 `FOMLN`。

但这件事要非常谨慎地表述：

- 这不是完全 publication-grade apples-to-apples；
- 因为预处理、目标域信息利用路径、validation 规则等还没完全统一。

所以更稳妥的结论是：

> **当前本地对比里，CD-MambAtt 的表现优于复现的 FOMLN，但严格公平性表述需要保守。**

---

## 4. SPD / DD-SSM 这条创新线做到哪了

### 4.1 它想解决什么问题

SPD 的想法是：

> 不只在最终 feature 上做对齐，而是在 Mamba 内部生成选择性参数 `dt / B / C` 的地方，把域不变信息和域特定信息拆开。

核心文件是：

- `cd_mambatt/models/dd_mamba.py`

### 4.2 它拿到了什么结果

早期在 canonical 任务 `FD001 -> FD003` 上，SPD 确实出现过正收益：

- SPD old best:
  - `20.37 ± 0.47`
- 对 v2 参考 `21.13 ± 1.59` 有提升

之后沿着“更早对齐”这条线继续推进，当前 canonical 最好结果已经到：

- `FD001 -> FD003 = 20.0151 ± 0.7523`

对应思路是：

- 保留 SPD 主体；
- 把 invariant 对齐位置前移到 `frontend_features`；
- 配合更合适的 adaptation LR 和 cosine。

### 4.3 但它为什么没有成为统一答案

虽然 canonical 任务进一步提升到了 `20.0151`，但跨任务看问题就暴露出来了：

- `FD001 -> FD004`：只有轻微改善；
- `FD003 -> FD001`：仍然没有打穿，还是落后于 v2 最好结果。

所以当前更准确的结论是：

> **SPD + frontend alignment 是真实有效的方向，但不是一个对所有任务统一奏效的机制答案。**

---

## 5. 我们已经明确知道哪些方向有效

### 5.1 已经验证有效的

1. `MMD + pseudo + monotonic`
   - 这是当前最稳的跨域主线。

2. `spec-domain-predictive`
   - 在 SPD 体系里，这是目前最有用的辅助监督。

3. `frontend alignment`
   - 特别是在 `FD001 -> FD003` 上，前端早对齐是明显有帮助的。

4. `更合适的优化强度`
   - 例如高 LR + cosine 曾经带来很大单次改善。

### 5.2 已经明显负面或至少不值得优先继续堆的

1. `contrastive`
   - 多轮实验都显示它要么没收益，要么变差。

2. `SSDA / hidden-state alignment`
   - 很快饱和，对权重不敏感。

3. `decoder-priority / 先只训 transformer+head`
   - 在完整协议下没有兑现收益。

4. `hard semantic-SPD`
   - 早期硬拆 prediction 的版本数值明显更差。

5. `conditional alignment / proto-ce` 现有版本
   - 在难任务上没有形成稳定正收益。

---

## 6. 当前最关键的机制认识

这是整个项目现在最重要的部分。

来自 `mamba_mechanism_diagnosis_2026-04-06.md` 的结论可以压缩成四句话：

1. **域偏移非常早就进入了共享前端**
   - `conv1d` 后的特征几乎可以被完美分域。

2. **SSM 递推会放大 drift，但不是偏移的唯一来源**
   - 59x drift 确实存在；
   - 但根子不只是 scan 内部，而是前端已经带偏了。

3. **few-shot 监督主要打在 frontend / transformer / head**
   - 不是优先打在 SPD 的 spec 支路上。

4. **spec 分支现在并没有真正承担主要适配职责**
   - gate 变化很小；
   - 拿掉 spec，RMSE 几乎不变。

这四点合在一起，得到的核心判断是：

> **我们之前把太多精力放在 Mamba 内部 `dt/B/C` 解耦上，但当前真正的主矛盾更像是“共享前端域偏移 + 后端 few-shot 校准”，而不是 spec 支路本身。**

---

## 7. 为什么现在会觉得“项目卡住了”

当前项目的困境不是“完全没结果”，而是：

> **主线已经成立，但新创新迟迟没有找到真正的突破点。**

更具体地说，有三个“卡点”。

### 7.1 Canonical 任务还能磨，但难任务过不去

`FD001 -> FD003` 已经被推到了 `20.0151`，说明这条线还能做出亮点。

但：

- `FD003 -> FD001` 仍然没有解决；
- 它现在是检验方法是否真的跨任务成立的关键难点。

### 7.2 很多新结构只是“更复杂”，不是“更有效”

过去这段时间试过很多内部创新：

- dual-state
- window gating
- semantic-SPD
- conditional alignment
- proto-ce
- decoder-priority

结论大多是：

- 有少量正信号；
- 但没有形成稳定、跨任务、可复现的大改进。

### 7.3 直接加外部模型容量没有解决问题

这是今天刚补的一个重要结论。

在 `FD001` 单域监督上，我做了一个受控容量试探：

| 配置 | seed42 test RMSE | 结论 |
|---|---:|---|
| `d21 / m1 / t3 / ff84` | **14.7311** | 基线最好 |
| `d42 / m1 / t3 / ff168` | 15.8325 | 变差 |
| `d42 / m2 / t4 / ff168` | 15.8250 | 变差 |
| `d84 / m2 / t4 / ff336` | 28.3166 | 发散 |

对应结果文件：

- `runs/supervised_capacity_sweep/baseline_d21_m1_t3_ff84/FD001/summary.json`
- `runs/supervised_capacity_sweep/mid_d42_m1_t3_ff168/FD001/summary.json`
- `runs/supervised_capacity_sweep/mid_d42_m2_t4_ff168/FD001/summary.json`
- `runs/supervised_capacity_sweep/large_d84_m2_t4_ff336/FD001/summary.json`

这个结果说明：

> **“外部宽度直接做大”不是当前监督 gap 的简单答案。**

换句话说，问题不是“模型太小，所以一加宽就会更强”。

---

## 8. 当前最合理的下一步是什么

结合所有结果，我认为当前最合理的优先级是：

### 8.1 第一优先级：把 `FD003 -> FD001` 作为关键诊断任务

原因：

- 这是 SPD 当前最明显的失败点；
- 如果这个任务解决不了，SPD 很难作为稳定创新线成立。

### 8.2 第二优先级：重点看“前端域偏移”，不是继续深挖 `dt/B/C`

因为机制诊断已经说明：

- 域偏移很早进入前端；
- spec 分支太弱；
- 再在 `dt/B/C` 上做很复杂的结构修改，信息增益越来越低。

### 8.3 第三优先级：如果还试容量，只试“内部容量”

今天已经说明：

- 外部 `d_model` 加宽不是有效突破口。

如果还想继续做容量实验，更值得测的是：

- `d_model=21` 不变；
- 只增大内部 `d_state`、`expand`；
- 或者谨慎增加 Mamba 层数，但不改外部宽度。

### 8.4 第四优先级：项目整理和收束

当前项目已经积累了大量实验脚本和日志。

接下来很重要的一件事不是再乱试，而是：

- 明确主干文件；
- 明确哪些实验方向已经基本证伪；
- 把当前最可信的故事整理成一条清晰主线。

---

## 9. 如果你现在想快速接手，建议按这个顺序读

### 第一组：先建立全局认识

1. `docs/project_overview_zh_2026-04-08.md`  
   - 就是这份文档。
2. `docs/project_status.md`
   - 英文版全局状态快照。
3. `docs/README.md`
   - 文档总入口。

### 第二组：看最重要的实验与结论

4. `docs/cross_domain_experiment_log.md`
   - 这是最完整的实验档案。
5. `docs/mamba_mechanism_diagnosis_2026-04-06.md`
   - 当前所有机制判断的核心依据。
6. `docs/spd_highlr_frontend_2026-04-08.md`
   - 当前 canonical 最好结果的来龙去脉。
7. `docs/spd_highlr_frontend_cross_task_2026-04-08.md`
   - 为什么这条线不是统一答案。

### 第三组：看代码入口

8. `train_supervised.py`
9. `train_cd_mambatt_v2.py`
10. `train_cd_mambatt_v3.py`
11. `cd_mambatt/models/mambatt.py`
12. `cd_mambatt/models/dd_mamba.py`

---

## 10. 当前一句话结论

如果只保留一句话，那就是：

> **项目已经证明“跨域 CD-MambAtt 这条路是成立的”，但 SPD 这条 Mamba 内部创新线还没有找到真正稳定、跨任务有效的突破点；当前最该做的是围绕前端域偏移和关键难任务做更聚焦的验证，而不是继续无边界扩展想法。**
