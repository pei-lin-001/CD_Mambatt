# CD-MambAtt 技术细节附录（供导师汇报 / 代码对照）

最后更新：`2026-04-11`

> 本文档是 [`stage_report_for_advisor_zh_2026-04-11.md`](./stage_report_for_advisor_zh_2026-04-11.md) 的**技术附录**。
> 
> 如果阶段总结报告回答的是“我们做了什么、做到哪里了、效果如何”，那么本文档回答的是：
> 
> - 当前仓库里**到底有哪些核心代码路径**；
> - 每条训练线的**数据流、模型流、损失流**分别是什么；
> - MambAtt、跨域 v2、SPD / DD-Mamba、自监督三条线在工程上**具体怎么实现**；
> - 到目前为止，哪些结构判断已经被实验支持，哪些只是候选推测。

相关主文档：

- 阶段汇报：[`stage_report_for_advisor_zh_2026-04-11.md`](./stage_report_for_advisor_zh_2026-04-11.md)
- 项目状态：[`project_status.md`](./project_status.md)
- 总览文档：[`project_overview_zh_2026-04-08.md`](./project_overview_zh_2026-04-08.md)
- 权威实验日志：[`cross_domain_experiment_log.md`](./cross_domain_experiment_log.md)
- 机制诊断：[`mamba_mechanism_diagnosis_2026-04-06.md`](./mamba_mechanism_diagnosis_2026-04-06.md)

---

## 1. 仓库技术结构总览

当前项目主仓库为 `cd_mambatt/`，主要技术文件可以按“数据、模型、训练入口、损失、实验脚本、文档”六部分理解。

### 1.1 训练入口脚本

| 文件 | 作用 | 当前定位 |
|---|---|---|
| `train_supervised.py` | 单域监督训练 | 原始 MambAtt 监督复现 |
| `train_self_supervised.py` | 同子集自监督预训练 + 下游 one-shot 评估 | 原论文自监督复现 |
| `train_cross_domain_baseline.py` | direct transfer / few-shot finetune 等基础跨域基线 | 最基础跨域对照 |
| `train_cd_mambatt_v2.py` | MMD + pseudo + monotonic 的稳定跨域主线 | 当前最稳 few-shot 主线 |
| `train_cd_mambatt_v3.py` | SPD / DD-SSM / 机制实验主入口 | 当前结构创新实验主入口 |
| `train_fomln_baseline.py` | 复现 FOMLN 风格对照 | 已实现的外部 baseline |

### 1.2 核心包目录

| 路径 | 作用 |
|---|---|
| `cd_mambatt/data.py` | C-MAPSS 读取、按发动机划分、标准化、滑窗、monotonic pairs、sensor subset |
| `cd_mambatt/cross_domain.py` | 任务注册、few-shot target 划分 |
| `cd_mambatt/models/mambatt.py` | 主骨干 `MambAttRegressor`、Mamba/Transformer block |
| `cd_mambatt/models/dd_mamba.py` | SPD / DD-Mamba 核心实现 |
| `cd_mambatt/self_supervised.py` | temporal / N-tuplet / pseudo 三种自监督任务 |
| `cd_mambatt/losses/mmd.py` | global MMD + conditional MMD |
| `cd_mambatt/losses/monotonic.py` | 局部单调性排序损失 |
| `cd_mambatt/losses/domain_adversarial.py` | GRL + 域判别器 |
| `cd_mambatt/pseudo_labeling.py` | 伪标签阶段统计与分配 |

### 1.3 工程组织状态

近期已经做过一次清理：

- 反复试验脚本已收拢到 `experiments/`
- 主文档集中在 `docs/`
- 历史记录移动到 `docs/history/`
- 当前推荐先看 `docs/README.md` 再跳到主报告或实验日志

因此，代码库现在已经具备：

1. 可重复的训练入口；
2. 可追溯的实验日志；
3. 可用于导师汇报的系统性文档；
4. 可继续扩展的结构创新支撑位。

---

## 2. 数据处理技术细节

数据相关逻辑主要集中在 `cd_mambatt/data.py` 和 `cd_mambatt/cross_domain.py`。

### 2.1 支持的数据子集与基础对象

当前代码默认围绕 NASA C-MAPSS 四个子集：

- `FD001`
- `FD002`
- `FD003`
- `FD004`

其中基础数据对象是 `CMAPSSSplit`，包含：

- `subset`
- `split`（train / test）
- `unit_ids`
- `cycles`
- `operating_settings`
- `sensors`
- `rul`
- `condition_ids`（条件归一化时使用）

这意味着项目里**所有后续划分与变换**，都不是直接操作原始 txt，而是操作 `CMAPSSSplit`。

### 2.2 RUL 计算与截断

`load_cmapss_split(...)` 负责从原始 `train_FDxxx.txt / test_FDxxx.txt / RUL_FDxxx.txt` 中加载矩阵并构建 `CMAPSSSplit`。

关键处理：

- 训练集 RUL：按每台发动机最大 cycle 反推；
- 测试集 RUL：结合 `RUL_FDxxx.txt` 计算总寿命后再反推；
- 默认使用 `rul_clip=125`，即分段线性截断。

这与大多数 C-MAPSS RUL 工作的标准设定一致。

### 2.3 归一化策略

`fit_normalizer(...)` 当前支持两种模式：

- `zscore`
- `minmax`

其中：

- `FD001` / `FD003` 默认做**全局归一化**；
- `FD002` / `FD004` 因为存在多工况，代码中使用 `CONDITION_NORMALIZED_SUBSETS = {"FD002", "FD004"}`，会先按 operating settings 聚成 condition，再做**condition-wise 归一化**。

技术上，`CMAPSSNormalizer` 同时保存：

- 全局 `mean/std/min/max`；
- 条件级 `condition_mean/std/min/max`；
- `condition_key_to_id` 与 `condition_centers`。

这一步非常重要，因为多工况子集如果不做条件化标准化，跨工况差异会污染故障退化信息。

### 2.4 传感器子集：paper14

为了和后期参考的 task-embedding / meta-learning 类论文做更接近的 protocol 对齐，代码新增了：

- `sensor_subset = paper14`

其 1-based 传感器编号为：

`2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21`

在代码内部转换成 0-based 索引：

- `PAPER_14_SENSOR_INDICES`

相关函数：

- `resolve_sensor_subset_preset(...)`
- `select_sensor_subset(...)`

这使得项目现在可以在：

- 原始 21 sensors 协议
- `paper14` 协议

之间切换。

### 2.5 滑动窗口构造

`build_windows(...)` 是整个项目里最核心的数据变换之一。

输入：一个 `CMAPSSSplit`
输出：`WindowedData`

输出字段：

- `windows`: `(num_windows, window_size, feature_dim)`
- `targets`
- `unit_ids`
- `end_cycles`

关键实现细节：

1. 默认 `stride=1`
2. 可选 `last_only=True/False`
3. 若序列短于窗口长度，默认 `pad_short=True`
4. 每个 window 的目标值取该 window **最后一个时间步**对应的 RUL

在项目中：

- 训练阶段通常用 `last_only=False`
- 最终测试阶段通常用 `last_only=True`
- source/target validation 是否也用 all windows，可由脚本参数控制

### 2.6 单调性窗口对构造

`build_monotonic_window_pairs(...)` 用于 monotonic loss。

它会在同一台发动机内部，按 `end_cycle` 排序，然后取：

- 更早 window
- 更晚 window

形成 pair。

参数：

- `pair_gap`
- `pair_stride`

输出：`WindowPairData`

后续 monotonic 损失要求：

> later window 预测的 RUL 不应比 earlier window 更大。

对应实现见：

- `cd_mambatt/losses/monotonic.py`

### 2.7 Source / Target 的单位级划分

### Source 侧

`get_train_validation_unit_ids(...)` 和 `split_train_validation_by_unit(...)` 负责 source train/val 的发动机级划分。

默认：

- `source_train_ratio = 0.8`
- 按发动机单位划分，不是按 window 随机打散

### Target 侧 few-shot

`cd_mambatt/cross_domain.py` 中的 `build_few_shot_target_partition(...)` 负责把 target train subset 划成三部分：

- `labeled_units`
- `validation_units`
- `unlabeled_units`

默认 v3 协议：

- `target_shots = 5`
- `target_val_units = 10`
- `resample_few_shot_per_seed = True`

这代表：

- 每个 seed 都会重新抽 few-shot 目标发动机；
- 统计更接近“真实 few-shot 不确定性”，但方差也更真实。

### 2.8 当前协议公平性要点

代码层面目前已经支持把以下 protocol 显式写进结果：

- `window_size`
- `sensor_subset`
- `normalization_mode`
- `target_shots`
- `target_val_units`
- 是否按 seed 重采样 source split / target partition
- source/target validation 是否用 all windows

这点很重要，因为项目后期已经明确意识到：

> 同一个数据集 ≠ 同一个实验协议。

很多文献对比差异，最后都来自这些“隐藏 protocol”。

---

## 3. 基础 MambAtt 骨干的技术实现

主模型定义在：`cd_mambatt/models/mambatt.py`

### 3.1 标准 `MambAttRegressor` 的前向流程

可以概括成：

`x -> input_proj -> Mamba blocks -> positional encoding -> Transformer decoder -> last token -> dropout -> head`

即：

1. 输入 `x` 形状为 `(B, L, C)`
2. 先过 `input_proj`
3. 再串联若干 `Mamba` block
4. 再加位置编码
5. 再过 Transformer block / encoder
6. 取最后一个 token 表征
7. 回归头输出 RUL

### 3.2 默认结构参数

`MambAttRegressor` 当前默认值：

- `input_dim = 21`
- `d_model = 21`
- `d_state = 16`
- `d_conv = 8`
- `expand = 2`
- `num_mamba_layers = 1`
- `num_transformer_layers = 3`
- `num_heads = 7`
- `dropout = 0.5`
- `dim_feedforward = 2048`（但复现实验中 `84` 更好）

需要注意：

- 论文/仓库默认 FFN 很大，但本地监督复现实验发现 `d_ff=84` 更优；
- 当前很多跨域主线实验也沿用了这个经验。

### 3.3 Mamba block 的三种模式

当前 `mamba_block_mode` 支持：

- `bare`
- `prenorm_residual`
- `dd_spd`

含义：

1. `bare`：直接用标准 `Mamba`
2. `prenorm_residual`：外包一层 `LayerNorm + residual`
3. `dd_spd`：切换到自定义 `DDMambaBlock`

这也是 v3 能在不重写整个骨干的前提下直接替换 Mamba 内部结构的原因。

### 3.4 Transformer 实现与归一化顺序

当前支持两种 Transformer 实现：

- `custom`
- `torch`

其中默认更常用的是 `custom`，因为它允许进一步插入：

- 域适配残差 / shift / FiLM

此外，`transformer_norm_mode` 支持：

- `pre`
- `post`

监督复现实验中曾验证：

- 某些场景 `post` 更好；
- 但跨域主线并不一定跟监督复现最优配置完全一致。

### 3.5 自定义 Transformer block 的域适配接口

`TransformerBlock` 中已经实现：

- `domain_adapter_mode = none`
- `domain_adapter_mode = target_shift`
- `domain_adapter_mode = target_film`

实现方式是在 LayerNorm 后对 target 样本做：

- 仅加偏移（shift）
- 或做 `(1 + gamma) * x + beta` 的 FiLM 式调制

这是后来为了探索“即使 SPD 工作了，Transformer 共享层是否仍会重新引入域偏移”而加入的轻量结构位。

### 3.6 输出头与 SPD 预测器模式

当 `mamba_block_mode = dd_spd` 时，预测端支持：

- `shared_head`
- `decomposed_residual`
- `shared_aux_residual`

含义：

1. `shared_head`：仍用一个共享回归头，最稳定
2. `decomposed_residual`：显式分成 invariant 预测和 specific residual 预测
3. `shared_aux_residual`：主输出仍走共享 head，但同时训练 `inv_head/spec_head` 做辅助约束

当前结论是：

- 完全解耦 predictor 容易过于扰动原有稳定训练路径；
- `shared_head` 或 `shared_aux_residual` 更稳。

### 3.7 CUDA 约束

`MambAttRegressor._forward_mamba_sequence(...)` 中明确要求：

> 当前安装的 Mamba kernel 仅支持 GPU 路径。

因此项目实际训练默认依赖 CUDA，这也是为什么仓库工作流主要在 WSL2 + GPU 下维护。

---

## 4. 原论文 self-supervised 路线的技术实现

自监督相关实现集中在：

- `cd_mambatt/self_supervised.py`
- `train_self_supervised.py`

### 4.1 这条线的定位

这条线复现的是：

> 原 MambAtt 论文中的“同一子集内自监督预训练 + 下游 one-shot 预测”

不是跨域 few-shot 方案本身。

所以它当前的意义主要有两个：

1. 验证我们对原论文训练机制的理解是否完整；
2. 为后续“跨域前是否要做 union SSL / source-only SSL”提供可复用模块。

### 4.2 三个自监督任务

### A. Temporal ordering

核心函数：

- `build_temporal_triplet_indices(...)`
- `forward_temporal_logits(...)`

实现思想：

- 从同一发动机的退化序列中采样 3 个 window
- 有序 triplet 标 1，无序 triplet 标 0
- 用 `BCEWithLogitsLoss` 训练 temporal classifier

代码里显式定义了：

- 多组 ordered pattern
- 多组 disordered pattern

而不是只使用单一模板。

### B. Consecutive pair N-tuplet

核心函数：

- `build_consecutive_pair_indices(...)`
- `ntuplet_loss(...)`

实现思想：

- 将同一发动机中相邻窗口视为 query / positive
- batch 内其它 pair 形成负样本
- 采用论文式 inner-product/logsumexp 风格损失

当前 `ntuplet_mode` 支持：

- `flatten`
- `last`
- `mean`

并可选择是否 `L2 normalize`。

### C. Pseudo sensor label

核心函数：

- `fit_sensor_pseudo_label_statistics(...)`
- `assign_window_sensor_pseudo_labels(...)`
- `forward_pseudo_logits(...)`

实现思想：

1. 丢弃部分传感器（`PSEUDO_LABEL_SENSOR_DROP`）
2. 对剩余每个 sensor 的归一化数值做 1D `k=3` 聚类
3. 得到 `b1/b2` 阈值
4. 把每个 window 的某个 sensor 值映射成 3 类伪标签
5. 训练 pseudo classifier 预测这些标签

这对应论文里“用传感器统计状态做自监督伪标签”的那一路。

### 4.3 SSL 训练器结构

`MambAttSelfSupervisedPretrainer` 内部是：

- 一个 `MambAttRegressor` backbone
- temporal classifier
- pseudo classifier
- N-tuplet embedding path

其中：

- temporal 头输入为 `3 * window_size * d_model`
- pseudo 头按每个 sensor 的 window trace 做三分类

### 4.4 重要约束：当前复现要求 `d_model == input_dim`

`train_self_supervised.py` 中有一个关键限制：

> 当前 same-subset SSL reproduction 期望 `d_model == input_dim`

原因是：

- pseudo-label loss 是直接定义在原始 sensor 维上的；
- 如果 `d_model != input_dim`，则 pseudo branch 的语义不再与原 sensor 一一对应。

这也是为什么当前 SSL 复现更像是“对原论文路径的工程再现”，而不是完全通用的预训练框架。

### 4.5 SSL 预设权重

`train_self_supervised.py` 已支持论文风格 preset：

- `paper_full`：`lambda_temporal=0.25`, `lambda_ntuplet=0.25`, `lambda_pseudo=0.5`
- `paper_temp_ntuplet`
- `temporal_only`
- `custom`

### 4.6 下游迁移方式

SSL 训练完成后，可：

1. 导出 encoder state dict
2. 加载到普通 `MambAttRegressor`
3. 选择冻结 encoder 或继续 finetune

相关辅助函数：

- `freeze_mambatt_encoder(...)`
- `load_encoder_weights(...)`
- `export_encoder_state_dict(...)`

### 4.7 当前已知作用边界

目前实验证据表明：

- same-subset SSL 对“复现论文 one-shot 结果”是有意义的；
- 但对跨域 few-shot 来说，并不能自动转化为收益；
- 真正对当前跨域最有效的是后续做过的 **source ∪ target train union SSL**，它的收益主要体现为**稳定 invariant path 和 Mamba dynamics**，而不是让 SPD 的 spec gate 打开。

---

## 5. 跨域训练线的技术实现

### 5.1 `train_cross_domain_baseline.py`

这是最基础的跨域脚本，主要用于建立：

- 直接迁移（source train 后直接测 target）
- source 预训练后在少量 target labeled 上 finetune

该脚本已经支持：

- source split 持久化
- target few-shot partition 持久化
- sensor subset
- normalization mode

它的定位是：

> 给后续 v2 / v3 一条最基础的“没有复杂 alignment 的对照线”。

### 5.2 `train_cd_mambatt_v2.py`：当前稳定跨域主线

v2 的核心思想是：

> 用 source 监督信号维持故障回归语义；
> 用 unlabeled target 做分布对齐；
> 用 pseudo-label 和 monotonic loss 给 target 内部结构加弱监督。

### v2 主要损失组成

设 adaptation 阶段总损失为：

`L = L_source + L_target + λ_mmd L_mmd + λ_stage L_source_stage + λ_pseudo L_pseudo + λ_contrastive L_contrastive + λ_mono L_mono`

其中：

1. `L_source`：source labeled RUL 回归
2. `L_target`：target few-shot labeled RUL 回归
3. `L_mmd`：source / target feature 对齐
4. `L_source_stage`：source 上的退化阶段分类
5. `L_pseudo`：target unlabeled 伪标签阶段分类
6. `L_contrastive`：跨域 N-tuplet / contrastive，对应代码中可选项
7. `L_mono`：target 局部单调性约束

### v2 默认/经验参数

代码 parser 默认：

- `pseudo_start_quantile = 0.30`
- `pseudo_end_quantile = 0.70`

但当前稳定主线更多使用：

- `lambda_mmd = 0.1`
- `lambda_source_stage = 1.0`
- `lambda_pseudo = 0.5`
- `lambda_monotonic = 0.05`
- `lambda_contrastive = 0.0`
- pseudo quantile `0.5 -> 0.9`

这条线已经在多任务上稳定优于 direct transfer。

### 5.3 `train_cd_mambatt_v3.py`：结构创新与机制实验总入口

v3 不是一个单一方法，而是一个**实验平台**。

它在 v2 的 source/target 数据流基础上，又加入了：

- `dd_spd` block
- invariant-path alignment
- GRL / inv-MMD 切换
- conditional alignment
- spec-domain loss
- inv/spec orth / xcorr
- semantic warmup
- frontend adapter
- domain-conditioned gate
- Transformer domain adapter

### v3 关键 parser 选项

数据协议相关：

- `--sensor-subset {none,paper14}`
- `--normalization-mode {zscore,minmax}`
- `--target-shots`（默认 5）
- `--target-val-units`（默认 10）
- `--resample-few-shot-per-seed`（默认 True）
- `--source-val-all-windows` / `--target-val-all-windows`（默认 True）

模型相关：

- `--mamba-block-mode {bare,prenorm_residual,dd_spd}`
- `--spd-scan-mode {mixed,dual_state}`
- `--spd-gate-mode {token,window}`
- `--spd-gate-scheme {shared,dt_bc}`
- `--spd-predictor-mode {shared_head,decomposed_residual,shared_aux_residual}`
- `--domain-conditioned-gate`
- `--frontend-adapter-mode {none,target_affine,target_residual}`
- `--transformer-domain-adapter-mode {none,target_shift,target_film}`

对齐/正则相关：

- `--inv-alignment-mode {grl,mmd,none}`
- `--lambda-domain-adv`
- `--lambda-inv-mmd`
- `--lambda-spec-domain`
- `--lambda-conditional-inv-mmd`
- `--lambda-conditional-proto`
- `--lambda-conditional-proto-ce`
- `--lambda-inv-spec-orth`
- `--lambda-inv-spec-xcorr`
- `--lambda-spec-residual`
- `--lambda-inv-aux`
- `--lambda-spec-reconstruction`
- `--lambda-transformer-domain-adapter-l2`

域特征抽取位置：

- `--domain-feature-tap {inv_mean,pre_transformer_last,frontend_mean,concat_inv_pre}`

这说明 v3 实际上已经具备了较完整的“从前端到 Mamba 内部再到 Transformer”的域适配实验能力。

---

## 6. SPD / DD-Mamba 的核心技术实现

SPD / DD-Mamba 的核心文件是：`cd_mambatt/models/dd_mamba.py`

### 6.1 设计原则：尽量保留标准 Mamba 主体

`DDMambaBlock` 不是完全重写 Mamba，而是：

**保留：**

- `in_proj`
- `conv1d`
- `A_log`
- `D`
- `out_proj`

**拆分：**

- `x_proj_inv / x_proj_spec`
- `dt_proj_inv / dt_proj_spec`

即：

- 标准 Mamba 的“选择性参数生成”部分被拆成 invariant / specific 两支；
- 但状态空间算子主体、离散化逻辑、跳连 D 路径、输出投影仍尽量和标准 Mamba 一致。

这样做的工程意义是：

1. 尽量少破坏原始 Mamba；
2. 能直接复用官方 `mamba_ssm.Mamba(use_fast_path=False)` 的参数初始化；
3. 更容易解释“创新到底改了哪一段”。

### 6.2 为什么使用 `use_fast_path=False`

因为 SPD 需要显式拿到：

- `dt`
- `B`
- `C`
- `gate`
- `inv/spec` 分支张量

如果走 fused fast path，这些中间量很难干预和导出。因此 `DDMambaBlock` 采用显式 scan 路径。

### 6.3 当前 SPD 的拆分点

前向流程简化如下：

1. `hidden_states` 经过 `in_proj`
2. 拆成 `x` 和 `z`
3. `x` 经过共享的 `conv1d + act` 得到 `x_shared`
4. 可选地在 `x_shared` 上插入 target-only frontend adapter，得到 `x_combined`
5. 用 `x_proj_inv/x_proj_spec` 生成两路 `dt/B/C`
6. 用 gate 控制两路选择性参数或两路 scan 输出的融合
7. 再经过共享 `D` 路径和 `silu(z)` 调制
8. 最后 `out_proj`

也就是说：

> 当前 SPD 的拆分点在 `conv1d` 之后、`selective_scan` 之前。

这也正是后期机制诊断指出的一个关键问题：

> 如果域信息在 `conv1d` 输出处就已高度饱和，那么 SPD 再往后拆，两个分支的输入都已经“脏了”。

### 6.4 Specific 分支初始化策略

代码中没有把 specific 分支严格初始化为 0，而是使用：

- `std = 1e-4` 的 near-zero 初始化

原因在代码注释里写得很明确：

> 如果 spec 完全是 0，则 `dL/dgate ∝ spec_output = 0`，gate 初期几乎拿不到梯度。

因此当前策略是：

- 让 spec 起始时“几乎不影响主模型”
- 但又保留一个很小的非零梯度入口

### 6.5 Gate 机制

### Gate mode

支持两种：

- `token`
- `window`

含义：

1. `token`：每个时间步单独算 gate
2. `window`：先对整个窗口求平均上下文，再广播成整段共享 gate

### Gate scheme

支持两种：

- `shared`
- `dt_bc`

含义：

1. `shared`：一个 gate 同时控制 `dt`、`B`、`C` 或 specific output
2. `dt_bc`：`dt` 和 `B/C` 各用一套 gate

### Domain-conditioned gate

若打开 `--domain-conditioned-gate`，则会在 gate logit 上额外加：

- 一个可学习的 target-domain shift

这不是直接输入 one-hot 域向量做 MLP，而是最轻量的：

> 在 gate bias 上增加一个与 domain label 挂钩的可学习偏移。

用途是验证：

- gate 打不开，是不是因为模型不知道什么时候该用 spec；
- 如果显式告诉它“当前是 target 样本”，spec 通路是否会更积极参与。

### 6.6 Frontend adapter

当前在 `conv1d` 输出后已经预留了 target-only 前端适配：

- `none`
- `target_affine`
- `target_residual`

实现方式：

1. `target_affine`：对 target 样本施加逐通道 scale/bias
2. `target_residual`：加一个 1x1 Conv residual

这条线背后的动机是：

> 如果域偏移在 frontend 就已经形成，那么应该给 target 一个极轻量的前端修正，而不是把全部压力都交给后面的 `dt/B/C` 解耦。

### 6.7 Scan mode：`mixed` vs `dual_state`

这是 SPD 后期最核心的一个结构开关。

### A. `mixed`

原始 SPD 思路：

- 先把 specific 分支混到 `dt/B/C` 里
- 再跑一次 scan

即更像“参数级解耦 + 参数级融合”。

### B. `dual_state`

后期实现的新路径：

- invariant 路径单独跑一遍 scan
- specific 路径单独跑一遍 scan
- 在 scan core 或 scan output 之后再融合

当前代码实现中特别注意了一点：

> 共享 `D` 路径和 `silu(z)`，避免简单相加导致 `D` 被重复计算。

因此当 `gate ≈ 0` 时，Dual-State 可以更接近标准 Mamba 行为，而不会把输出幅度无端翻倍。

### 6.8 `DDMambaBlock` 输出的辅助张量

当 `return_aux=True` 时，block 会返回很多对机制诊断很关键的张量：

- `inv_sequence`
- `spec_sequence`（若存在）
- `gate_sequence`
- `gate_mean`
- `gate_dt_sequence`
- `gate_bc_sequence`
- `conv_sequence`
- `combined_conv_sequence`

而 `MambAttRegressor._encode_internal(...)` 进一步把这些整理成：

- `features`
- `domain_features`
- `pre_transformer_features`
- `invariant_features`
- `specific_features`
- `frontend_features`
- `frontend_last`

这也是为什么后期能做：

- frontend MMD
- invariant-path GRL
- inv/spec orth
- hidden state drift 诊断
- gate 响应诊断

### 6.9 当前从机制上得到的已验证结论

根据已有诊断文档，以下结论已经有实验支持：

### 已被证据支持的结论

1. **域偏移进入得非常早**
   - `x_conv`、`z_raw`、`D_x` 在很早阶段就几乎完全可分域。
2. **SPD 当前的 spec 分支偏弱**
   - `gate_mean` 常停在 `0.12 ~ 0.18`。
3. **few-shot 梯度主要打在前端/共享层**
   - frontend、Transformer、head 的梯度更强，spec gate 更弱。
4. **Mamba hidden drift 确实存在**
   - 曾观测到约 `59x` 的累积放大；后续更细测量中，state drift 也明显高于理想稳定值。
5. **union SSL 的提升主要不是来自 spec gate 打开**
   - 更像是 invariant path 变强、hidden drift 变稳。

### 仍属于推测/待验证的解释

1. 把 SPD 拆分点前移到 `conv1d` 内部是否一定有效；
2. domain-conditioned gate 是否能稳定解决所有迁移方向；
3. Transformer adapter 是否能补齐 `FD003 -> FD001` 的反向退化；
4. 更大容量 Mamba 是否能同时提升 source fit 和 cross-domain upper bound。

---

## 7. 损失函数与训练目标细节

### 7.1 回归主损失

项目中的主任务始终是 RUL 回归，source / target few-shot labeled 样本上都用 MSE。

在跨域阶段通常同时存在：

- `source supervised loss`
- `target supervised loss`

这样可以避免模型在适配 target few-shot 时完全遗忘 source 上的故障语义。

### 7.2 Global MMD

`cd_mambatt/losses/mmd.py` 中：

- `gaussian_mmd_loss(...)`

实现为多核 Gaussian MMD，默认带宽组：

- `(1, 2, 4, 8, 16)`

要求输入是二维特征 `(batch, feature_dim)`。

### 7.3 Conditional MMD

- `conditional_gaussian_mmd_loss(...)`

做法：

- 仅在 source / target 共享的 stage label 上计算子集 MMD；
- 再按样本数加权平均。

它的作用是：

> 不只对齐整体分布，还尝试对齐“相同退化阶段”的条件分布。

### 7.4 Monotonic ranking loss

`local_monotonicity_loss(...)` 的形式很直接：

`relu(later - earlier + margin).mean()`

含义：

- 更晚窗口的预测值不应该更大；
- 如果出现违背，就产生惩罚。

这是 v2 中非常重要的一条 target 内部结构约束。

### 7.5 Domain adversarial loss (GRL)

`cd_mambatt/losses/domain_adversarial.py` 中实现了：

- 自定义 `gradient_reverse`
- `DomainDiscriminator`
- `compute_domain_adversarial_loss(...)`

域判别器结构非常轻：

- `Linear -> ReLU -> optional Dropout -> Linear(2类)`

返回统计包括：

- 总域分类准确率
- source 域准确率
- target 域准确率

这对机制诊断很有用，因为我们不只看 loss，还能看“到底还有多可分域”。

### 7.6 SPD / v3 中额外支持的正则项

v3 当前已接入或预留的附加项包括：

- `lambda-spec-domain`：鼓励 spec 特征对域更敏感
- `lambda-conditional-inv-mmd`：对 invariant branch 做条件 MMD
- `lambda-conditional-proto`：stage prototype 对齐
- `lambda-conditional-proto-ce`：prototype-based CE
- `lambda-inv-spec-orth`：inv/spec 正交约束
- `lambda-inv-spec-xcorr`：inv/spec 交叉相关抑制（稳定 MI surrogate）
- `lambda-spec-residual`：约束 specific residual 大小
- `lambda-inv-aux`：对 invariant branch 单独做回归监督
- `lambda-spec-reconstruction`：要求 spec 分支能重构 shared 预测中的残差部分
- `lambda-transformer-domain-adapter-l2`：限制 Transformer target adapter 幅度

这些项并不是全部都已经证明有效，但它们已经把代码平台扩展到了：

> 可以系统探索“前端 - Mamba 内部 - Transformer 后端 - 预测头”四个层面的域适配机制。

---

## 8. 当前已验证的关键结果与技术含义

### 8.1 稳定跨域主线：v2

当前已经验证的 5-shot 多任务主线结果：

| 任务 | Direct Transfer | CD-MambAtt v2 |
|---|---:|---:|
| `FD001 -> FD003` | 34.8204 | **21.9608** |
| `FD003 -> FD001` | 24.6059 | **19.8137** |
| `FD002 -> FD004` | 29.5028 | **22.1776** |
| `FD001 -> FD004` | 32.3820 | **24.3449** |
| `FD004 -> FD002` | 21.3666 | **19.5311** |
| `FD001 -> FD002` | 31.1029 | **22.9457** |

这说明：

- MambAtt 骨干用于跨域 few-shot RUL 是成立的；
- 当前最稳收益来源不是复杂结构，而是 `MMD + pseudo + monotonic` 的组合。

### 8.2 SPD 非 SSL 最优线

当前最强 non-SSL SPD 线（canonical `FD001 -> FD003`）约为：

- **`20.0151 ± 0.7523`**

它优于 v2，但提升幅度有限，且跨任务不完全稳定。

技术含义：

- SPD 不是完全无效；
- 但目前它的收益更像“局部可用的结构修正”，还没有成为稳定主导项。

### 8.3 当前 canonical 最优线：union SSL + no-spec adaptation

当前已验证最优 canonical 结果：

- `FD001 -> FD003 = 19.8483 ± 1.1460`

其特点是：

- 先做 source train ∪ target train 的 union SSL
- 后续 adaptation 反而关闭 spec-domain 强推路线（`lambda_spec_domain = 0`）

技术含义：

- 最有效的不是让 spec 分支更“激进”；
- 而是让整个 Mamba 编码过程更稳、更 domain-agnostic。

### 8.4 论文对齐 `K=1` probe 的结果边界

对齐某篇 task-embedding / meta-learning 论文的严格 probe 中，当前结果约为：

- `CD-MambAtt v3 strict probe = 46.30`
- plain `1-shot` finetune baseline = `56.32`

说明：

- 我们的方法对严格 `1-shot meta-learning` 场景仍然偏弱；
- 当前强项主要还是 **5-shot domain adaptation**，不是严格 few-shot meta-learning。

---

## 9. 机制诊断总结：我们目前真正知道了什么

这一部分是目前项目最有价值的技术收获之一。

### 9.1 已经比较清楚的因果链

根据多轮 probe，当前最可信的因果链是：

`原始输入 -> conv1d / frontend -> 特征已高度可分域 -> SPD 再往后拆时两支都已带域信息 -> spec gate 又很弱 -> 实际收益主要退化为 feature-level 对齐`

这条链条解释了为什么：

- 单纯在 `dt/B/C` 层面拆分，经常“拆了等于没拆”；
- 为什么 frontend 对齐、union SSL 稳定化经常比 spec gate 设计更有效。

### 9.2 当前最值得信任的工程判断

### 已经被证据支持

1. **先看前端，再看 SSM 内部。**
2. **先看 invariant path 是否变强，再谈 spec branch 是否解耦。**
3. **失败后必须先诊断改动有没有真正被模型用上。**
4. **跨域 few-shot 中，协议细节会显著改变结论。**

### 还不能下定论

1. 只增大容量是否一定能带来跨域上限提升；
2. SPD 如果继续前移拆分点，是否一定优于前端 adapter；
3. MI 惩罚、Gumbel gate、Decoder residual adapter 等新方向是否能在多任务上稳定成立。

---

## 10. 当前代码库已经具备但尚未完全用尽的能力

从技术储备角度看，当前仓库其实已经比最初大很多，具备以下“随时可做系统实验”的接口：

1. **协议对齐能力**
   - 21 sensors / paper14
   - zscore / minmax
   - 5-shot / 1-shot
   - fixed split / resampled split

2. **前端适配能力**
   - frontend MMD / frontend feature tap
   - target affine / target residual adapter

3. **Mamba 内部适配能力**
   - mixed / dual-state
   - token / window gate
   - shared / dt_bc gate
   - domain-conditioned gate

4. **Transformer 后端适配能力**
   - target shift / target FiLM

5. **对齐损失能力**
   - MMD
   - conditional MMD
   - GRL
   - prototype alignment
   - inv/spec orth / xcorr

6. **预训练能力**
   - 同子集 SSL
   - 跨域 union SSL

换句话说：

> 当前瓶颈不再是“代码做不到”，而是“哪条结构假设最值得继续投入算力”。

---

## 11. 当前技术瓶颈与下一步可操作方向

### 11.1 当前瓶颈

### 瓶颈 A：source fit 仍不够强

监督复现仍在 `15.x` RMSE 左右，没有贴近论文 `11.46`。

这意味着：

- source 表征上限本身可能还偏低；
- 会压低后续迁移上限。

### 瓶颈 B：SPD 的功能性弱于其结构存在感

虽然 SPD 在代码和理论上都已经很完整，但从诊断看：

- gate 偏小
- spec 贡献偏弱
- inv/spec 域差异不够大

说明“有结构”不等于“有功能”。

### 瓶颈 C：当前最有效增益并不直接来自主创新点

当前最优收益来自：

- union SSL
- invariant path 稳定化
- frontend / shared path 的域偏移缓解

这意味着后续论文叙事需要谨慎：

- 若继续坚持 SPD 为中心，必须证明它确实提供不可替代收益；
- 否则更合理的主线可能是“稳定化 + 前端对齐 + 轻量域适配”。

### 11.2 下一步更合理的技术顺序

结合当前代码能力与已有证据，更合理的顺序应该是：

1. **先做定向诊断，再决定结构修改**
2. **优先验证前端域偏移修正是否能稳定多任务收益**
3. **若继续做 SPD，优先证明新改动真的改变了 gate / spec / drift 统计**
4. **再考虑是否扩大容量，而不是无止境堆 loss 权重**

这也是当前全局 `AGENTS.md` 中已经固化下来的原则：

> 实验失败后，先查原因和证据，再决定下一步；禁止无诊断连续堆新想法。

---

## 12. 导师沟通时可直接使用的技术摘要

如果要用一段较短的话向导师解释当前项目的“技术实质”，可以直接表述为：

> 我们已经完成了 MambAtt 在监督和同子集自监督场景下的主路径复现，并在此基础上建立了一个稳定的跨域 few-shot RUL 框架。当前跨域主线由 source supervision、target few-shot supervision、MMD 对齐、伪标签阶段学习和局部单调性约束构成；在此之上，我们进一步把 Mamba 内部改造成带 invariant/specific 双支路的 SPD/DD-Mamba，并加入 dual-state、domain-conditioned gate、frontend adapter、Transformer domain adapter 等结构位。通过后续机制诊断，我们发现当前跨域瓶颈并不只在 SSM 内部解耦，而更早出现在 frontend 的域信息刻入以及 invariant path 的稳定性上。也因此，当前最优结果更多来自 union SSL 对 Mamba 动力学的稳定化，而不是简单依赖 specific 分支打开。这一结论已经让项目从“盲目试错”转向“基于证据的结构研究”。

---

## 13. 建议与本文档配套阅读的文件

若需要继续深入，请按以下顺序阅读：

1. [`stage_report_for_advisor_zh_2026-04-11.md`](./stage_report_for_advisor_zh_2026-04-11.md)
2. [`project_status.md`](./project_status.md)
3. [`cross_domain_experiment_log.md`](./cross_domain_experiment_log.md)
4. [`mamba_mechanism_diagnosis_2026-04-06.md`](./mamba_mechanism_diagnosis_2026-04-06.md)
5. [`self_supervised_reproduction.md`](./self_supervised_reproduction.md)
6. [`supervised_reproduction.md`](./supervised_reproduction.md)
7. [`published_baselines.md`](./published_baselines.md)

