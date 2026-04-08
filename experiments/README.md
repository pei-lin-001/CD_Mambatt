# Experiments

这个目录存放**一次性实验脚本、机制探针、原型验证脚本**，避免把仓库根目录塞满。

## 当前结构

- `ablations/`：相对正式的定向对比、3-seed 验证、跨任务验证
- `diagnostics/`：机制分析、gate / hidden-state / drift 探针
- `prototypes/`：替代架构或组合思路原型
- `quick_tests/`：快速 A/B、临时验证、保留的历史小脚本

## 与根目录脚本的分工

根目录只保留主训练入口：

- `train_supervised.py`
- `train_self_supervised.py`
- `train_cross_domain_baseline.py`
- `train_cd_mambatt_v1.py`
- `train_cd_mambatt_v2.py`
- `train_cd_mambatt_v3.py`
- `train_fomln_baseline.py`
- `train_minimal.py`

## 运行说明

各子目录里都放了一份本地 `_pathfix.py`，因此脚本仍然可以直接运行，例如：

```bash
conda run -n cd_mamba python experiments/ablations/spd_highlr_frontend_3seed.py --help
```

如果后面再新增零散实验，优先放到对应子目录，不要再直接丢回仓库根目录。
