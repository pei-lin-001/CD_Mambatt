# Scripts

这里放**批处理脚本、结果汇总脚本、批量实验驱动器**，和 `experiments/` 不同：

- `experiments/` 更偏“一次性研究脚手架”
- `scripts/` 更偏“可复用的批量运行/汇总工具”

当前主要文件：

- `run_supervised_capacity_sweep.py`：监督容量 sweep
- `run_targeted_adaptation_experiment.py`：跨域定向机制实验驱动
- `run_ssl_targeted_adaptation_experiment.py`：SSL 相关定向实验驱动
- `mamba_mechanism_diagnosis.py`：机制诊断主脚本
- `aggregate_formal_result_tables.py`：汇总正式结果表
