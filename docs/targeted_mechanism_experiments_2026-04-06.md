# Targeted mechanism experiments (2026-04-06)

基于 `mamba_mechanism_diagnosis_2026-04-06.md` 的结论，做了两类定向实验：

1. **decoder-priority**
   - 新增 `transformer_head` adaptation freeze mode
   - 只让 Transformer + head 适配，或先适配它们再全量解冻

2. **front-end alignment**
   - 新增 `frontend_mean` domain tap
   - 直接把 invariant MMD 对齐位置前移到 `conv1d` 后的前端表征均值

相关代码：

- `cd_mambatt/models/dd_mamba.py`
- `cd_mambatt/models/mambatt.py`
- `train_cd_mambatt_v3.py`
- `scripts/run_targeted_adaptation_experiment.py`

实验汇总 JSON：

- `docs/generated/targeted_mechanism_experiments_summary_2026-04-06.json`

---

## 结果摘要

### A. decoder-priority 并没有带来收益

在 `FD001 -> FD003` / seed42 上：

- baseline SPD fullmatch: **22.40**
- `transformer_head` only: **27.12**
- `transformer_head` 5 epochs then full: **25.30**

结论：

- 单靠“先适配 decoder”并不能转化为更好的最终 test RMSE。
- quick-screen 里的短期优势，放到完整 20 epoch / val-selection 里没有兑现。

### B. front-end alignment 是目前更有效的方向

把 invariant MMD tap 从 `inv_mean` 改到 `frontend_mean` 后：

- seed42: `22.40 -> 21.39`
- seed43: `20.04 -> 19.49`
- seed44: `20.82 -> 20.29`

3-seed mean:

- baseline SPD fullmatch: **21.09 ± 0.98**
- `frontend_mean` full: **20.39 ± 0.78**

平均提升：

- **-0.70 RMSE**

结论：

- 这和机制诊断是一致的：域偏移在共享前端就已经进入，因此把对齐位置前移是有效的。
- 目前最值得继续加力的不是 decoder-only freeze，而是 **更早位置的域对齐 / 早期表征去域偏**。

