# FOMLN Reproduction Progress (2026-04-01)

## Status

This document is the running record for the `FOMLN` published-baseline
reproduction track inside the `CD-MambAtt` project.

Current status:

- **do not use the early `fomln_*` skeleton runs as formal baseline results**
- the implementation has been reset from a runnable skeleton to a
  **paper-alignment-in-progress reproduction**
- all validation from this point forward is **CUDA-only**
- the current strongest evidence now suggests the paper's `15-shot`
  protocol is much more plausibly interpreted at the
  **unit / trajectory level**, not as `15` random sliding windows

---

## 1. Why the first version was rejected as a formal reproduction

The earlier runnable baseline was useful for environment debugging, but it was
not sufficiently aligned with the paper to count as a real reproduction.

Main problems in the old version:

1. attention implementation was too approximate
2. source meta-training used mini-batch style sampling instead of explicit
   source meta-tasks
3. target adaptation repeatedly resampled support batches instead of using a
   fixed `15-shot` support set
4. model width was not tied tightly enough to the paper's `1D conv -> Conformer`
   description

Because of that, the old outputs under:

- `/home/shelterpl/cd_mambatt/runs/fomln_smoke*`
- `/home/shelterpl/cd_mambatt/runs/fomln_gpu_stage*`

should be treated only as **historical smoke/debug runs**.

---

## 2. Paper audit currently driving the reproduction

From the local FOMLN paper and extracted text, the key reproducible settings are:

- target regime:
  - `15-shot`
- selected sensors:
  - `2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21`
- window length:
  - `30`
- label cap:
  - `125`
- multi-condition preprocessing:
  - operating-condition-based standardization (`CS`) for `FD002` and `FD004`
- encoder front-end:
  - `1D convolution`
  - kernel size `10`
  - stride `1`
  - kernel count `10`
- Conformer block hyperparameters reported in the paper:
  - number of heads `8`
  - `d_k = d_v = 64`
  - attention dropout `0.2`
  - FFM dropout `0.2`
  - convolution-module depthwise kernel size `31`
  - convolution-module dropout `0.2`
- meta-training hyperparameters:
  - inner lr `0.001`
  - outer lr `0.1`
  - inner loops `10`
  - outer loops `50`
- adaptive weighted loss:
  - `beta = 1`
  - `lambda = 1`

---

## 3. Current implementation changes completed

Files:

- model:
  - `/home/shelterpl/cd_mambatt/cd_mambatt/models/fomln.py`
- runner:
  - `/home/shelterpl/cd_mambatt/train_fomln_baseline.py`

### 3.1 Model-side changes

The implementation is now much closer to the paper:

- custom multi-head self-attention instead of relying on a simplified stock
  attention path
- explicit `d_k = d_v = 64`
- current working default `d_model = 512`
- front-end convolution uses:
  - in-channels = `15`
  - out-channels = `10`
  - kernel size = `10`
  - stride = `1`
  - no extra same-padding shortcut
- regressor remains:
  - flatten
  - fully connected layer

### 3.2 Training-side changes

The training flow has also been made more paper-like:

- source-side training now uses **explicit source meta-tasks**
- the runner now supports both:
  - `source-task-level = windows`
  - `source-task-level = units`
- the source meta-update has been corrected to the paper-like
  **sequential interpolation**
  - `Φ ← (1 - α) Φ + α Φ_j`
- default source meta-task size is `15`
- each inner loop updates on the full sampled source task
- the source inner-loop optimizer default is now:
  - **`Adam`**
- target adaptation now uses a **fixed 15-shot support set**
- target adaptation optimizer is now configurable separately because the
  paper explicitly anchors the meta-training inner loop, but does not
  clearly formalize the meta-test optimizer
- evaluation remains on target test last windows
- CUDA is required by the runner
- run outputs now save:
  - `config.json`
  - `run_config.json`
  - `meta_history.json`
  - `adapt_history.json`

---

## 4. Remaining paper-alignment gaps

The implementation is now substantially closer to the paper, but I am **not yet
claiming full exact reproduction**.

The main unresolved ambiguities are in the paper itself:

1. **meta-task construction is underspecified**
   - the paper says the training set is divided into multiple meta-tasks
   - it does not clearly define whether these tasks are:
     - window-level sample groups
     - trajectory-level tasks
     - condition-specific tasks
     - or another partitioning strategy
   - current strongest working choice:
     - **unit-level source meta-tasks**
     - **unit-level target `15-shot` support**
   - rationale:
     - this interpretation produces by far the best same-domain and
       cross-domain reproduction-development results so far

2. **Conformer internal width is not fully explicit**
   - the paper clearly reports:
     - conv kernel count = `10`
     - `h = 8`
     - `d_k = d_v = 64`
   - but it does not explicitly state the final encoder channel width after the
     front-end conv and before/inside the conformer stack
   - current choice:
     - use `d_model = 512` as the working default
     - rationale:
       - `h = 8` and `d_k = d_v = 64` strongly suggest a `512`-dim conformer
         hidden size
       - a direct probe showed `d_model = 512` is much stronger than `10` on a
         same-domain alignment test
     - keep custom projections for `Q/K/V`

3. **final normalization after the Conformer block is not clearly stated**
   - current default:
     - disabled
   - optional flag exists:
     - `--use-final-norm`

4. **same-domain / cross-domain task construction details are not fully formalized**
   - current reproduction focuses on the paper's cross-domain `15-shot` claim
   - but some source-task sampling details still require interpretation

---

## 5. CUDA environment status

The `cd_mamba` environment is confirmed usable for CUDA-only experiments:

- `nvidia-smi` works
- `torch.cuda.is_available()` is `True`
- GPU:
  - `NVIDIA GeForce RTX 4060 Laptop GPU`
- `mamba_ssm` CUDA forward works
- `nvcc` is available inside `cd_mamba`

Extra environment hooks were added:

- `/home/shelterpl/miniconda3/envs/cd_mamba/etc/conda/activate.d/cuda_env.sh`
- `/home/shelterpl/miniconda3/envs/cd_mamba/etc/conda/deactivate.d/cuda_env.sh`

---

## 6. First post-reset CUDA validation

This run is a **runtime validation of the revised paper-alignment implementation**,
not a formal benchmark.

### 6.1 CUDA smoke: revised reproduction path

Command:

```bash
conda run -n cd_mamba python /home/shelterpl/cd_mambatt/train_fomln_baseline.py \
  --task FD001_TO_FD003 \
  --device cuda \
  --outer-loops 1 \
  --meta-batch-tasks 1 \
  --source-task-size 15 \
  --inner-steps 1 \
  --adapt-steps 1 \
  --target-scale 125 \
  --grad-clip-norm 1.0 \
  --max-source-train-windows 128 \
  --max-target-train-windows 256 \
  --max-target-test-windows 64 \
  --output-dir /home/shelterpl/cd_mambatt/runs/fomln_repro_cuda_smoke_v3
```

Output:

- `/home/shelterpl/cd_mambatt/runs/fomln_repro_cuda_smoke_v3/FD001_TO_FD003/summary.json`

Observed:

- run completed successfully on **CUDA**
- revised implementation is numerically stable enough for continued work
- this run is for **implementation validation only**

Smoke metrics:

- `source_meta_task_count = 9`
- `direct_target_rmse = 83.4471`
- `adapted_target_rmse = 82.7856`

Interpretation:

- the new code path is functional
- but formal reproduction-quality results should **not** be judged from this
  tiny smoke configuration

### 6.2 CUDA candidate stability run: revised reproduction path

This run is still **not a formal baseline table result**, but it is the first
more substantive CUDA check after switching from the skeleton to the revised
paper-alignment code path.

Command:

```bash
conda run -n cd_mamba python /home/shelterpl/cd_mambatt/train_fomln_baseline.py \
  --task FD001_TO_FD003 \
  --device cuda \
  --outer-loops 10 \
  --meta-batch-tasks 4 \
  --source-task-size 15 \
  --inner-steps 10 \
  --adapt-steps 10 \
  --target-scale 125 \
  --grad-clip-norm 1.0 \
  --max-source-train-windows 2048 \
  --max-target-train-windows 4096 \
  --output-dir /home/shelterpl/cd_mambatt/runs/fomln_repro_cuda_candidate1
```

Output:

- `/home/shelterpl/cd_mambatt/runs/fomln_repro_cuda_candidate1/FD001_TO_FD003/summary.json`

Observed:

- `source_meta_task_count = 137`
- `direct_target_rmse = 71.5558`
- `adapted_target_rmse = 69.1342`
- `best_adapt_step = 10`

Interpretation:

- the revised implementation remains stable under a noticeably stronger CUDA
  setting
- adaptation is still improving over direct transfer
- however, the result is still far from the published FOMLN paper number, so
  this run remains a **reproduction-development checkpoint**, not a publishable
  comparison

### 6.3 Same-domain ambiguity probe: `d_model = 10` vs `512`

Purpose:

- determine which hidden width is more plausible for the paper, because the
  paper explicitly reports:
  - `h = 8`
  - `d_k = d_v = 64`
  - but does not explicitly print the final conformer hidden width

Commands:

```bash
conda run -n cd_mamba python /home/shelterpl/cd_mambatt/train_fomln_baseline.py \
  --same-domain-subset FD001 \
  --device cuda \
  --outer-loops 10 \
  --meta-batch-tasks 4 \
  --source-task-size 15 \
  --inner-steps 10 \
  --adapt-steps 10 \
  --target-scale 125 \
  --grad-clip-norm 1.0 \
  --max-source-train-windows 2048 \
  --max-target-train-windows 2048 \
  --output-dir /home/shelterpl/cd_mambatt/runs/fomln_same_fd001_dmodel10_probe \
  --d-model 10
```

```bash
conda run -n cd_mamba python /home/shelterpl/cd_mambatt/train_fomln_baseline.py \
  --same-domain-subset FD001 \
  --device cuda \
  --outer-loops 10 \
  --meta-batch-tasks 4 \
  --source-task-size 15 \
  --inner-steps 10 \
  --adapt-steps 10 \
  --target-scale 125 \
  --grad-clip-norm 1.0 \
  --max-source-train-windows 2048 \
  --max-target-train-windows 2048 \
  --output-dir /home/shelterpl/cd_mambatt/runs/fomln_same_fd001_dmodel512_probe \
  --d-model 512
```

Observed:

- `d_model = 10`
  - direct RMSE = `72.6529`
  - adapted RMSE = `66.1854`
- `d_model = 512`
  - direct RMSE = `53.3864`
  - adapted RMSE = `38.1539`

Interpretation:

- `d_model = 512` is dramatically stronger than `10`
- this is consistent with the paper's Table 1 attention settings
- therefore the runner default has been moved to:
  - **`d_model = 512`**

### 6.4 Sequential meta-update + all-Adam probe

Purpose:

- verify the paper-corrected sequential meta-update
- test the OCR-indicated `Adam` inner loop directly in both
  source meta-training and target adaptation

Outputs:

- `/home/shelterpl/cd_mambatt/runs/fomln_seqadam_same_fd001_probe/FD001_TO_FD003`
- `/home/shelterpl/cd_mambatt/runs/fomln_seqadam_fd001_to_fd003_probe/FD001_TO_FD003`

Protocol:

- `d_model = 512`
- `outer-loops = 50`
- `meta-batch-tasks = 4`
- `inner-steps = 10`
- `adapt-steps = 10`
- `target-scale = 125`
- `grad-clip-norm = 1.0`
- `source-task-level = windows`
- `shot-level = windows`
- source inner optimizer = `Adam`
- target adaptation optimizer = `Adam`

Observed:

- same-domain `FD001 -> FD001`
  - direct RMSE = `25.1680`
  - adapted RMSE = `61.5891`
- cross-domain `FD001 -> FD003`
  - direct RMSE = `36.9894`
  - adapted RMSE = `107.2370`

Interpretation:

- the **sequential meta-update itself is beneficial** because the direct
  cross-domain RMSE improved substantially relative to older candidate runs
- however, using `Adam` unchanged for the target adaptation loop is
  **numerically unstable** in this implementation
- practical decision:
  - keep the paper-like `Adam` source inner loop
  - decouple the target adaptation optimizer for further reproduction work

### 6.5 Decoupled optimizer probe: source inner `Adam`, target adapt `SGD`

Purpose:

- preserve the paper-aligned `Adam` source inner loop
- stabilize the target support adaptation stage

Outputs:

- `/home/shelterpl/cd_mambatt/runs/fomln_seqadam_meta_sgd_adapt_same_fd001_probe/FD001_TO_FD003`
- `/home/shelterpl/cd_mambatt/runs/fomln_seqadam_meta_sgd_adapt_fd001_to_fd003_probe/FD001_TO_FD003`

Protocol:

- same as Section 6.4, except:
  - target adaptation optimizer = `SGD`

Observed:

- same-domain `FD001 -> FD001`
  - direct RMSE = `25.1680`
  - adapted RMSE = `21.6903`
- cross-domain `FD001 -> FD003`
  - direct RMSE = `36.9894`
  - adapted RMSE = `34.9648`

Interpretation:

- decoupling the target optimizer removes the severe adaptation explosion
- this confirms the main instability was in the **meta-test adaptation
  optimizer choice**, not the corrected sequential meta-update itself
- but the result quality still remained far from the paper, so the
  `15-shot` definition was then revisited

### 6.6 `15-shot` definition probe: windows vs units

Purpose:

- test whether the paper's `15-shot` wording is more plausibly referring to:
  - `15` random sliding-window samples
  - or `15` full units / trajectories

Output:

- `/home/shelterpl/cd_mambatt/runs/fomln_seqadam_meta_sgd_adapt_same_fd001_unitshot_probe/FD001_TO_FD003`

Protocol:

- `source-task-level = windows`
- `shot-level = units`
- source inner optimizer = `Adam`
- target adaptation optimizer = `SGD`
- other settings kept as in Section 6.5

Observed:

- same-domain `FD001 -> FD001`
  - direct RMSE = `25.1680`
  - adapted RMSE = `19.6785`

Interpretation:

- simply changing the target `15-shot` definition from windows to units
  produced a **clear improvement**
- this strongly suggests the paper's `15-shot` wording is likely
  trajectory-level in practice

### 6.7 Current strongest reproduction-development candidate

Purpose:

- align both:
  - source meta-task construction
  - target `15-shot` support construction
  with the unit / trajectory-level interpretation

Outputs:

- `/home/shelterpl/cd_mambatt/runs/fomln_seqadam_metaunits_sgd_adapt_same_fd001_unitshot_probe/FD001_TO_FD003`
- `/home/shelterpl/cd_mambatt/runs/fomln_seqadam_metaunits_sgd_adapt_fd001_to_fd003_probe/FD001_TO_FD003`

Protocol:

- `source-task-level = units`
- `source-task-size = 15`
- `shot-level = units`
- `target-shots = 15`
- source inner optimizer = `Adam`
- target adaptation optimizer = `SGD`
- `d_model = 512`
- `outer-loops = 50`
- `meta-batch-tasks = 4`
- `inner-steps = 10`
- `adapt-steps = 10`
- `target-scale = 125`
- `grad-clip-norm = 1.0`

Observed:

- same-domain `FD001 -> FD001`
  - direct RMSE = `26.3622`
  - adapted RMSE = `14.9430`
- cross-domain `FD001 -> FD003`
  - direct RMSE = `32.2795`
  - adapted RMSE = `21.4792`

Interpretation:

- this is the **strongest FOMLN reproduction-development configuration**
  achieved so far
- compared with the earlier window-level candidate on `FD001 -> FD003`:
  - adapted RMSE improved from `33.0446` to **`21.4792`**
- compared with the previous best same-domain strong run:
  - adapted RMSE improved from `26.7983` to **`14.9430`**
- key conclusion:
  - the paper's notion of "sample" is very likely being operationalized at the
    **unit / trajectory level**
  - this is now the main working interpretation for the reproduction track

### 6.8 Additional cross-domain transfer validation under the current candidate

Purpose:

- check whether the current unit-level interpretation also improves on
  transfer pairs beyond the canonical `FD001 -> FD003`

Outputs:

- `/home/shelterpl/cd_mambatt/runs/fomln_seqadam_metaunits_sgd_adapt_fd003_to_fd001_probe/FD003_TO_FD001`
- `/home/shelterpl/cd_mambatt/runs/fomln_seqadam_metaunits_sgd_adapt_fd001_to_fd004_probe/FD001_TO_FD004`
- `/home/shelterpl/cd_mambatt/runs/fomln_seqadam_metaunits_sgd_adapt_fd002_to_fd004_probe/FD002_TO_FD004`

Protocol:

- identical to Section 6.7:
  - `source-task-level = units`
  - `shot-level = units`
  - source inner optimizer = `Adam`
  - target adaptation optimizer = `SGD`
  - `d_model = 512`
  - `outer-loops = 50`
  - `meta-batch-tasks = 4`
  - `inner-steps = 10`
  - `adapt-steps = 10`
  - `target-scale = 125`
  - `grad-clip-norm = 1.0`

Observed:

- `FD003 -> FD001`
  - direct RMSE = `17.6081`
  - adapted RMSE = `16.7696`
- `FD001 -> FD004`
  - direct RMSE = `34.4970`
  - adapted RMSE = `23.3202`
- `FD002 -> FD004`
  - direct RMSE = `29.0239`
  - adapted RMSE = `22.0029`

Interpretation:

- the current candidate is **not limited to one task pair**
- target adaptation consistently improves over direct transfer:
  - `FD003 -> FD001`: gain ≈ `0.84` RMSE
  - `FD001 -> FD004`: gain ≈ `11.18` RMSE
  - `FD002 -> FD004`: gain ≈ `7.02` RMSE
- the improvement is largest on the harder transfers involving `FD004`
- current best single-seed reproduction-development results are now:

| Task | Direct RMSE | Adapted RMSE |
|---|---:|---:|
| `FD001 -> FD003` | 32.2795 | **21.4792** |
| `FD003 -> FD001` | 17.6081 | **16.7696** |
| `FD001 -> FD004` | 34.4970 | **23.3202** |
| `FD002 -> FD004` | 29.0239 | **22.0029** |

### 6.9 Source-task coverage probe: full unit-task meta-batch

Purpose:

- test the next major ambiguity:
  whether each outer loop should see **all** source unit-level meta-tasks
  instead of only a sampled subset

Output:

- `/home/shelterpl/cd_mambatt/runs/fomln_seqadam_metaunitsFULL_sgd_adapt_fd001_to_fd003_probe/FD001_TO_FD003`

Protocol:

- same as the working candidate, except:
  - `meta-batch-tasks = 7`
  - this equals the full source unit-task count for `FD001`

Observed:

- `FD001 -> FD003`
  - direct RMSE = `33.6033`
  - adapted RMSE = `24.1990`

Interpretation:

- using **all** source unit-level tasks in every outer step is **worse**
  than the sampled-task setting
- compared with the current best candidate on the same task:
  - sampled unit-task meta-batch (`4`) -> adapted RMSE `21.4792`
  - full unit-task meta-batch (`7`) -> adapted RMSE `24.1990`
- practical decision:
  - keep `meta-batch-tasks = 4` as the working default for now

### 6.10 GPU-memory diagnosis and mini-batch inner-loop correction

Problem:

- even after switching to single-seed / single-process execution, the
  unit-level protocol could still push GPU usage high enough to trigger
  Windows shared GPU memory spillover on the RTX 4060 laptop

Diagnosis:

- the previous implementation moved the **entire source task** and the
  **entire target support pool** onto the GPU during each inner/adaptation
  update
- under unit-level `15-shot`, one task/support set can contain thousands of
  sliding windows
- this is also not ideal conceptually, because the paper's adaptive weighted
  loss explicitly defines `freq(y_i)` within a **batch**

Code change:

- `train_fomln_baseline.py`
  - source-task inner loop now samples a mini-batch each step
  - target adaptation loop now also samples a mini-batch each step
  - full task/support tensors remain on CPU and only the sampled batch is moved
    to CUDA
- new CLI flags:
  - `--inner-batch-size`
  - `--adapt-batch-size`
- current stable choice:
  - `inner_batch_size = 256`
  - `adapt_batch_size = 256`

Interpretation:

- this change is both:
  - a **memory-stability fix**
  - and a **better match to the paper's batch-based weighted-loss wording**

### 6.11 Mini-batch stability verification

Output:

- `/home/shelterpl/cd_mambatt/runs/fomln_minibatch_verify_fd003_to_fd001/FD003_TO_FD001`

Protocol:

- same unit-level candidate as before, plus:
  - `inner_batch_size = 256`
  - `adapt_batch_size = 256`

Observed:

- `FD003 -> FD001`, `seed = 44`
  - direct RMSE = `24.5753`
  - adapted RMSE = `19.0092`
- the run completed normally on CUDA without reproducing the previous shared
  memory / VRAM blow-up

Interpretation:

- mini-batch inner-loop training is now the preferred execution mode for the
  FOMLN baseline track

### 6.12 Formal FOMLN baseline results (current stable mini-batch protocol)

Formal protocol used below:

- `source-task-level = units`
- `shot-level = units`
- `source-task-size = 15`
- `target-shots = 15`
- `meta-batch-tasks = 4`
- `inner-steps = 10`
- `adapt-steps = 10`
- `inner_batch_size = 256`
- `adapt_batch_size = 256`
- source inner optimizer = `Adam`
- target adaptation optimizer = `SGD`
- `d_model = 512`
- `target-scale = 125`
- `grad-clip-norm = 1.0`
- seeds = `42, 43, 44, 45, 46`

Outputs:

- `/home/shelterpl/cd_mambatt/runs/fomln_formal_mb256_unit15_5seeds_fd001_to_fd003/FD001_TO_FD003/summary_aggregated.json`
- `/home/shelterpl/cd_mambatt/runs/fomln_formal_mb256_unit15_5seeds_fd003_to_fd001/FD003_TO_FD001/summary_aggregated.json`
- `/home/shelterpl/cd_mambatt/runs/fomln_formal_mb256_unit15_5seeds_fd002_to_fd004/FD002_TO_FD004/summary_aggregated.json`
- `/home/shelterpl/cd_mambatt/runs/fomln_formal_mb256_unit15_5seeds_fd001_to_fd004/FD001_TO_FD004/summary_aggregated.json`

Results:

| Task | Mean direct RMSE | Mean adapted RMSE | Std | Best adapted RMSE |
|---|---:|---:|---:|---:|
| `FD001 -> FD003` | 34.0750 | **25.4748** | 1.9631 | 22.2528 |
| `FD003 -> FD001` | 25.0772 | **20.1261** | 1.3369 | 18.0928 |
| `FD002 -> FD004` | 31.1871 | **26.0206** | 2.7396 | 22.6677 |
| `FD001 -> FD004` | 30.9944 | **25.0808** | 1.4670 | 23.6054 |

Interpretation:

- the current stable FOMLN baseline now has a **formal 5-seed result set**
  on four important transfer pairs
- adaptation improves over direct transfer in all four cases
- among these tasks, the strongest mean adapted result is:
  - `FD003 -> FD001 = 20.1261`
- the most difficult currently tested pair remains:
  - `FD002 -> FD004 = 26.0206`

---

## 7. Next actions

Before reporting any formal FOMLN result table, the next steps are:

1. treat the following as the current **stable formal baseline protocol**:
   - `source-task-level = units`
   - `shot-level = units`
   - source inner optimizer = `Adam`
   - target adaptation optimizer = `SGD`
   - `inner_batch_size = 256`
   - `adapt_batch_size = 256`
2. move this candidate to:
   - optional additional task:
     - `FD004 -> FD002`
3. then compare this formal baseline table with:
   - published FOMLN paper numbers
   - our current CD-MambAtt results
4. optionally add:
   - `FD004 -> FD002`
   - as a symmetry / robustness check before the comparison table
5. mark the first run set that satisfies the above as:
   - **formal FOMLN baseline**

---

## 8. Important rule for the project log

From this point on:

- every FOMLN code-path change
- every CUDA validation run
- every decision about paper ambiguity

must be written to the project record, with command path and conclusion.
