# DD-SSM Roadmap

Last updated: `2026-04-02`

## 1. Feasibility answer first

> **Yes, the DD-SSM direction is technically feasible.**
> But the three proposed parts should not be treated as equal.

Current judgment:

1. **SPD (Selectivity Projection Disentanglement)** = the real core innovation and the first thing to implement
2. **SSDA (State-Space Domain Alignment)** = feasible, but should come only after SPD works
3. **DAAD (Degradation-Aware Adaptive Discretization)** = feasible, but best treated as an auxiliary enhancement

## 2. Why the current `CD-MambAtt v2` is not enough as a final paper

Current `v2` improves results, but most of the adaptation happens at the
feature/loss level:

- `MMD`
- pseudo-labeling
- monotonic regularization

The Mamba backbone itself is still mostly a black box.

So the current weakness is:

> good empirical gains, but insufficient architecture-level novelty.

## 3. Why SPD is feasible in the current codebase

The current model already wraps official `mamba_ssm.Mamba`, and the official
implementation exposes the right places to intervene.

Important implementation facts already verified locally:

- official Mamba uses `in_proj`, `conv1d`, `x_proj`, `dt_proj`, `out_proj`
- `B` and `C` are generated from chunks of `x_proj`
- `Delta` is generated through `x_proj -> dt_proj`

That means the innovation can be framed correctly as:

> **projector-level disentanglement of Mamba selectivity generation**

rather than as a vague feature-space trick.

## 4. The right scope for v0

Do **not** implement `SPD + SSDA + DAAD` all at once.

The safest scope is:

### Phase 1: SPD-only v0

- replace the current Mamba block with a `DDMambaBlock`
- split the selectivity-generation path into invariant / specific branches
- keep the rest of the current `CD-MambAtt v2` scaffold as stable as possible
- add only the minimum new alignment pressure needed for SPD to matter

### Phase 2: add SSDA if SPD works

- expose or collect intermediate state statistics
- align state-space statistics only after SPD is stable

### Phase 3: optionally add DAAD

- inject stage/degradation prior into the delta path
- keep it as an enhancement, not the main novelty claim

## 5. Safe technical constraints

### 5.1 Use the correct official-Mamba terminology

Prefer:

- projector family / selectivity generation
- `x_proj` / `dt_proj`

Avoid overselling with inaccurate `Linear_B / Linear_C / Linear_Delta` language.

### 5.2 Preserve Delta positivity / stability

Do **not** use an unconstrained additive form like:

- `Delta = Delta_inv + g * Delta_spec`

Safer options:

- pre-softplus fusion:
  - `dt_pre = dt_inv + g * dt_spec`
  - `Delta = softplus(dt_pre + bias)`
- or positive multiplicative modulation:
  - `Delta = Delta_inv * exp(g * delta_spec)`

### 5.3 `g = 0` is only an ablation

It should not be the default inference rule.

### 5.4 If internal SSM disentanglement is unstable, fall back in layers

Fallback order:

1. split `x_proj` / `dt_proj` directly
2. if unstable, keep the split but apply the domain loss at DD-block output instead of the deepest internal tensors
3. only after that consider a lighter projector-level fallback

## 6. Minimal implementation plan

### 6.1 Files to add or modify

Recommended first-pass write set:

- `cd_mambatt/models/dd_mamba.py`
  - implement `DDMambaBlock`
- `cd_mambatt/models/mambatt.py`
  - add an option to instantiate `DDMambaBlock` inside the regressor
- `cd_mambatt/losses/domain_adversarial.py`
  - GRL + small domain classifier for the invariant branch or DD-block output
- `train_cd_mambatt_v3.py`
  - start from `train_cd_mambatt_v2.py` and add the SPD-specific path

Leave for later, not v0:

- state-level alignment loss file
- `selective_scan_ref` fork
- DAAD-specific delta conditioning

### 6.2 Minimal loss setup for SPD-only v0

Keep the current stable training scaffold as much as possible:

- source supervised regression loss
- existing target-side pseudo / monotonic / MMD path can stay unchanged initially
- **new loss in v0**: domain-adversarial loss tied to the invariant path (or DD-block output if safer)

The point of v0 is not to redesign the whole training objective again.
It is to verify whether putting the innovation **inside Mamba selectivity generation** helps.

## 7. Minimal experiment plan

Recommended order:

1. **single-domain smoke / stability test** on `FD001`
   - verify no NaN / OOM / severe regression
2. **canonical transfer** `FD001 -> FD003`
   - 3 seeds first
3. **harder transfer** `FD001 -> FD004`
   - 3 seeds first
4. only after that extend to 5 seeds and broader task coverage

## 8. Success criteria

SPD-only v0 should satisfy all three:

1. no obvious numerical instability on CUDA
2. no severe collapse versus the current backbone on single-domain `FD001`
3. on `FD001 -> FD003`, it should at least match or beat the current `CD-MambAtt v2` canonical result before we spend time on SSDA / DAAD

If SPD-only cannot clear that bar, do not add more modules yet.

## 9. Current bottom line

The right decision is:

> **Proceed with SPD first.**

Not because the other two ideas are impossible, but because only SPD has the
right mix of:

- feasibility
- novelty depth
- fit with the current codebase
- potential to fix the project's present "innovation not deep enough" problem
