"""Dual-Path MambAtt v2: gate reset + entropy regularization, 3-seed.

Fix from v1: the gate collapsed during source pre-training (to 0 or 1 depending
on seed).  This version:
  1) Resets the gate weights after loading the source checkpoint
  2) Initializes gate bias = +2.0 (start at ~0.88 favoring pre-trained Mamba path)
  3) Adds gate entropy regularization to prevent full collapse

Compare against:
  - v2 reference (5-shot):  21.96 RMSE
  - SPD best (3 seeds):    20.37 +/- 0.47 RMSE
  - DP-MambAtt v1:         21.36 +/- 1.52 (unstable gate)
  - DP-MambAtt v1 seed 44: 19.36 (when gate worked)
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch import nn
from _pathfix import ensure_repo_root

ensure_repo_root()

from cd_mambatt.data import load_cmapss_split
from cd_mambatt.losses.mmd import gaussian_mmd_loss
from cd_mambatt.models.mambatt import DualPathMambAttRegressor
from cd_mambatt.pseudo_labeling import assign_rul_stage_labels
from train_supervised import build_loader, set_seed, infer_device, evaluate
from train_cross_domain_baseline import (
    load_or_create_source_split, build_source_stage_data, load_or_create_target_partition,
)
from train_cd_mambatt_v1 import build_target_cd_data, fit_source_stage


def endless(loader):
    while True:
        for b in loader:
            yield b


def gate_entropy_loss(gate: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Negative entropy of gate values — minimizing this keeps gate near 0.5."""
    g = gate.clamp(eps, 1.0 - eps)
    return (g * g.log() + (1.0 - g) * (1.0 - g).log()).mean()


def reset_gate(model: DualPathMambAttRegressor, init_bias: float = 2.0):
    """Re-initialize the fusion gate after loading a checkpoint."""
    for m in model.fusion_gate.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    # Set bias on the last linear to favor Mamba path
    last_linear = model.fusion_gate[-1]
    with torch.no_grad():
        last_linear.bias.fill_(init_bias)


def build_dp_model(args, input_dim):
    return DualPathMambAttRegressor(
        input_dim=input_dim,
        d_model=args.d_model or input_dim,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        num_mamba_layers=args.num_mamba_layers,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward,
        transformer_impl=args.transformer_impl,
        transformer_norm_mode=args.transformer_norm_mode,
        transformer_inner_dropout=args.transformer_inner_dropout,
        mamba_block_mode="bare",
        fusion_gate_init_bias=0.0,
    )


def make_args():
    return argparse.Namespace(
        d_model=None, d_state=16, d_conv=8, expand=2, num_mamba_layers=1,
        num_transformer_layers=3, num_heads=7, dropout=0.5, dim_feedforward=84,
        transformer_impl="custom", transformer_norm_mode="pre",
        transformer_inner_dropout=0.0,
        mamba_block_mode="bare",
        spd_gate_init_bias=-2.0, spd_predictor_mode="shared_head",
        root="/home/shelterpl/data/CMAPSS", window_size=20, stride=1, rul_clip=125,
        target_scale=1.0, grad_clip_norm=0.0, source_train_ratio=0.8,
        source_split_seed=42, source_split_path=None,
        resample_source_split_per_seed=False, source_normalizer_fit_scope="train_only",
        target_shots=5, target_val_units=10,
        target_partition_path=None, resample_few_shot_per_seed=True,
        batch_size=64, source_epochs=50, target_epochs=20,
        lr=1e-3, target_lr=5e-4, weight_decay=0.0, target_weight_decay=0.0,
        source_loss_weight=1.0, target_loss_weight=1.0,
        lambda_mmd=0.1, mmd_sigmas="1,2,4,8,16",
        lambda_source_stage=1.0, lambda_pseudo=0.5, lambda_contrastive=0.0,
        lambda_monotonic=0.05, contrastive_temperature=0.1,
        disable_contrastive_feature_normalization=False,
        monotonic_margin=0.0, monotonic_pair_gap=1, monotonic_pair_stride=1,
        num_pseudo_stages=3, pseudo_start_quantile=0.5, pseudo_end_quantile=0.9,
        disable_pseudo_feature_normalization=False,
        source_val_all_windows=True, target_val_all_windows=True,
        num_workers=0, max_source_train_batches=None, max_target_train_batches=None,
    )


# --- Hyperparameters for gate control ---
GATE_INIT_BIAS = 2.0       # sigmoid(2.0) ≈ 0.88 → favor pre-trained Mamba
LAMBDA_GATE_ENTROPY = 0.1  # entropy reg to prevent gate collapse
CD_LR = 2e-3


def run_one_seed(seed, device):
    set_seed(seed)
    args = make_args()
    args.few_shot_seed = seed

    out_dir = Path("runs/dualpath_v2_3seed/FD001_TO_FD003")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    sf = load_cmapss_split(args.root, "FD001", "train", rul_clip=125)
    st = load_cmapss_split(args.root, "FD001", "test", rul_clip=125)
    tf = load_cmapss_split(args.root, "FD003", "train", rul_clip=125)
    tt = load_cmapss_split(args.root, "FD003", "test", rul_clip=125)

    ss = load_or_create_source_split(args, sf, out_dir, seed)
    sl, _ = build_source_stage_data(args, sf, st, ss)
    tp = load_or_create_target_partition(args, tf, out_dir, seed)
    tl, _ = build_target_cd_data(args, tf, tt, tp)

    # --- Source stage ---
    model = build_dp_model(args, 21).to(device)
    print(f"  [seed {seed}] Model params: {sum(p.numel() for p in model.parameters())}")
    src_stage = fit_source_stage(args, seed, model, sl, device, run_dir, 1.0, None)
    direct = evaluate(model, tl["test"], device, 1.0)["rmse"]
    print(f"  [seed {seed}] Source best val RMSE: {src_stage['best_val_rmse']:.2f}, "
          f"direct target RMSE: {direct:.2f}")

    # --- CD stage: load checkpoint, RESET gate, then adapt ---
    cd = build_dp_model(args, 21).to(device)
    cd.load_state_dict(torch.load(src_stage["checkpoint"], map_location=device)["model_state_dict"])

    # KEY FIX: reset gate after loading checkpoint (prevents collapse from source stage)
    reset_gate(cd, init_bias=GATE_INIT_BIAS)
    with torch.no_grad():
        test_gate = torch.sigmoid(cd.fusion_gate(torch.randn(1, 42, device=device)))
        print(f"  [seed {seed}] Gate after reset: ~{test_gate.item():.4f}")

    stage_head = nn.Linear(21, 3).to(device)
    all_params = list(cd.parameters()) + list(stage_head.parameters())
    optimizer = torch.optim.Adam(all_params, lr=CD_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    best_val, best_test, best_ep, best_gate = float("inf"), None, -1, 0.0
    for ep in range(1, 21):
        cd.train()
        stage_head.train()
        si = endless(sl["train"])
        ti = endless(tl["labeled"])
        ui = endless(tl["unlabeled"])
        ep_gate_sum, ep_gate_count = 0.0, 0

        for _ in range(max(len(sl["train"]), len(tl["labeled"]))):
            sx, sy = next(si)
            tx, ty = next(ti)
            tu, _ = next(ui)
            sx, sy, tx, ty, tu = [t.to(device) for t in [sx, sy, tx, ty, tu]]

            sa = cd.forward_features_with_aux(sx)
            ta = cd.forward_features_with_aux(tx)
            ua = cd.forward_features_with_aux(tu)

            # Supervised losses on fused features
            l_src = mse(cd.predict_from_features(sa["features"]), sy)
            l_tgt = mse(cd.predict_from_features(ta["features"]), ty)

            # Stage classification
            l_stage = ce(stage_head(sa["features"]),
                         assign_rul_stage_labels(sy, rul_clip=125., num_stages=3))

            # Global MMD on fused features
            l_mmd = gaussian_mmd_loss(
                sa["features"],
                torch.cat([ta["features"], ua["features"]]))

            # MMD on attention-only path (domain-invariant alignment)
            l_attn_mmd = gaussian_mmd_loss(
                sa["features_attn"],
                torch.cat([ta["features_attn"], ua["features_attn"]]))

            # Gate entropy regularization (prevents collapse to 0 or 1)
            l_gate_ent = gate_entropy_loss(sa["gate"]) + gate_entropy_loss(ta["gate"])

            loss = (l_src + l_tgt
                    + 0.1 * l_mmd
                    + 1.0 * l_stage
                    + 0.1 * l_attn_mmd
                    + LAMBDA_GATE_ENTROPY * l_gate_ent)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            ep_gate_sum += sa["gate_mean"].item() + ta["gate_mean"].item()
            ep_gate_count += 2

        scheduler.step()
        avg_gate = ep_gate_sum / max(ep_gate_count, 1)
        v = evaluate(cd, tl["val"], device, 1.0)
        t = evaluate(cd, tl["test"], device, 1.0)
        print(f"    ep {ep:2d}: val={v['rmse']:.2f} test={t['rmse']:.2f} "
              f"gate={avg_gate:.4f} lr={scheduler.get_last_lr()[0]:.6f}")
        if v["rmse"] < best_val:
            best_val, best_test, best_ep, best_gate = v["rmse"], t["rmse"], ep, avg_gate

    result = {
        "seed": seed, "direct_rmse": direct, "adapted_rmse": best_test,
        "best_epoch": best_ep, "best_gate_mean": best_gate,
    }
    with (run_dir / "result.json").open("w") as f:
        json.dump(result, f, indent=2)

    print(f"  seed {seed}: direct={direct:.2f} adapted={best_test:.2f} "
          f"epoch={best_ep} gate={best_gate:.4f}")
    return best_test, best_gate


def main():
    device = infer_device("auto")
    print("=" * 70)
    print("Dual-Path MambAtt v2: gate reset + entropy reg")
    print(f"  Gate init bias: {GATE_INIT_BIAS} (sigmoid ≈ {torch.sigmoid(torch.tensor(GATE_INIT_BIAS)).item():.3f})")
    print(f"  Gate entropy lambda: {LAMBDA_GATE_ENTROPY}")
    print(f"  CD LR: {CD_LR} + cosine")
    print("=" * 70)
    start = time.time()

    results, gates = [], []
    for s in [42, 43, 44]:
        r, g = run_one_seed(s, device)
        results.append(r)
        gates.append(g)

    m, sd = np.mean(results), np.std(results)
    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"RESULT: {m:.4f} +/- {sd:.4f}  ({elapsed/60:.1f} min)")
    print(f"Gate means: {[f'{g:.4f}' for g in gates]}")
    print(f"References:")
    print(f"  v2 ref (5-shot):        21.9608")
    print(f"  SPD best (3 seeds):     20.3700 +/- 0.4700")
    print(f"  DP-MambAtt v1:          21.3639 +/- 1.5239")
    print(f"  DP-MambAtt v1 best:     19.36 (seed 44)")
    for s, r, g in zip([42, 43, 44], results, gates):
        print(f"  seed {s}: {r:.2f}  (gate={g:.4f})")


if __name__ == "__main__":
    main()
