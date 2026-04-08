"""Focused test of 3 big-lever ideas. Levers 0,1 already tested."""
import torch, argparse, numpy as np
import torch.nn.functional as F
from pathlib import Path
from torch import nn
from _pathfix import ensure_repo_root

ensure_repo_root()

from cd_mambatt.data import load_cmapss_split, fit_normalizer, CMAPSSWindowDataset, build_windows
from cd_mambatt.losses.mmd import gaussian_mmd_loss
from cd_mambatt.pseudo_labeling import assign_rul_stage_labels
from train_supervised import build_model, build_loader, set_seed, infer_device, evaluate
from train_cross_domain_baseline import (
    load_or_create_source_split, build_source_stage_data, load_or_create_target_partition,
)
from train_cd_mambatt_v1 import build_target_cd_data, fit_source_stage

def endless(loader):
    while True:
        for b in loader:
            yield b

def make_args():
    return argparse.Namespace(
        d_model=None, d_state=16, d_conv=8, expand=2, num_mamba_layers=1,
        num_transformer_layers=3, num_heads=7, dropout=0.5, dim_feedforward=84,
        transformer_impl="custom", transformer_norm_mode="pre",
        transformer_inner_dropout=0.0, mamba_block_mode="bare",
        spd_gate_init_bias=-2.0, spd_predictor_mode="shared_head",
        root="/home/shelterpl/data/CMAPSS", window_size=20, stride=1, rul_clip=125,
        target_scale=1.0, grad_clip_norm=0.0, source_train_ratio=0.8,
        source_split_seed=42, source_split_path=None,
        resample_source_split_per_seed=False, source_normalizer_fit_scope="train_only",
        target_shots=5, target_val_units=10, few_shot_seed=42,
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

seed = 42
device = infer_device("auto")
set_seed(seed)
args = make_args()

out_dir = Path("runs/big_lever_v2/FD001_TO_FD003")
out_dir.mkdir(parents=True, exist_ok=True)
run_dir = out_dir / f"seed_{seed}"
run_dir.mkdir(parents=True, exist_ok=True)

src_full = load_cmapss_split(args.root, "FD001", "train", rul_clip=125)
src_test_raw = load_cmapss_split(args.root, "FD001", "test", rul_clip=125)
tgt_full = load_cmapss_split(args.root, "FD003", "train", rul_clip=125)
tgt_test_raw = load_cmapss_split(args.root, "FD003", "test", rul_clip=125)

src_split = load_or_create_source_split(args, src_full, out_dir, seed)
src_loaders, _ = build_source_stage_data(args, src_full, src_test_raw, src_split)
tgt_part = load_or_create_target_partition(args, tgt_full, out_dir, seed)
tgt_loaders, _ = build_target_cd_data(args, tgt_full, tgt_test_raw, tgt_part)

# Shared source pre-training
model_base = build_model(args, 21).to(device)
s_stage = fit_source_stage(args, seed, model_base, src_loaders, device, run_dir, 1.0, None)
src_ckpt = s_stage["checkpoint"]
src_rmse = evaluate(model_base, src_loaders["test"], device, 1.0)["rmse"]
direct_rmse = evaluate(model_base, tgt_loaders["test"], device, 1.0)["rmse"]
print(f"Source RMSE: {src_rmse:.2f}, Direct: {direct_rmse:.2f}")

mse_fn = nn.MSELoss()
ce_fn = nn.CrossEntropyLoss()

def cd_adapt(name, extra_fn=None, extra_params=None, lr=5e-4, epochs=20):
    set_seed(seed)
    m = build_model(args, 21).to(device)
    m.load_state_dict(torch.load(src_ckpt, map_location=device)["model_state_dict"])
    sh = nn.Linear(21, 3).to(device)
    params = list(m.parameters()) + list(sh.parameters())
    if extra_params:
        params += list(extra_params)
    opt = torch.optim.Adam(params, lr=lr)
    best_v, best_t, best_e = float("inf"), None, -1
    
    for ep in range(1, epochs+1):
        m.train(); sh.train()
        si = endless(src_loaders["train"])
        ti = endless(tgt_loaders["labeled"])
        ui = endless(tgt_loaders["unlabeled"])
        for _ in range(max(len(src_loaders["train"]), len(tgt_loaders["labeled"]))):
            sx,sy = next(si); tx,ty = next(ti); tu,_ = next(ui)
            sx,sy,tx,ty,tu = [t.to(device) for t in [sx,sy,tx,ty,tu]]
            sf = m.forward_features(sx); tf = m.forward_features(tx); uf = m.forward_features(tu)
            loss = (mse_fn(m.predict_from_features(sf), sy) +
                    mse_fn(m.predict_from_features(tf), ty) +
                    0.1 * gaussian_mmd_loss(sf, torch.cat([tf, uf])) +
                    ce_fn(sh(sf), assign_rul_stage_labels(sy, rul_clip=125., num_stages=3)))
            if extra_fn:
                loss = loss + extra_fn(m, sf, tf, uf, sx, tx, tu, sy, ty, sh, ep)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        v = evaluate(m, tgt_loaders["val"], device, 1.0)
        t = evaluate(m, tgt_loaders["test"], device, 1.0)
        if v["rmse"] < best_v:
            best_v, best_t, best_e = v["rmse"], t["rmse"], ep
    print(f"  {name}: RMSE={best_t:.2f} epoch={best_e}")
    return best_t, best_e

results = []

# --- Baseline ---
print("\n--- BASELINE (bare, standard CD) ---")
r, e = cd_adapt("baseline")
results.append(("baseline", r, e))

# --- LEVER A: Self-supervised pretrain then adapt ---
print("\n--- LEVER A: Cross-domain self-supervised pretrain ---")
set_seed(seed)
m_ss = build_model(args, 21).to(device)
# Pretrain: temporal ordering on both domains
src_norm = fit_normalizer(src_full)
tgt_norm = fit_normalizer(tgt_full)
src_pw = build_windows(src_norm.transform(src_full), 20, stride=1)
tgt_pw = build_windows(tgt_norm.transform(tgt_full), 20, stride=1)
src_pl = build_loader(CMAPSSWindowDataset(src_pw), 64, True, 0)
tgt_pl = build_loader(CMAPSSWindowDataset(tgt_pw), 64, True, 0)

opt_ss = torch.optim.Adam(m_ss.parameters(), lr=1e-3)
for ep in range(1, 11):
    m_ss.train()
    si = endless(src_pl); ti = endless(tgt_pl)
    total_l = 0; cnt = 0
    for _ in range(max(len(src_pl), len(tgt_pl))):
        sx, _ = next(si); tx, _ = next(ti)
        sx, tx = sx.to(device), tx.to(device)
        x_all = torch.cat([sx, tx])
        f_fwd = m_ss.forward_features(x_all)
        f_rev = m_ss.forward_features(x_all.flip(1))
        # Temporal loss: forward and reverse should be distinguishable
        l_temp = F.relu(F.cosine_similarity(f_fwd, f_rev, dim=-1) + 0.2).mean()
        # Cross-domain alignment during pretrain
        n_s = sx.shape[0]
        l_align = gaussian_mmd_loss(f_fwd[:n_s], f_fwd[n_s:])
        loss = l_temp + 0.1 * l_align
        opt_ss.zero_grad(set_to_none=True); loss.backward(); opt_ss.step()
        total_l += loss.item() * x_all.shape[0]; cnt += x_all.shape[0]
    print(f"  pretrain ep {ep}: loss={total_l/cnt:.4f}")

# Save pretrained weights, then source fine-tune + CD adapt
torch.save(m_ss.state_dict(), run_dir / "ss_pretrained.pt")
m_ss2 = build_model(args, 21).to(device)
m_ss2.load_state_dict(torch.load(run_dir / "ss_pretrained.pt", map_location=device))
s_ss = fit_source_stage(args, seed, m_ss2, src_loaders, device, run_dir / "leverA", 1.0, None)
# Now adapt from ss-pretrained source model
src_ckpt_ss = s_ss["checkpoint"]
set_seed(seed)
m_cd_ss = build_model(args, 21).to(device)
m_cd_ss.load_state_dict(torch.load(src_ckpt_ss, map_location=device)["model_state_dict"])
sh_ss = nn.Linear(21, 3).to(device)
opt_cd = torch.optim.Adam(list(m_cd_ss.parameters()) + list(sh_ss.parameters()), lr=5e-4)
best_v, best_t, best_e = float("inf"), None, -1
for ep in range(1, 21):
    m_cd_ss.train(); sh_ss.train()
    si = endless(src_loaders["train"]); ti = endless(tgt_loaders["labeled"]); ui = endless(tgt_loaders["unlabeled"])
    for _ in range(max(len(src_loaders["train"]), len(tgt_loaders["labeled"]))):
        sx,sy = next(si); tx,ty = next(ti); tu,_ = next(ui)
        sx,sy,tx,ty,tu = [t.to(device) for t in [sx,sy,tx,ty,tu]]
        sf = m_cd_ss.forward_features(sx); tf = m_cd_ss.forward_features(tx); uf = m_cd_ss.forward_features(tu)
        loss = (mse_fn(m_cd_ss.predict_from_features(sf), sy) + mse_fn(m_cd_ss.predict_from_features(tf), ty) +
                0.1 * gaussian_mmd_loss(sf, torch.cat([tf, uf])) +
                ce_fn(sh_ss(sf), assign_rul_stage_labels(sy, rul_clip=125., num_stages=3)))
        opt_cd.zero_grad(set_to_none=True); loss.backward(); opt_cd.step()
    v = evaluate(m_cd_ss, tgt_loaders["val"], device, 1.0)
    t = evaluate(m_cd_ss, tgt_loaders["test"], device, 1.0)
    if v["rmse"] < best_v:
        best_v, best_t, best_e = v["rmse"], t["rmse"], ep
print(f"  ss_pretrain+cd: RMSE={best_t:.2f} epoch={best_e}")
results.append(("ss_pretrain", best_t, best_e))

# --- LEVER B: Temporal mixup at input level ---
print("\n--- LEVER B: Input-level temporal mixup ---")
def tmixup(m, sf, tf, uf, sx, tx, tu, sy, ty, sh, ep):
    n = min(sx.shape[0], tx.shape[0])
    if n < 2: return torch.tensor(0., device=device)
    lam = torch.rand(n, sx.shape[1], 1, device=device) * 0.4 + 0.3
    mx = lam * sx[:n] + (1-lam) * tx[:n]
    my = lam.mean(dim=(1,2)) * sy[:n] + (1-lam.mean(dim=(1,2))) * ty[:n]
    mf = m.forward_features(mx)
    return 0.3 * mse_fn(m.predict_from_features(mf), my)
r, e = cd_adapt("temporal_mixup", extra_fn=tmixup)
results.append(("temporal_mixup", r, e))

# --- LEVER C: Target augmentation + consistency ---
print("\n--- LEVER C: Target augmentation + consistency ---")
def tgt_aug(m, sf, tf, uf, sx, tx, tu, sy, ty, sh, ep):
    noise = 0.1 * torch.randn_like(tx)
    f_aug = m.forward_features(tx + noise)
    l_aug = 0.3 * mse_fn(m.predict_from_features(f_aug), ty)
    with torch.no_grad():
        p_clean = m.predict_from_features(tf).detach()
    l_con = 0.1 * mse_fn(m.predict_from_features(f_aug), p_clean)
    return l_aug + l_con
r, e = cd_adapt("target_augment", extra_fn=tgt_aug)
results.append(("target_augment", r, e))

# --- LEVER D: Higher LR + cosine schedule ---
print("\n--- LEVER D: LR=2e-3 + cosine ---")
r, e = cd_adapt("lr2e3", lr=2e-3, epochs=20)
results.append(("lr_2e-3", r, e))

# --- Summary ---
print("\n" + "="*60)
print(f"{'Method':<25s} {'RMSE':>7s} {'Ep':>4s} {'Delta':>7s}")
print("-"*45)
base = results[0][1]
for name, rmse, ep in results:
    print(f"{name:<25s} {rmse:7.2f} {ep:4d} {rmse-base:+7.2f}")
print(f"\nPrior refs: v2=23.34  SPD_best=20.98  baseline_d21=22.88  d_model_42=22.53")
