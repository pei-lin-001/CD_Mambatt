"""
Probe: does domain drift accumulate in Mamba hidden states?
If yes, SSDA (state-space domain alignment) has a chance of working.
Also tests: does stage-conditional alignment outperform global alignment?
"""
import torch, json, argparse, numpy as np
from pathlib import Path
from _pathfix import ensure_repo_root

ensure_repo_root()

from cd_mambatt.data import load_cmapss_split, fit_normalizer, CMAPSSWindowDataset, build_windows
from cd_mambatt.pseudo_labeling import assign_rul_stage_labels
from cd_mambatt.losses.mmd import gaussian_mmd_loss
from train_supervised import build_model, build_loader, set_seed, infer_device

set_seed(42)
device = infer_device("auto")

# Load trained SPD model
ckpt = torch.load("runs/cd_mambatt_v3_spd_fullmatch/FD001_TO_FD003/seed_42/cd_stage/best.pt", map_location=device)
args = argparse.Namespace(
    d_model=None, d_state=16, d_conv=8, expand=2, num_mamba_layers=1,
    num_transformer_layers=3, num_heads=7, dropout=0.5, dim_feedforward=84,
    transformer_impl="custom", transformer_norm_mode="pre",
    transformer_inner_dropout=0.0, mamba_block_mode="dd_spd",
    spd_gate_init_bias=-2.0, spd_predictor_mode="shared_head",
)
model = build_model(args, 21).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Load data (test sets, last-window only)
src_train = load_cmapss_split("/home/shelterpl/data/CMAPSS", "FD001", "train", rul_clip=125)
tgt_train = load_cmapss_split("/home/shelterpl/data/CMAPSS", "FD003", "train", rul_clip=125)
src_test = load_cmapss_split("/home/shelterpl/data/CMAPSS", "FD001", "test", rul_clip=125)
tgt_test = load_cmapss_split("/home/shelterpl/data/CMAPSS", "FD003", "test", rul_clip=125)

src_norm = fit_normalizer(src_train)
tgt_norm = fit_normalizer(tgt_train)
src_w = build_windows(src_norm.transform(src_test), 20, stride=1)
tgt_w = build_windows(tgt_norm.transform(tgt_test), 20, stride=1)
src_loader = build_loader(CMAPSSWindowDataset(src_w), 256, False, 0)
tgt_loader = build_loader(CMAPSSWindowDataset(tgt_w), 256, False, 0)

# ===== PROBE 1: Hidden state domain drift across timesteps =====
print("=" * 70)
print("PROBE 1: Hidden state MMD at each timestep (domain drift analysis)")
print("=" * 70)

import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
from einops import rearrange

def collect_hidden_states(model, loader, device, max_batches=5):
    """Run DDMamba manually to collect per-timestep hidden states."""
    dd_block = model.mamba_blocks[0]
    all_h_per_step = [[] for _ in range(20)]  # W=20 timesteps
    all_ruls = []
    
    with torch.no_grad():
        for batch_idx, (windows, targets) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            x = model.input_proj(windows.to(device))
            B, L, D = x.shape
            
            # Replicate DDMamba forward to get hidden states
            xz = rearrange(
                dd_block.in_proj.weight @ rearrange(x, "b l d -> d (b l)"),
                "d (b l) -> b d l", l=L,
            )
            if dd_block.in_proj.bias is not None:
                xz = xz + rearrange(dd_block.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
            x_part, z = xz.chunk(2, dim=1)
            x_conv = dd_block.act(dd_block.conv1d(x_part)[..., :L])
            
            # Get combined selectivity
            sel = dd_block._project_selectivity(x_conv, batch=B, seqlen=L)
            dt = sel["dt_combined"]
            B_val = sel["B_combined"]
            C_val = sel["C_combined"]
            
            # Manual SSM scan to collect hidden states
            A = -torch.exp(dd_block.A_log.float())
            delta_bias = dd_block.dt_proj_inv.bias.float() if dd_block.dt_proj_inv.bias is not None else None
            
            dt_with_bias = dt.float()
            if delta_bias is not None:
                dt_with_bias = dt_with_bias + delta_bias.unsqueeze(0).unsqueeze(-1)
            dt_soft = F.softplus(dt_with_bias)
            
            d_inner = A.shape[0]
            d_state = A.shape[1]
            h = torch.zeros(B, d_inner, d_state, device=device)
            
            for k in range(L):
                dA = torch.exp(dt_soft[:, :, k].unsqueeze(-1) * A.unsqueeze(0))
                if B_val.dim() == 3:  # (B, N, L)
                    dB_u = dt_soft[:, :, k].unsqueeze(-1) * B_val[:, :, k].unsqueeze(1) * x_conv[:, :, k].unsqueeze(-1)
                else:
                    dB_u = dt_soft[:, :, k].unsqueeze(-1) * B_val[:, :, k].unsqueeze(1) * x_conv[:, :, k].unsqueeze(-1)
                h = dA * h + dB_u
                # Flatten h to (B, d_inner*d_state) for MMD
                all_h_per_step[k].append(h.reshape(B, -1).cpu())
            
            all_ruls.append(targets)
    
    return [torch.cat(hs, dim=0) for hs in all_h_per_step], torch.cat(all_ruls)

print("Collecting source hidden states...")
src_h_steps, src_ruls = collect_hidden_states(model, src_loader, device)
print("Collecting target hidden states...")
tgt_h_steps, tgt_ruls = collect_hidden_states(model, tgt_loader, device)

print(f"\nSamples: source={src_h_steps[0].shape[0]}, target={tgt_h_steps[0].shape[0]}")
print(f"Hidden state dim per step: {src_h_steps[0].shape[1]} (d_inner={42} × d_state={16} = {42*16})")

# Compute MMD at each timestep
print("\nMMD between source and target hidden states at each timestep:")
print(f"{'Step':>4s}  {'MMD':>10s}  {'Relative':>10s}")
mmds = []
for k in range(20):
    # Subsample to 200 for speed
    n = min(200, src_h_steps[k].shape[0], tgt_h_steps[k].shape[0])
    mmd_val = gaussian_mmd_loss(
        src_h_steps[k][:n].cuda(), 
        tgt_h_steps[k][:n].cuda(),
        sigmas=(0.1, 0.5, 1.0, 2.0, 5.0),  # adjusted for high-dim
    ).item()
    mmds.append(mmd_val)

mmd_first = mmds[0] if mmds[0] > 1e-8 else 1e-8
for k, mmd_val in enumerate(mmds):
    print(f"  {k+1:2d}    {mmd_val:.6f}    {mmd_val/mmd_first:.2f}x")

print(f"\nDrift ratio (step20 / step1): {mmds[-1]/mmd_first:.2f}x")
if mmds[-1] > mmds[0] * 1.5:
    print(">>> DOMAIN DRIFT CONFIRMED: MMD increases over timesteps <<<")
    print(">>> SSDA (hidden state alignment) has potential <<<")
else:
    print(">>> No significant drift detected <<<")

# ===== PROBE 2: Stage-conditional vs global MMD on inv features =====
print("\n" + "=" * 70)
print("PROBE 2: Stage-conditional vs global MMD on invariant features")
print("=" * 70)

# Collect inv features with RUL labels
all_inv_src, all_inv_tgt = [], []
all_rul_src, all_rul_tgt = [], []
with torch.no_grad():
    for w, t in src_loader:
        aux = model.forward_features_with_aux(w.to(device))
        all_inv_src.append(aux["invariant_features"].cpu())
        all_rul_src.append(t)
    for w, t in tgt_loader:
        aux = model.forward_features_with_aux(w.to(device))
        all_inv_tgt.append(aux["invariant_features"].cpu())
        all_rul_tgt.append(t)

inv_s = torch.cat(all_inv_src)
inv_t = torch.cat(all_inv_tgt)
rul_s = torch.cat(all_rul_src)
rul_t = torch.cat(all_rul_tgt)
stage_s = assign_rul_stage_labels(rul_s, rul_clip=125.0, num_stages=3)
stage_t = assign_rul_stage_labels(rul_t, rul_clip=125.0, num_stages=3)

# Global MMD
n = min(500, len(inv_s), len(inv_t))
global_mmd = gaussian_mmd_loss(inv_s[:n].cuda(), inv_t[:n].cuda()).item()
print(f"\nGlobal inv-MMD: {global_mmd:.6f}")

# Per-stage MMD
stage_names = ["late (RUL<42)", "mid (42-83)", "early (RUL>83)"]
print("\nPer-stage inv-MMD:")
for s in range(3):
    mask_s = stage_s == s
    mask_t = stage_t == s
    if mask_s.sum() > 10 and mask_t.sum() > 10:
        ns = min(200, mask_s.sum().item(), mask_t.sum().item())
        stage_mmd = gaussian_mmd_loss(
            inv_s[mask_s][:ns].cuda(),
            inv_t[mask_t][:ns].cuda(),
        ).item()
        print(f"  {stage_names[s]:20s}: MMD = {stage_mmd:.6f}  (src_n={mask_s.sum()}, tgt_n={mask_t.sum()})")

print("\nDone.")
