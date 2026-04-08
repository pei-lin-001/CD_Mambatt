"""SPD diagnostic: gate behavior + domain separability analysis."""
import json, torch, numpy as np
from pathlib import Path
from _pathfix import ensure_repo_root

ensure_repo_root()

from cd_mambatt.data import load_cmapss_split, fit_normalizer, CMAPSSWindowDataset, build_windows
from cd_mambatt.cross_domain import resolve_cross_domain_task
from cd_mambatt.pseudo_labeling import assign_rul_stage_labels
from train_supervised import build_loader, build_model, set_seed, infer_device

set_seed(42)
device = infer_device("auto")

# Load the trained SPD model
ckpt_dir = Path("runs/cd_mambatt_v3_spd_fullmatch/FD001_TO_FD003/seed_42")
cd_ckpt = torch.load(ckpt_dir / "cd_stage/best.pt", map_location=device)

# Build model with SPD
import argparse
args = argparse.Namespace(
    d_model=None, d_state=16, d_conv=8, expand=2, num_mamba_layers=1,
    num_transformer_layers=3, num_heads=7, dropout=0.5, dim_feedforward=84,
    transformer_impl="custom", transformer_norm_mode="pre",
    transformer_inner_dropout=0.0, mamba_block_mode="dd_spd",
    spd_gate_init_bias=-2.0, spd_predictor_mode="shared_head",
)
model = build_model(args, 21).to(device)
model.load_state_dict(cd_ckpt["model_state_dict"])
model.eval()

# Load source and target data
task = resolve_cross_domain_task("FD001_TO_FD003")
source_train = load_cmapss_split("/home/shelterpl/data/CMAPSS", "FD001", "train", rul_clip=125)
source_test = load_cmapss_split("/home/shelterpl/data/CMAPSS", "FD001", "test", rul_clip=125)
target_train = load_cmapss_split("/home/shelterpl/data/CMAPSS", "FD003", "train", rul_clip=125)
target_test = load_cmapss_split("/home/shelterpl/data/CMAPSS", "FD003", "test", rul_clip=125)


src_norm = fit_normalizer(source_train)
tgt_norm = fit_normalizer(target_train)

src_test_normed = src_norm.transform(source_test)
tgt_test_normed = tgt_norm.transform(target_test)

src_windows = build_windows(src_test_normed, window_size=20, stride=1)
tgt_windows = build_windows(tgt_test_normed, window_size=20, stride=1)

src_ds = CMAPSSWindowDataset(src_windows)
tgt_ds = CMAPSSWindowDataset(tgt_windows)
src_loader = build_loader(src_ds, 64, False, 0)
tgt_loader = build_loader(tgt_ds, 64, False, 0)

# ===== DIAGNOSTIC 1: Gate behavior across degradation stages =====
print("=" * 60)
print("DIAGNOSTIC 1: Gate behavior across degradation stages")
print("=" * 60)

all_gates_src = []
all_ruls_src = []
all_gates_tgt = []
all_ruls_tgt = []

with torch.no_grad():
    for windows, targets in src_loader:
        windows = windows.to(device)
        aux = model.forward_features_with_aux(windows)
        # gate_sequence: (B, L, 1), take mean over L
        gate_per_sample = aux["gate_sequence"].mean(dim=1).squeeze(-1).cpu()  # (B,)
        all_gates_src.append(gate_per_sample)
        all_ruls_src.append(targets)

    for windows, targets in tgt_loader:
        windows = windows.to(device)
        aux = model.forward_features_with_aux(windows)
        gate_per_sample = aux["gate_sequence"].mean(dim=1).squeeze(-1).cpu()
        all_gates_tgt.append(gate_per_sample)
        all_ruls_tgt.append(targets)

gates_src = torch.cat(all_gates_src)
ruls_src = torch.cat(all_ruls_src)
gates_tgt = torch.cat(all_gates_tgt)
ruls_tgt = torch.cat(all_ruls_tgt)

# Stage assignment
stages_src = assign_rul_stage_labels(ruls_src, rul_clip=125.0, num_stages=3)
stages_tgt = assign_rul_stage_labels(ruls_tgt, rul_clip=125.0, num_stages=3)

stage_names = ["late (RUL<42)", "mid (42-83)", "early (RUL>83)"]

print("\nSource domain (FD001) gate statistics by stage:")
for s in range(3):
    mask = stages_src == s
    if mask.sum() > 0:
        g = gates_src[mask]
        print(f"  {stage_names[s]:20s}: mean={g.mean():.4f} std={g.std():.4f} n={mask.sum()}")

print("\nTarget domain (FD003) gate statistics by stage:")
for s in range(3):
    mask = stages_tgt == s
    if mask.sum() > 0:
        g = gates_tgt[mask]
        print(f"  {stage_names[s]:20s}: mean={g.mean():.4f} std={g.std():.4f} n={mask.sum()}")

print(f"\nGate overall: source={gates_src.mean():.4f} target={gates_tgt.mean():.4f}")
print(f"Gate range: source=[{gates_src.min():.4f}, {gates_src.max():.4f}] target=[{gates_tgt.min():.4f}, {gates_tgt.max():.4f}]")

# ===== DIAGNOSTIC 2: Domain separability of inv vs combined features =====
print("\n" + "=" * 60)
print("DIAGNOSTIC 2: Domain separability (inv vs combined features)")
print("=" * 60)

all_inv_src = []
all_comb_src = []
all_inv_tgt = []
all_comb_tgt = []

with torch.no_grad():
    for windows, targets in src_loader:
        windows = windows.to(device)
        aux = model.forward_features_with_aux(windows)
        all_inv_src.append(aux["invariant_features"].cpu())
        all_comb_src.append(aux["features"].cpu())

    for windows, targets in tgt_loader:
        windows = windows.to(device)
        aux = model.forward_features_with_aux(windows)
        all_inv_tgt.append(aux["invariant_features"].cpu())
        all_comb_tgt.append(aux["features"].cpu())

inv_src = torch.cat(all_inv_src)
inv_tgt = torch.cat(all_inv_tgt)
comb_src = torch.cat(all_comb_src)
comb_tgt = torch.cat(all_comb_tgt)

# Train a simple linear classifier to distinguish domains
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def domain_separability(feat_src, feat_tgt, name):
    X = np.concatenate([feat_src.numpy(), feat_tgt.numpy()], axis=0)
    y = np.concatenate([np.zeros(len(feat_src)), np.ones(len(feat_tgt))])
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"  {name:30s}: domain accuracy = {scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean()

print("\nLinear domain classification accuracy (lower = more domain-invariant):")
acc_inv = domain_separability(inv_src, inv_tgt, "invariant features")
acc_comb = domain_separability(comb_src, comb_tgt, "combined features")
acc_diff = domain_separability(comb_src - inv_src, comb_tgt - inv_tgt, "specific features (comb-inv)")

print(f"\nDelta (combined - invariant): {acc_comb - acc_inv:+.4f}")
if acc_inv < acc_comb:
    print(">>> Invariant features ARE more domain-invariant than combined <<<")
else:
    print(">>> WARNING: Invariant features are NOT more domain-invariant <<<")

# Also compute MMD between domains for each feature type
def compute_mmd(src, tgt):
    from cd_mambatt.losses.mmd import gaussian_mmd_loss
    return gaussian_mmd_loss(src.cuda(), tgt.cuda()).item()

print(f"\nMMD between domains:")
print(f"  invariant features: {compute_mmd(inv_src, inv_tgt):.6f}")
print(f"  combined features:  {compute_mmd(comb_src, comb_tgt):.6f}")
print(f"  specific features:  {compute_mmd(comb_src - inv_src, comb_tgt - inv_tgt):.6f}")

print("\nDone.")
