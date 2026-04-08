"""Best SPD config on FD001->FD004 (harder task: single→multi condition+fault)."""
import torch, argparse, numpy as np
import torch.nn.functional as F
from pathlib import Path
from torch import nn
from _pathfix import ensure_repo_root

ensure_repo_root()

from cd_mambatt.data import load_cmapss_split
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

def make_args(task="FD001_TO_FD004"):
    return argparse.Namespace(
        d_model=None, d_state=16, d_conv=8, expand=2, num_mamba_layers=1,
        num_transformer_layers=3, num_heads=7, dropout=0.5, dim_feedforward=84,
        transformer_impl="custom", transformer_norm_mode="pre",
        transformer_inner_dropout=0.0, mamba_block_mode="dd_spd",
        spd_gate_init_bias=-2.0, spd_predictor_mode="shared_head",
        root="/home/shelterpl/data/CMAPSS", task=task,
        source_subset=None, target_subset=None,
        window_size=20, stride=1, rul_clip=125,
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

def run_one(seed, src_sub, tgt_sub, task_name, device):
    set_seed(seed)
    args = make_args(task_name)
    args.few_shot_seed = seed
    out_dir = Path(f"runs/spd_best_{task_name.lower()}/{task_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    sf = load_cmapss_split(args.root, src_sub, "train", rul_clip=125)
    st = load_cmapss_split(args.root, src_sub, "test", rul_clip=125)
    tf = load_cmapss_split(args.root, tgt_sub, "train", rul_clip=125)
    tt = load_cmapss_split(args.root, tgt_sub, "test", rul_clip=125)

    ss = load_or_create_source_split(args, sf, out_dir, seed)
    sl, _ = build_source_stage_data(args, sf, st, ss)
    tp = load_or_create_target_partition(args, tf, out_dir, seed)
    tl, _ = build_target_cd_data(args, tf, tt, tp)

    model = build_model(args, 21).to(device)
    src_stg = fit_source_stage(args, seed, model, sl, device, run_dir, 1.0, None)
    direct = evaluate(model, tl["test"], device, 1.0)["rmse"]

    cd = build_model(args, 21).to(device)
    cd.load_state_dict(torch.load(src_stg["checkpoint"], map_location=device)["model_state_dict"])
    sh = nn.Linear(21, 3).to(device)
    sc = nn.Linear(21, 2).to(device)
    opt = torch.optim.Adam(list(cd.parameters())+list(sh.parameters())+list(sc.parameters()), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=1e-5)
    mse=nn.MSELoss(); ce=nn.CrossEntropyLoss()

    bv, bt, be = float("inf"), None, -1
    for ep in range(1, 21):
        cd.train(); sh.train(); sc.train()
        si=endless(sl["train"]); ti=endless(tl["labeled"]); ui=endless(tl["unlabeled"])
        for _ in range(max(len(sl["train"]), len(tl["labeled"]))):
            sx,sy=next(si);tx,ty=next(ti);tu,_=next(ui)
            sx,sy,tx,ty,tu=[t.to(device) for t in [sx,sy,tx,ty,tu]]
            sa=cd.forward_features_with_aux(sx);ta=cd.forward_features_with_aux(tx);ua=cd.forward_features_with_aux(tu)
            l=mse(cd.predict_from_features(sa["features"]),sy)+mse(cd.predict_from_features(ta["features"]),ty)
            l+=0.1*gaussian_mmd_loss(sa["features"],torch.cat([ta["features"],ua["features"]]))
            l+=ce(sh(sa["features"]),assign_rul_stage_labels(sy,rul_clip=125.,num_stages=3))
            l+=0.1*gaussian_mmd_loss(sa["domain_features"],torch.cat([ta["domain_features"],ua["domain_features"]]))
            ss_f=sa.get("specific_features");ts_f=ta.get("specific_features")
            if ss_f is not None and ts_f is not None:
                sp=torch.cat([ss_f,ts_f])
                dl=torch.cat([torch.zeros(ss_f.shape[0],dtype=torch.long,device=device),torch.ones(ts_f.shape[0],dtype=torch.long,device=device)])
                l+=0.1*ce(sc(sp),dl)
            opt.zero_grad(set_to_none=True);l.backward();opt.step()
        sched.step()
        v=evaluate(cd,tl["val"],device,1.0);t=evaluate(cd,tl["test"],device,1.0)
        if v["rmse"]<bv: bv,bt,be=v["rmse"],t["rmse"],ep
    print(f"  seed {seed}: direct={direct:.2f} adapted={bt:.2f} epoch={be}")
    return bt

device = infer_device("auto")

# FD001 -> FD004
print("=== FD001 -> FD004 (3 seeds) ===")
r1 = [run_one(s, "FD001", "FD004", "FD001_TO_FD004", device) for s in [42,43,44]]
print(f"FD001->FD004: {np.mean(r1):.4f} ± {np.std(r1):.4f}")
print(f"v2 ref (5-shot): 24.3449")

# FD003 -> FD001
print("\n=== FD003 -> FD001 (3 seeds) ===")
r2 = [run_one(s, "FD003", "FD001", "FD003_TO_FD001", device) for s in [42,43,44]]
print(f"FD003->FD001: {np.mean(r2):.4f} ± {np.std(r2):.4f}")
print(f"v2 ref (5-shot): 19.8137")

print("\n=== SUMMARY ===")
print(f"FD001->FD003: 20.37 ± 0.47 (from earlier)")
print(f"FD001->FD004: {np.mean(r1):.4f} ± {np.std(r1):.4f}")
print(f"FD003->FD001: {np.mean(r2):.4f} ± {np.std(r2):.4f}")
