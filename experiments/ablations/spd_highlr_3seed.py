"""SPD + inv-MMD + spec-domain + LR=2e-3 + cosine schedule, 3-seed."""
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

def make_args():
    return argparse.Namespace(
        d_model=None, d_state=16, d_conv=8, expand=2, num_mamba_layers=1,
        num_transformer_layers=3, num_heads=7, dropout=0.5, dim_feedforward=84,
        transformer_impl="custom", transformer_norm_mode="pre",
        transformer_inner_dropout=0.0, mamba_block_mode="dd_spd",
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

def run_one_seed(seed, device):
    set_seed(seed)
    args = make_args()
    args.few_shot_seed = seed

    out_dir = Path("runs/spd_highlr_3seed/FD001_TO_FD003")
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

    model = build_model(args, 21).to(device)
    src_stage = fit_source_stage(args, seed, model, sl, device, run_dir, 1.0, None)
    direct = evaluate(model, tl["test"], device, 1.0)["rmse"]

    cd = build_model(args, 21).to(device)
    cd.load_state_dict(torch.load(src_stage["checkpoint"], map_location=device)["model_state_dict"])

    stage_head = nn.Linear(21, 3).to(device)
    spec_clf = nn.Linear(21, 2).to(device)
    all_params = list(cd.parameters()) + list(stage_head.parameters()) + list(spec_clf.parameters())
    optimizer = torch.optim.Adam(all_params, lr=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
    mse = nn.MSELoss(); ce = nn.CrossEntropyLoss()

    best_val, best_test, best_ep = float("inf"), None, -1
    for ep in range(1, 21):
        cd.train(); stage_head.train(); spec_clf.train()
        si = endless(sl["train"]); ti = endless(tl["labeled"]); ui = endless(tl["unlabeled"])
        for _ in range(max(len(sl["train"]), len(tl["labeled"]))):
            sx,sy = next(si); tx,ty = next(ti); tu,_ = next(ui)
            sx,sy,tx,ty,tu = [t.to(device) for t in [sx,sy,tx,ty,tu]]

            sa = cd.forward_features_with_aux(sx)
            ta = cd.forward_features_with_aux(tx)
            ua = cd.forward_features_with_aux(tu)

            l_src = mse(cd.predict_from_features(sa["features"]), sy)
            l_tgt = mse(cd.predict_from_features(ta["features"]), ty)
            l_stage = ce(stage_head(sa["features"]),
                         assign_rul_stage_labels(sy, rul_clip=125., num_stages=3))
            l_mmd = gaussian_mmd_loss(sa["features"], torch.cat([ta["features"], ua["features"]]))
            l_inv = gaussian_mmd_loss(sa["domain_features"],
                                      torch.cat([ta["domain_features"], ua["domain_features"]]))

            # Spec domain-predictive
            l_spec = torch.tensor(0., device=device)
            ss_f = sa.get("specific_features")
            ts_f = ta.get("specific_features")
            if ss_f is not None and ts_f is not None:
                spec_all = torch.cat([ss_f, ts_f])
                dl = torch.cat([torch.zeros(ss_f.shape[0], dtype=torch.long, device=device),
                                torch.ones(ts_f.shape[0], dtype=torch.long, device=device)])
                l_spec = 0.1 * ce(spec_clf(spec_all), dl)

            loss = l_src + l_tgt + 0.1*l_mmd + 1.0*l_stage + 0.1*l_inv + l_spec
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()

        scheduler.step()
        v = evaluate(cd, tl["val"], device, 1.0)
        t = evaluate(cd, tl["test"], device, 1.0)
        if v["rmse"] < best_val:
            best_val, best_test, best_ep = v["rmse"], t["rmse"], ep

    print(f"  seed {seed}: direct={direct:.2f} adapted={best_test:.2f} epoch={best_ep}")
    return best_test

def main():
    device = infer_device("auto")
    print("SPD + inv-MMD + spec-domain + LR=2e-3 + cosine, 3 seeds")
    rs = []
    for s in [42, 43, 44]:
        rs.append(run_one_seed(s, device))
    m, sd = np.mean(rs), np.std(rs)
    print(f"\nRESULT: {m:.4f} ± {sd:.4f}")
    print(f"Prior best SPD (lr=5e-4): 20.98 ± 0.69")
    print(f"v2 reference:             21.13 ± 1.59")
    for s, r in zip([42,43,44], rs):
        print(f"  seed {s}: {r:.2f}")

if __name__ == "__main__":
    main()
