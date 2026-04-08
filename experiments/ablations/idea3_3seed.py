"""IDEA 3 (Spec domain-predictive) - 3-seed validation."""
import torch, argparse, numpy as np
import torch.nn.functional as F
from pathlib import Path
from torch import nn
from _pathfix import ensure_repo_root

ensure_repo_root()

from cd_mambatt.data import load_cmapss_split, fit_normalizer, CMAPSSWindowDataset, build_windows, select_units
from cd_mambatt.cross_domain import resolve_cross_domain_task
from cd_mambatt.losses.mmd import gaussian_mmd_loss
from cd_mambatt.pseudo_labeling import assign_rul_stage_labels
from train_supervised import build_model, build_loader, set_seed, infer_device, evaluate
from train_cross_domain_baseline import load_or_create_source_split, build_source_stage_data, load_or_create_target_partition
from train_cd_mambatt_v1 import build_target_cd_data, fit_source_stage

def endless(loader):
    while True:
        for b in loader:
            yield b

def run_seed(seed, device):
    set_seed(seed)
    model_args = argparse.Namespace(
        d_model=None, d_state=16, d_conv=8, expand=2, num_mamba_layers=1,
        num_transformer_layers=3, num_heads=7, dropout=0.5, dim_feedforward=84,
        transformer_impl="custom", transformer_norm_mode="pre",
        transformer_inner_dropout=0.0, mamba_block_mode="dd_spd",
        spd_gate_init_bias=-2.0, spd_predictor_mode="shared_head",
        root="/home/shelterpl/data/CMAPSS", window_size=20, stride=1, rul_clip=125,
        target_scale=1.0, grad_clip_norm=0.0, source_train_ratio=0.8,
        source_split_seed=42, source_split_path=None,
        resample_source_split_per_seed=False, source_normalizer_fit_scope="train_only",
        target_shots=5, target_val_units=10, few_shot_seed=seed,
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
    
    output_dir = Path("runs/idea3_spec_domain_3seed/FD001_TO_FD003")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    src_full = load_cmapss_split(model_args.root, "FD001", "train", rul_clip=125)
    src_test = load_cmapss_split(model_args.root, "FD001", "test", rul_clip=125)
    tgt_full = load_cmapss_split(model_args.root, "FD003", "train", rul_clip=125)
    tgt_test = load_cmapss_split(model_args.root, "FD003", "test", rul_clip=125)
    
    src_split = load_or_create_source_split(model_args, src_full, output_dir, seed)
    src_loaders, _ = build_source_stage_data(model_args, src_full, src_test, src_split)
    
    model = build_model(model_args, 21).to(device)
    source_stage = fit_source_stage(model_args, seed, model, src_loaders, device, run_dir, 1.0, None)
    
    tgt_part = load_or_create_target_partition(model_args, tgt_full, output_dir, seed)
    tgt_loaders, _ = build_target_cd_data(model_args, tgt_full, tgt_test, tgt_part)
    
    direct = evaluate(model, tgt_loaders["test"], device, 1.0)
    
    # Fresh model for adaptation
    cd_model = build_model(model_args, 21).to(device)
    cd_model.load_state_dict(torch.load(source_stage["checkpoint"], map_location=device)["model_state_dict"])
    
    stage_head = nn.Linear(21, 3).to(device)
    spec_domain_clf = nn.Linear(21, 2).to(device)
    
    optimizer = torch.optim.Adam(
        list(cd_model.parameters()) + list(stage_head.parameters()) + list(spec_domain_clf.parameters()),
        lr=5e-4,
    )
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    
    best_val = float("inf")
    best_test = None
    best_ep = -1
    
    for epoch in range(1, 21):
        cd_model.train(); stage_head.train(); spec_domain_clf.train()
        src_iter = endless(src_loaders["train"])
        tgt_lab_iter = endless(tgt_loaders["labeled"])
        tgt_unlab_iter = endless(tgt_loaders["unlabeled"])
        n_steps = max(len(src_loaders["train"]), len(tgt_loaders["labeled"]))
        
        for _ in range(n_steps):
            sx, sy = next(src_iter); tx, ty = next(tgt_lab_iter); tu, _ = next(tgt_unlab_iter)
            sx, sy = sx.to(device), sy.to(device)
            tx, ty = tx.to(device), ty.to(device)
            tu = tu.to(device)
            
            sa = cd_model.forward_features_with_aux(sx)
            ta = cd_model.forward_features_with_aux(tx)
            ua = cd_model.forward_features_with_aux(tu)
            
            l_src = mse(cd_model.predict_from_features(sa["features"]), sy)
            l_tgt = mse(cd_model.predict_from_features(ta["features"]), ty)
            l_stage = ce(stage_head(sa["features"]),
                         assign_rul_stage_labels(sy, rul_clip=125.0, num_stages=3))
            
            tgt_all_f = torch.cat([ta["features"], ua["features"]], dim=0)
            l_mmd = gaussian_mmd_loss(sa["features"], tgt_all_f)
            
            tgt_all_inv = torch.cat([ta["domain_features"], ua["domain_features"]], dim=0)
            l_inv_mmd = gaussian_mmd_loss(sa["domain_features"], tgt_all_inv)
            
            # IDEA 3: Spec domain-predictive loss
            spec_s = sa.get("specific_features")
            spec_t = ta.get("specific_features")
            if spec_s is not None and spec_t is not None:
                spec_all = torch.cat([spec_s, spec_t], dim=0)
                dlabels = torch.cat([
                    torch.zeros(spec_s.shape[0], dtype=torch.long, device=device),
                    torch.ones(spec_t.shape[0], dtype=torch.long, device=device),
                ])
                l_spec_domain = 0.1 * ce(spec_domain_clf(spec_all), dlabels)
            else:
                l_spec_domain = torch.tensor(0.0, device=device)
            
            loss = l_src + l_tgt + 0.1*l_mmd + 1.0*l_stage + 0.1*l_inv_mmd + l_spec_domain
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        val = evaluate(cd_model, tgt_loaders["val"], device, 1.0)
        test = evaluate(cd_model, tgt_loaders["test"], device, 1.0)
        if val["rmse"] < best_val:
            best_val = val["rmse"]
            best_test = test["rmse"]
            best_ep = epoch
    
    print(f"  seed {seed}: direct={direct['rmse']:.2f} adapted={best_test:.2f} epoch={best_ep}")
    return {"seed": seed, "direct": direct["rmse"], "adapted": best_test, "epoch": best_ep}

def main():
    device = infer_device("auto")
    print("=" * 60)
    print("IDEA 3: Spec domain-predictive loss - 3-seed validation")
    print("=" * 60)
    
    results = []
    for seed in [42, 43, 44]:
        r = run_seed(seed, device)
        results.append(r)
    
    adapted = [r["adapted"] for r in results]
    mean_r = np.mean(adapted)
    std_r = np.std(adapted)
    
    print(f"\n=== RESULT ===")
    print(f"IDEA 3 mean RMSE: {mean_r:.4f} ± {std_r:.4f}")
    print(f"SPD+inv-MMD ref:  21.0859 ± 0.9814")
    print(f"v2 reference:     21.1291 ± 1.5928")
    for r in results:
        print(f"  seed {r['seed']}: {r['adapted']:.2f}")

if __name__ == "__main__":
    main()
