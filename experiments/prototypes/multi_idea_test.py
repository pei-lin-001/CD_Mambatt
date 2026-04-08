"""
Test multiple innovation ideas in one script.
Each idea modifies the adaptation epoch loop in a minimal way.
All share the same source pre-training and data setup.
"""
import torch, json, argparse, time, numpy as np, copy
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
from train_cross_domain_baseline import load_or_create_source_split, build_source_stage_data, build_target_direct_test_loader, load_or_create_target_partition
from train_cd_mambatt_v1 import build_target_cd_data, fit_source_stage

def endless(loader):
    while True:
        for b in loader:
            yield b

def setup_experiment(seed=42):
    """Shared setup: model, data, source pre-training."""
    set_seed(seed)
    device = infer_device("auto")
    
    model_args = argparse.Namespace(
        d_model=None, d_state=16, d_conv=8, expand=2, num_mamba_layers=1,
        num_transformer_layers=3, num_heads=7, dropout=0.5, dim_feedforward=84,
        transformer_impl="custom", transformer_norm_mode="pre",
        transformer_inner_dropout=0.0, mamba_block_mode="dd_spd",
        spd_gate_init_bias=-2.0, spd_predictor_mode="shared_head",
        root="/home/shelterpl/data/CMAPSS", task="FD001_TO_FD003",
        source_subset=None, target_subset=None,
        window_size=20, stride=1, rul_clip=125, target_scale=1.0,
        grad_clip_norm=0.0, source_train_ratio=0.8, source_split_seed=42,
        source_split_path=None, resample_source_split_per_seed=False,
        source_normalizer_fit_scope="train_only",
        target_shots=5, target_val_units=10, few_shot_seed=seed,
        target_partition_path=None, resample_few_shot_per_seed=True,
        batch_size=64, source_epochs=50, target_epochs=20,
        lr=1e-3, target_lr=5e-4, weight_decay=0.0, target_weight_decay=0.0,
        source_loss_weight=1.0, target_loss_weight=1.0,
        lambda_mmd=0.1, mmd_sigmas="1,2,4,8,16",
        lambda_source_stage=1.0, lambda_pseudo=0.5,
        lambda_contrastive=0.0, lambda_monotonic=0.05,
        contrastive_temperature=0.1,
        disable_contrastive_feature_normalization=False,
        monotonic_margin=0.0, monotonic_pair_gap=1, monotonic_pair_stride=1,
        num_pseudo_stages=3, pseudo_start_quantile=0.5, pseudo_end_quantile=0.9,
        disable_pseudo_feature_normalization=False,
        source_val_all_windows=True, target_val_all_windows=True,
        num_workers=0, max_source_train_batches=None, max_target_train_batches=None,
    )
    
    output_dir = Path(f"runs/multi_idea_test/FD001_TO_FD003")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    source_train_full = load_cmapss_split(model_args.root, "FD001", "train", rul_clip=125)
    source_test_raw = load_cmapss_split(model_args.root, "FD001", "test", rul_clip=125)
    target_train_full = load_cmapss_split(model_args.root, "FD003", "train", rul_clip=125)
    target_test_raw = load_cmapss_split(model_args.root, "FD003", "test", rul_clip=125)
    
    source_split = load_or_create_source_split(model_args, source_train_full, output_dir, seed)
    source_loaders, source_meta = build_source_stage_data(model_args, source_train_full, source_test_raw, source_split)
    
    # Source pre-training (shared across all ideas)
    model = build_model(model_args, 21).to(device)
    source_stage = fit_source_stage(model_args, seed, model, source_loaders, device, run_dir, 1.0, None)
    src_ckpt_path = source_stage["checkpoint"]
    
    target_partition = load_or_create_target_partition(model_args, target_train_full, output_dir, seed)
    target_loaders, target_meta = build_target_cd_data(model_args, target_train_full, target_test_raw, target_partition)
    
    direct_test = evaluate(model, target_loaders["test"], device, 1.0)
    
    return model_args, device, source_loaders, target_loaders, src_ckpt_path, direct_test


def run_adaptation(name, model_args, device, source_loaders, target_loaders, src_ckpt_path, 
                   extra_loss_fn=None, epochs=20):
    """Run one adaptation experiment with an optional extra loss."""
    model = build_model(model_args, 21).to(device)
    ckpt = torch.load(src_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    stage_head = nn.Linear(model.head.in_features, 3).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(stage_head.parameters()), lr=5e-4)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    
    best_val = float("inf")
    best_test_rmse = None
    best_ep = -1
    
    for epoch in range(1, epochs + 1):
        model.train(); stage_head.train()
        src_iter = endless(source_loaders["train"])
        tgt_lab_iter = endless(target_loaders["labeled"])
        tgt_unlab_iter = endless(target_loaders["unlabeled"])
        n_steps = max(len(source_loaders["train"]), len(target_loaders["labeled"]))
        
        for _ in range(n_steps):
            src_x, src_y = next(src_iter)
            tgt_x, tgt_y = next(tgt_lab_iter)
            tgt_u, _ = next(tgt_unlab_iter)
            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x, tgt_y = tgt_x.to(device), tgt_y.to(device)
            tgt_u = tgt_u.to(device)
            
            src_aux = model.forward_features_with_aux(src_x)
            tgt_aux = model.forward_features_with_aux(tgt_x)
            tgt_u_aux = model.forward_features_with_aux(tgt_u)
            
            src_pred = model.predict_from_features(src_aux["features"])
            tgt_pred = model.predict_from_features(tgt_aux["features"])
            l_src = mse(src_pred, src_y)
            l_tgt = mse(tgt_pred, tgt_y)
            
            src_stages = assign_rul_stage_labels(src_y, rul_clip=125.0, num_stages=3)
            l_stage = ce(stage_head(src_aux["features"]), src_stages)
            
            tgt_all_feat = torch.cat([tgt_aux["features"], tgt_u_aux["features"]], dim=0)
            l_mmd = gaussian_mmd_loss(src_aux["features"], tgt_all_feat)
            
            # inv-MMD (baseline SPD component)
            tgt_all_inv = torch.cat([tgt_aux["domain_features"], tgt_u_aux["domain_features"]], dim=0)
            l_inv_mmd = gaussian_mmd_loss(src_aux["domain_features"], tgt_all_inv)
            
            loss = l_src + l_tgt + 0.1 * l_mmd + 1.0 * l_stage + 0.5 * ce(stage_head(tgt_aux["features"]),
                assign_rul_stage_labels(tgt_y, rul_clip=125.0, num_stages=3)) * 0 + 0.1 * l_inv_mmd
            
            # Pseudo loss (simplified)
            loss = loss + 0.05 * torch.tensor(0.0, device=device)  # placeholder for monotonic
            
            # Add extra loss if provided
            if extra_loss_fn is not None:
                extra = extra_loss_fn(model, src_aux, tgt_aux, tgt_u_aux, src_y, tgt_y, stage_head, epoch)
                loss = loss + extra
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        val = evaluate(model, target_loaders["val"], device, 1.0)
        test = evaluate(model, target_loaders["test"], device, 1.0)
        if val["rmse"] < best_val:
            best_val = val["rmse"]
            best_test_rmse = test["rmse"]
            best_ep = epoch
    
    return {"name": name, "best_epoch": best_ep, "test_rmse": best_test_rmse}


def main():
    print("=" * 70)
    print("MULTI-IDEA RAPID TEST")
    print("=" * 70)
    
    model_args, device, source_loaders, target_loaders, src_ckpt, direct = setup_experiment(42)
    print(f"\nDirect RMSE: {direct['rmse']:.2f}")
    
    results = []
    
    # === IDEA 0: Baseline SPD + inv-MMD (no extra) ===
    print("\n--- IDEA 0: Baseline SPD + inv-MMD ---")
    r = run_adaptation("baseline_spd_invmmd", model_args, device, source_loaders, target_loaders, src_ckpt)
    results.append(r)
    print(f"  {r['name']}: RMSE={r['test_rmse']:.2f} epoch={r['best_epoch']}")

    # === IDEA 1: Stage-conditional inv-MMD ===
    # Align inv features PER degradation stage, not globally
    print("\n--- IDEA 1: Stage-conditional inv-MMD ---")
    def stage_cond_mmd(model, src_aux, tgt_aux, tgt_u_aux, src_y, tgt_y, stage_head, epoch):
        src_inv = src_aux["domain_features"]
        tgt_inv = torch.cat([tgt_aux["domain_features"], tgt_u_aux["domain_features"]], dim=0)
        src_stages = assign_rul_stage_labels(src_y, rul_clip=125.0, num_stages=3)
        # For target unlabeled, use stage_head predictions as pseudo-stages
        with torch.no_grad():
            tgt_u_stage_logits = stage_head(tgt_u_aux["features"])
            tgt_u_stages = tgt_u_stage_logits.argmax(dim=-1)
        tgt_stages = torch.cat([
            assign_rul_stage_labels(tgt_y, rul_clip=125.0, num_stages=3),
            tgt_u_stages
        ], dim=0)
        
        loss = torch.tensor(0.0, device=src_inv.device)
        for s in range(3):
            s_mask = src_stages == s
            t_mask = tgt_stages == s
            if s_mask.sum() >= 2 and t_mask.sum() >= 2:
                loss = loss + gaussian_mmd_loss(src_inv[s_mask], tgt_inv[t_mask])
        return 0.1 * loss
    
    r = run_adaptation("stage_cond_invmmd", model_args, device, source_loaders, target_loaders, src_ckpt,
                       extra_loss_fn=stage_cond_mmd)
    results.append(r)
    print(f"  {r['name']}: RMSE={r['test_rmse']:.2f} epoch={r['best_epoch']}")

    # === IDEA 2: Inv/Spec orthogonality constraint ===
    # Force inv and spec features to be orthogonal -> stronger separation
    print("\n--- IDEA 2: Inv/Spec orthogonality ---")
    def ortho_loss(model, src_aux, tgt_aux, tgt_u_aux, src_y, tgt_y, stage_head, epoch):
        inv_f = src_aux.get("invariant_features")
        spec_f = src_aux.get("specific_features")
        if inv_f is None or spec_f is None:
            return torch.tensor(0.0, device=src_aux["features"].device)
        # Cosine similarity should be 0
        cos_sim = F.cosine_similarity(inv_f, spec_f, dim=-1).abs().mean()
        return 0.5 * cos_sim
    
    r = run_adaptation("ortho_inv_spec", model_args, device, source_loaders, target_loaders, src_ckpt,
                       extra_loss_fn=ortho_loss)
    results.append(r)
    print(f"  {r['name']}: RMSE={r['test_rmse']:.2f} epoch={r['best_epoch']}")

    # === IDEA 3: Domain-specific spec supervision ===
    # Encourage spec features to be GOOD at domain classification (opposite of inv)
    print("\n--- IDEA 3: Spec domain-predictive loss ---")
    spec_domain_clf = nn.Linear(model_args.d_state if model_args.d_model is None else model_args.d_model, 2).to(device).requires_grad_(True)
    # Actually d_model=21
    spec_domain_clf = nn.Linear(21, 2).to(device)
    
    def spec_domain_loss(model, src_aux, tgt_aux, tgt_u_aux, src_y, tgt_y, stage_head, epoch):
        spec_s = src_aux.get("specific_features")
        spec_t = tgt_aux.get("specific_features")
        if spec_s is None or spec_t is None:
            return torch.tensor(0.0, device=src_aux["features"].device)
        spec_all = torch.cat([spec_s, spec_t], dim=0)
        labels = torch.cat([
            torch.zeros(spec_s.shape[0], dtype=torch.long, device=device),
            torch.ones(spec_t.shape[0], dtype=torch.long, device=device),
        ])
        logits = spec_domain_clf(spec_all)
        # We WANT spec to be domain-predictive (no GRL, just normal CE)
        return 0.1 * F.cross_entropy(logits, labels)
    
    r = run_adaptation("spec_domain_predictive", model_args, device, source_loaders, target_loaders, src_ckpt,
                       extra_loss_fn=spec_domain_loss)
    results.append(r)
    print(f"  {r['name']}: RMSE={r['test_rmse']:.2f} epoch={r['best_epoch']}")

    # === IDEA 4: Instance Normalization in Mamba hidden path ===
    # Apply instance norm to the Mamba output before Transformer
    # This removes per-sample style/domain info while keeping content
    print("\n--- IDEA 4: Instance Norm on Mamba output ---")
    instance_norm = nn.InstanceNorm1d(21, affine=True).to(device)
    
    def apply_instance_norm_hook(model, src_aux, tgt_aux, tgt_u_aux, src_y, tgt_y, stage_head, epoch):
        # Can't easily hook into forward here, so we apply IN as a feature transformation loss
        # Minimize the difference between IN(features) and features for the inv path
        # This encourages features to already be instance-normalized
        inv_s = src_aux.get("invariant_features")
        if inv_s is None:
            return torch.tensor(0.0, device=device)
        # Normalize: zero mean, unit var per sample
        inv_normed = (inv_s - inv_s.mean(dim=-1, keepdim=True)) / (inv_s.std(dim=-1, keepdim=True) + 1e-6)
        # The domain MMD on normalized features should be lower
        inv_t = tgt_aux.get("invariant_features")
        if inv_t is None:
            return torch.tensor(0.0, device=device)
        inv_t_normed = (inv_t - inv_t.mean(dim=-1, keepdim=True)) / (inv_t.std(dim=-1, keepdim=True) + 1e-6)
        
        tgt_u_inv = tgt_u_aux.get("invariant_features")
        if tgt_u_inv is not None:
            tgt_u_normed = (tgt_u_inv - tgt_u_inv.mean(dim=-1, keepdim=True)) / (tgt_u_inv.std(dim=-1, keepdim=True) + 1e-6)
            inv_t_normed = torch.cat([inv_t_normed, tgt_u_normed], dim=0)
        
        return 0.1 * gaussian_mmd_loss(inv_normed, inv_t_normed)
    
    r = run_adaptation("instance_norm_invmmd", model_args, device, source_loaders, target_loaders, src_ckpt,
                       extra_loss_fn=apply_instance_norm_hook)
    results.append(r)
    print(f"  {r['name']}: RMSE={r['test_rmse']:.2f} epoch={r['best_epoch']}")

    # === IDEA 5: Mixup across domains ===
    # Create virtual cross-domain samples to smooth the domain boundary
    print("\n--- IDEA 5: Cross-domain mixup ---")
    def mixup_loss(model, src_aux, tgt_aux, tgt_u_aux, src_y, tgt_y, stage_head, epoch):
        src_f = src_aux["features"]
        tgt_f = tgt_aux["features"]
        n = min(src_f.shape[0], tgt_f.shape[0])
        if n < 2:
            return torch.tensor(0.0, device=device)
        lam = torch.rand(n, 1, device=device) * 0.3 + 0.35  # lambda in [0.35, 0.65]
        mixed_f = lam * src_f[:n] + (1 - lam) * tgt_f[:n]
        mixed_y = lam.squeeze() * src_y[:n] + (1 - lam.squeeze()) * tgt_y[:n]
        mixed_pred = model.predict_from_features(mixed_f)
        return 0.3 * F.mse_loss(mixed_pred, mixed_y)
    
    r = run_adaptation("crossdomain_mixup", model_args, device, source_loaders, target_loaders, src_ckpt,
                       extra_loss_fn=mixup_loss)
    results.append(r)
    print(f"  {r['name']}: RMSE={r['test_rmse']:.2f} epoch={r['best_epoch']}")
    
    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY (seed 42, FD001 -> FD003)")
    print("=" * 70)
    print(f"{'Method':<30s} {'RMSE':>8s} {'Epoch':>6s} {'vs Base':>8s}")
    print("-" * 55)
    base_rmse = results[0]["test_rmse"]
    for r in results:
        delta = r["test_rmse"] - base_rmse
        print(f"{r['name']:<30s} {r['test_rmse']:8.2f} {r['best_epoch']:6d} {delta:+8.2f}")
    print(f"\n{'v2 reference (seed42)':<30s} {'23.34':>8s}")
    print(f"{'SPD+inv-MMD matched (seed42)':<30s} {'22.40':>8s}")

if __name__ == "__main__":
    main()
