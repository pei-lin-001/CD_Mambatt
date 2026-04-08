"""
Quick SSDA test: add hidden-state alignment loss to the existing SPD pipeline.
We modify DDMambaBlock to expose hidden state statistics, then add MMD on them.
"""
import torch, json, argparse, time, numpy as np
import torch.nn.functional as F
from pathlib import Path
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader
from _pathfix import ensure_repo_root

ensure_repo_root()

from cd_mambatt.data import (
    load_cmapss_split, fit_normalizer, CMAPSSWindowDataset, build_windows, select_units,
)
from cd_mambatt.cross_domain import resolve_cross_domain_task
from cd_mambatt.losses.mmd import gaussian_mmd_loss
from cd_mambatt.pseudo_labeling import assign_rul_stage_labels, compute_source_stage_statistics, assign_pseudo_stage_labels, SourceStageStatistics
from train_supervised import build_loader, set_seed, infer_device, evaluate
from train_cross_domain_baseline import load_or_create_source_split, build_source_stage_data, build_target_direct_test_loader, load_or_create_target_partition
from train_cd_mambatt_v1 import build_target_cd_data, fit_source_stage


def collect_hidden_state_stats(dd_block, x_input, device):
    """Run DDMamba manually and return hidden state mean/var over timesteps."""
    B, L, D = x_input.shape
    
    xz = rearrange(
        dd_block.in_proj.weight @ rearrange(x_input, "b l d -> d (b l)"),
        "d (b l) -> b d l", l=L,
    )
    if dd_block.in_proj.bias is not None:
        xz = xz + rearrange(dd_block.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
    x_part, z = xz.chunk(2, dim=1)
    x_conv = dd_block.act(dd_block.conv1d(x_part)[..., :L])
    
    sel = dd_block._project_selectivity(x_conv, batch=B, seqlen=L)
    dt = sel["dt_combined"]
    B_val = sel["B_combined"]
    
    A = -torch.exp(dd_block.A_log.float())
    delta_bias = dd_block.dt_proj_inv.bias.float() if dd_block.dt_proj_inv.bias is not None else None
    
    dt_f = dt.float()
    if delta_bias is not None:
        dt_f = dt_f + delta_bias.unsqueeze(0).unsqueeze(-1)
    dt_soft = F.softplus(dt_f)
    
    d_inner, d_state = A.shape
    h = torch.zeros(B, d_inner, d_state, device=device)
    
    # Collect stats at 4 checkpoint steps
    checkpoint_steps = [4, 9, 14, 19]  # 0-indexed: steps 5, 10, 15, 20
    h_checkpoints = []
    
    for k in range(L):
        dA = torch.exp(dt_soft[:, :, k].unsqueeze(-1) * A.unsqueeze(0))
        dB_u = dt_soft[:, :, k].unsqueeze(-1) * B_val[:, :, k].unsqueeze(1) * x_conv[:, :, k].unsqueeze(-1)
        h = dA * h + dB_u
        
        if k in checkpoint_steps:
            # Flatten to (B, d_inner*d_state) but use mean over d_state for efficiency
            h_flat = h.mean(dim=-1)  # (B, d_inner) - mean over state dim
            h_checkpoints.append(h_flat)
    
    # Return mean and var over checkpoint steps
    h_stack = torch.stack(h_checkpoints, dim=1)  # (B, 4, d_inner)
    h_mean = h_stack.mean(dim=1)  # (B, d_inner)
    h_var = h_stack.var(dim=1)    # (B, d_inner)
    return torch.cat([h_mean, h_var], dim=-1)  # (B, 2*d_inner)


def main():
    set_seed(42)
    device = infer_device("auto")
    
    # Build model args matching v3 SPD config
    model_args = argparse.Namespace(
        d_model=None, d_state=16, d_conv=8, expand=2, num_mamba_layers=1,
        num_transformer_layers=3, num_heads=7, dropout=0.5, dim_feedforward=84,
        transformer_impl="custom", transformer_norm_mode="pre",
        transformer_inner_dropout=0.0, mamba_block_mode="dd_spd",
        spd_gate_init_bias=-2.0, spd_predictor_mode="shared_head",
        # data args
        root="/home/shelterpl/data/CMAPSS", task="FD001_TO_FD003",
        source_subset=None, target_subset=None,
        window_size=20, stride=1, rul_clip=125, target_scale=1.0,
        grad_clip_norm=0.0, source_train_ratio=0.8, source_split_seed=42,
        source_split_path=None, resample_source_split_per_seed=False,
        source_normalizer_fit_scope="train_only",
        target_shots=5, target_val_units=10, few_shot_seed=42,
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
        num_workers=0,
        max_source_train_batches=None, max_target_train_batches=None,
    )
    
    task = resolve_cross_domain_task("FD001_TO_FD003")
    output_dir = Path("runs/ssda_quick_test/FD001_TO_FD003")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / "seed_42"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    source_train_full = load_cmapss_split(model_args.root, "FD001", "train", rul_clip=125)
    source_test_raw = load_cmapss_split(model_args.root, "FD001", "test", rul_clip=125)
    target_train_full = load_cmapss_split(model_args.root, "FD003", "train", rul_clip=125)
    target_test_raw = load_cmapss_split(model_args.root, "FD003", "test", rul_clip=125)
    
    source_split = load_or_create_source_split(model_args, source_train_full, output_dir, 42)
    source_loaders, source_meta = build_source_stage_data(model_args, source_train_full, source_test_raw, source_split)
    
    from train_supervised import build_model
    model = build_model(model_args, 21).to(device)
    
    # Source pre-training
    print("=== Source pre-training ===")
    source_stage = fit_source_stage(model_args, 42, model, source_loaders, device, run_dir, 1.0, None)
    src_test = evaluate(model, source_loaders["test"], device, 1.0)
    print(f"Source test RMSE: {src_test['rmse']:.4f}")
    
    # Target setup
    target_partition = load_or_create_target_partition(model_args, target_train_full, output_dir, 42)
    target_loaders, target_meta = build_target_cd_data(model_args, target_train_full, target_test_raw, target_partition)
    target_direct = evaluate(model, target_loaders["test"], device, 1.0)
    print(f"Direct target RMSE: {target_direct['rmse']:.4f}")
    
    # Fresh model for adaptation
    cd_model = build_model(model_args, 21).to(device)
    src_ckpt = torch.load(source_stage["checkpoint"], map_location=device)
    cd_model.load_state_dict(src_ckpt["model_state_dict"])
    
    # Stage head
    stage_head = nn.Linear(cd_model.head.in_features, 3).to(device)
    
    # Optimizer includes all params
    optimizer = torch.optim.Adam(
        list(cd_model.parameters()) + list(stage_head.parameters()),
        lr=5e-4,
    )
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    
    def endless(loader):
        while True:
            for batch in loader:
                yield batch
    
    best_val = float("inf")
    best_test = None
    
    print("\n=== CD adaptation with SSDA ===")
    for epoch in range(1, 21):
        cd_model.train()
        stage_head.train()
        
        src_iter = endless(source_loaders["train"])
        tgt_lab_iter = endless(target_loaders["labeled"])
        tgt_unlab_iter = endless(target_loaders["unlabeled"])
        
        n_steps = max(len(source_loaders["train"]), len(target_loaders["labeled"]))
        
        epoch_loss = 0.0
        epoch_ssda = 0.0
        n = 0
        
        for _ in range(n_steps):
            src_x, src_y = next(src_iter)
            tgt_x, tgt_y = next(tgt_lab_iter)
            tgt_u, _ = next(tgt_unlab_iter)
            
            src_x = src_x.to(device)
            src_y = src_y.to(device)
            tgt_x = tgt_x.to(device)
            tgt_y = tgt_y.to(device)
            tgt_u = tgt_u.to(device)
            
            # Forward with aux for inv-MMD
            src_aux = cd_model.forward_features_with_aux(src_x)
            tgt_aux = cd_model.forward_features_with_aux(tgt_x)
            tgt_u_aux = cd_model.forward_features_with_aux(tgt_u)
            
            # Task losses
            src_pred = cd_model.predict_from_features(src_aux["features"])
            tgt_pred = cd_model.predict_from_features(tgt_aux["features"])
            l_src = mse(src_pred, src_y)
            l_tgt = mse(tgt_pred, tgt_y)
            
            # Stage loss
            src_stages = assign_rul_stage_labels(src_y, rul_clip=125.0, num_stages=3)
            stage_logits = stage_head(src_aux["features"])
            l_stage = ce(stage_logits, src_stages)
            
            # Global MMD on combined features
            tgt_all_feat = torch.cat([tgt_aux["features"], tgt_u_aux["features"]], dim=0)
            l_mmd = gaussian_mmd_loss(src_aux["features"], tgt_all_feat)
            
            # inv-MMD
            tgt_all_inv = torch.cat([tgt_aux["domain_features"], tgt_u_aux["domain_features"]], dim=0)
            l_inv_mmd = gaussian_mmd_loss(src_aux["domain_features"], tgt_all_inv)
            
            # === SSDA: hidden state statistics alignment ===
            dd_block = cd_model.mamba_blocks[0]
            src_input = cd_model.input_proj(src_x)
            tgt_input = cd_model.input_proj(torch.cat([tgt_x, tgt_u], dim=0))
            
            src_h_stats = collect_hidden_state_stats(dd_block, src_input, device)
            tgt_h_stats = collect_hidden_state_stats(dd_block, tgt_input, device)
            l_ssda = gaussian_mmd_loss(src_h_stats, tgt_h_stats, sigmas=(0.1, 0.5, 1.0, 2.0, 5.0))
            
            # Total loss
            loss = (
                l_src + l_tgt
                + 0.1 * l_mmd
                + 1.0 * l_stage
                + 0.1 * l_inv_mmd
                + 1.0 * l_ssda  # SSDA weight - conservative start
            )
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            bs = src_x.shape[0]
            epoch_loss += loss.item() * bs
            epoch_ssda += l_ssda.item() * bs
            n += bs
        
        # Validate
        val = evaluate(cd_model, target_loaders["val"], device, 1.0)
        test = evaluate(cd_model, target_loaders["test"], device, 1.0)
        
        print(f"  epoch {epoch:2d}: loss={epoch_loss/n:.4f} ssda={epoch_ssda/n:.6f} val_rmse={val['rmse']:.2f} test_rmse={test['rmse']:.2f}")
        
        if val["rmse"] < best_val:
            best_val = val["rmse"]
            best_test = test
            best_epoch = epoch
    
    print(f"\n=== RESULT ===")
    print(f"Best epoch: {best_epoch}")
    print(f"Best test RMSE: {best_test['rmse']:.4f}")
    print(f"Direct RMSE: {target_direct['rmse']:.4f}")
    print(f"Gain: {target_direct['rmse'] - best_test['rmse']:.4f}")
    print(f"\nComparison:")
    print(f"  SPD + inv-MMD only (seed42): 22.40")
    print(f"  SPD + inv-MMD + SSDA:        {best_test['rmse']:.2f}")
    print(f"  v2 reference (seed42):       23.34")


if __name__ == "__main__":
    main()
