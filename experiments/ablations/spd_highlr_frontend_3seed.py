"""SPD + spec-domain + high LR + cosine, with invariant MMD on frontend features."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from _pathfix import ensure_repo_root

ensure_repo_root()

from cd_mambatt.data import load_cmapss_split
from cd_mambatt.losses.mmd import gaussian_mmd_loss
from cd_mambatt.pseudo_labeling import assign_rul_stage_labels
from train_cd_mambatt_v1 import build_target_cd_data, fit_source_stage
from train_cross_domain_baseline import (
    build_source_stage_data,
    load_or_create_source_split,
    load_or_create_target_partition,
)
from train_supervised import build_model, evaluate, infer_device, set_seed
from cd_mambatt.cross_domain import resolve_cross_domain_task


def endless(loader):
    while True:
        for batch in loader:
            yield batch


def make_args():
    return argparse.Namespace(
        d_model=None,
        d_state=16,
        d_conv=8,
        expand=2,
        num_mamba_layers=1,
        num_transformer_layers=3,
        num_heads=7,
        dropout=0.5,
        dim_feedforward=84,
        transformer_impl="custom",
        transformer_norm_mode="pre",
        transformer_inner_dropout=0.0,
        mamba_block_mode="dd_spd",
        spd_gate_init_bias=-2.0,
        spd_predictor_mode="shared_head",
        root="/home/shelterpl/data/CMAPSS",
        window_size=20,
        stride=1,
        rul_clip=125,
        target_scale=1.0,
        grad_clip_norm=0.0,
        source_train_ratio=0.8,
        source_split_seed=42,
        source_split_path=None,
        resample_source_split_per_seed=False,
        source_normalizer_fit_scope="train_only",
        target_shots=5,
        target_val_units=10,
        target_partition_path=None,
        resample_few_shot_per_seed=True,
        batch_size=64,
        source_epochs=50,
        target_epochs=20,
        lr=1e-3,
        target_lr=5e-4,
        weight_decay=0.0,
        target_weight_decay=0.0,
        source_loss_weight=1.0,
        target_loss_weight=1.0,
        lambda_mmd=0.1,
        mmd_sigmas="1,2,4,8,16",
        lambda_source_stage=1.0,
        lambda_pseudo=0.5,
        lambda_contrastive=0.0,
        lambda_monotonic=0.05,
        contrastive_temperature=0.1,
        disable_contrastive_feature_normalization=False,
        monotonic_margin=0.0,
        monotonic_pair_gap=1,
        monotonic_pair_stride=1,
        num_pseudo_stages=3,
        pseudo_start_quantile=0.5,
        pseudo_end_quantile=0.9,
        disable_pseudo_feature_normalization=False,
        source_val_all_windows=True,
        target_val_all_windows=True,
        num_workers=0,
        max_source_train_batches=None,
        max_target_train_batches=None,
    )


def run_one_seed(
    seed: int,
    device: torch.device,
    *,
    source_subset: str,
    target_subset: str,
    task_name: str,
    output_root: Path,
    target_lr: float,
    eta_min: float,
    inv_tap: str,
) -> dict[str, float]:
    set_seed(seed)
    args = make_args()
    args.few_shot_seed = seed

    out_dir = output_root / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    src_full = load_cmapss_split(args.root, source_subset, "train", rul_clip=125)
    src_test = load_cmapss_split(args.root, source_subset, "test", rul_clip=125)
    tgt_full = load_cmapss_split(args.root, target_subset, "train", rul_clip=125)
    tgt_test = load_cmapss_split(args.root, target_subset, "test", rul_clip=125)

    src_split = load_or_create_source_split(args, src_full, out_dir, seed)
    src_loaders, _ = build_source_stage_data(args, src_full, src_test, src_split)
    tgt_partition = load_or_create_target_partition(args, tgt_full, out_dir, seed)
    tgt_loaders, _ = build_target_cd_data(args, tgt_full, tgt_test, tgt_partition)

    model = build_model(args, 21).to(device)
    src_stage = fit_source_stage(args, seed, model, src_loaders, device, run_dir, 1.0, None)
    direct = evaluate(model, tgt_loaders["test"], device, 1.0)["rmse"]

    cd_model = build_model(args, 21).to(device)
    cd_model.load_state_dict(torch.load(src_stage["checkpoint"], map_location=device)["model_state_dict"])

    stage_head = nn.Linear(21, 3).to(device)
    spec_clf = nn.Linear(21, 2).to(device)
    optimizer = torch.optim.Adam(
        list(cd_model.parameters()) + list(stage_head.parameters()) + list(spec_clf.parameters()),
        lr=target_lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=eta_min)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_test = None
    best_epoch = -1
    for epoch in range(1, 21):
        cd_model.train()
        stage_head.train()
        spec_clf.train()
        src_iter = endless(src_loaders["train"])
        tgt_lab_iter = endless(tgt_loaders["labeled"])
        tgt_unlab_iter = endless(tgt_loaders["unlabeled"])
        n_steps = max(len(src_loaders["train"]), len(tgt_loaders["labeled"]))

        for _ in range(n_steps):
            sx, sy = next(src_iter)
            tx, ty = next(tgt_lab_iter)
            tu, _ = next(tgt_unlab_iter)
            sx, sy = sx.to(device), sy.to(device)
            tx, ty = tx.to(device), ty.to(device)
            tu = tu.to(device)

            sa = cd_model.forward_features_with_aux(sx)
            ta = cd_model.forward_features_with_aux(tx)
            ua = cd_model.forward_features_with_aux(tu)

            l_src = mse(cd_model.predict_from_features(sa["features"]), sy)
            l_tgt = mse(cd_model.predict_from_features(ta["features"]), ty)
            l_stage = ce(stage_head(sa["features"]), assign_rul_stage_labels(sy, rul_clip=125.0, num_stages=3))
            l_mmd = gaussian_mmd_loss(sa["features"], torch.cat([ta["features"], ua["features"]], dim=0))

            if inv_tap == "frontend":
                src_inv = sa.get("frontend_features", sa["domain_features"])
                tgt_inv = ta.get("frontend_features", ta["domain_features"])
                tgt_u_inv = ua.get("frontend_features", ua["domain_features"])
            elif inv_tap == "inv":
                src_inv = sa["domain_features"]
                tgt_inv = ta["domain_features"]
                tgt_u_inv = ua["domain_features"]
            else:
                raise ValueError(f"Unsupported inv_tap: {inv_tap}")
            l_inv = gaussian_mmd_loss(src_inv, torch.cat([tgt_inv, tgt_u_inv], dim=0))

            spec_s = sa.get("specific_features")
            spec_t = ta.get("specific_features")
            if spec_s is not None and spec_t is not None:
                spec_all = torch.cat([spec_s, spec_t], dim=0)
                dlabels = torch.cat(
                    [
                        torch.zeros(spec_s.shape[0], dtype=torch.long, device=device),
                        torch.ones(spec_t.shape[0], dtype=torch.long, device=device),
                    ],
                    dim=0,
                )
                l_spec = 0.1 * ce(spec_clf(spec_all), dlabels)
            else:
                l_spec = torch.tensor(0.0, device=device)

            loss = l_src + l_tgt + 0.1 * l_mmd + 1.0 * l_stage + 0.1 * l_inv + l_spec
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        scheduler.step()
        val = evaluate(cd_model, tgt_loaders["val"], device, 1.0)
        test = evaluate(cd_model, tgt_loaders["test"], device, 1.0)
        if val["rmse"] < best_val:
            best_val = val["rmse"]
            best_test = test["rmse"]
            best_epoch = epoch

    print(f"seed {seed}: direct={direct:.4f} adapted={best_test:.4f} epoch={best_epoch}")
    return {"seed": seed, "direct": direct, "adapted": best_test, "epoch": best_epoch}


def main():
    parser = argparse.ArgumentParser(description="SPD high-LR frontend-alignment sweep")
    parser.add_argument("--target-lr", type=float, default=2e-3)
    parser.add_argument("--eta-min", type=float, default=1e-5)
    parser.add_argument("--task", default="FD001_TO_FD003")
    parser.add_argument("--source-subset", default=None)
    parser.add_argument("--target-subset", default=None)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--output-root", default="runs/spd_highlr_frontend_3seed")
    parser.add_argument("--inv-tap", choices=("frontend", "inv"), default="frontend")
    args = parser.parse_args()

    task = resolve_cross_domain_task(args.task, source_subset=args.source_subset, target_subset=args.target_subset)
    seeds = [int(token.strip()) for token in str(args.seeds).split(",") if token.strip()]
    output_root = Path(args.output_root)
    device = infer_device("auto")
    results = [
        run_one_seed(
            seed,
            device,
            source_subset=task.source_subset,
            target_subset=task.target_subset,
            task_name=task.name,
            output_root=output_root,
            target_lr=float(args.target_lr),
            eta_min=float(args.eta_min),
            inv_tap=str(args.inv_tap),
        )
        for seed in seeds
    ]
    adapted = [item["adapted"] for item in results]
    print("task", task.name, "source", task.source_subset, "target", task.target_subset)
    print("seeds", seeds)
    print("target_lr", float(args.target_lr), "eta_min", float(args.eta_min), "inv_tap", str(args.inv_tap))
    print("mean", float(np.mean(adapted)), "std", float(np.std(adapted)))


if __name__ == "__main__":
    main()
