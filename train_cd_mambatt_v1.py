from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cd_mambatt.data import CMAPSSSplit, build_windows, fit_normalizer, load_cmapss_split, select_units
from cd_mambatt.losses import gaussian_mmd_loss
from train_cross_domain_baseline import (
    build_source_stage_data,
    build_target_direct_test_loader,
    build_window_loader_from_split,
    load_or_create_source_split,
    load_or_create_target_partition,
)
from train_supervised import build_model, evaluate, infer_device, parse_seeds, set_seed


def build_target_cd_data(
    args: argparse.Namespace,
    target_train_full: CMAPSSSplit,
    target_test_raw: CMAPSSSplit,
    partition: dict[str, object],
) -> tuple[dict[str, DataLoader], dict[str, object]]:
    labeled_raw = select_units(target_train_full, partition["labeled_units"])
    val_raw = select_units(target_train_full, partition["validation_units"])
    unlabeled_raw = select_units(target_train_full, partition["unlabeled_units"])

    normalizer = fit_normalizer(target_train_full)
    labeled_split = normalizer.transform(labeled_raw)
    val_split = normalizer.transform(val_raw)
    unlabeled_split = normalizer.transform(unlabeled_raw)
    test_split = normalizer.transform(target_test_raw)

    labeled_loader, labeled_shape = build_window_loader_from_split(
        labeled_split,
        window_size=args.window_size,
        stride=args.stride,
        last_only=False,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    unlabeled_loader, unlabeled_shape = build_window_loader_from_split(
        unlabeled_split,
        window_size=args.window_size,
        stride=args.stride,
        last_only=False,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader, val_shape = build_window_loader_from_split(
        val_split,
        window_size=args.window_size,
        stride=args.stride,
        last_only=not args.target_val_all_windows,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader, test_shape = build_window_loader_from_split(
        test_split,
        window_size=args.window_size,
        stride=args.stride,
        last_only=True,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return (
        {
            "labeled": labeled_loader,
            "unlabeled": unlabeled_loader,
            "val": val_loader,
            "test": test_loader,
        },
        {
            "labeled_windows_shape": labeled_shape,
            "unlabeled_windows_shape": unlabeled_shape,
            "val_windows_shape": val_shape,
            "test_windows_shape": test_shape,
            "labeled_unit_count": int(labeled_raw.num_units),
            "val_unit_count": int(val_raw.num_units),
            "unlabeled_unit_count": int(unlabeled_raw.num_units),
        },
    )


def endless_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def run_cd_epoch(
    model: nn.Module,
    source_loader: DataLoader,
    target_labeled_loader: DataLoader,
    target_unlabeled_loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    source_loss_weight: float,
    target_loss_weight: float,
    lambda_mmd: float,
    target_scale: float,
    grad_clip_norm: float | None,
    max_batches: int | None,
    mmd_sigmas: tuple[float, ...],
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    mse_loss = nn.MSELoss()

    source_iter = endless_loader(source_loader)
    target_labeled_iter = endless_loader(target_labeled_loader)
    target_unlabeled_iter = endless_loader(target_unlabeled_loader)
    num_steps = max(len(source_loader), len(target_labeled_loader), len(target_unlabeled_loader))
    if max_batches is not None:
        num_steps = min(num_steps, max_batches)

    total_examples = 0
    total_source_loss = 0.0
    total_target_loss = 0.0
    total_mmd_loss = 0.0
    total_combined_loss = 0.0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for _ in range(num_steps):
            source_windows, source_targets = next(source_iter)
            target_labeled_windows, target_labeled_targets = next(target_labeled_iter)
            target_unlabeled_windows, _ = next(target_unlabeled_iter)

            source_windows = source_windows.to(device, non_blocking=True)
            source_targets = source_targets.to(device, non_blocking=True) / target_scale
            target_labeled_windows = target_labeled_windows.to(device, non_blocking=True)
            target_labeled_targets = target_labeled_targets.to(device, non_blocking=True) / target_scale
            target_unlabeled_windows = target_unlabeled_windows.to(device, non_blocking=True)

            source_features = model.forward_features(source_windows)
            source_predictions = model.predict_from_features(source_features)
            source_loss = mse_loss(source_predictions, source_targets)

            target_labeled_features = model.forward_features(target_labeled_windows)
            target_predictions = model.predict_from_features(target_labeled_features)
            target_loss = mse_loss(target_predictions, target_labeled_targets)

            target_unlabeled_features = model.forward_features(target_unlabeled_windows)
            target_alignment_features = torch.cat([target_labeled_features, target_unlabeled_features], dim=0)
            mmd_loss = gaussian_mmd_loss(source_features, target_alignment_features, sigmas=mmd_sigmas)

            total_loss = (
                source_loss_weight * source_loss
                + target_loss_weight * target_loss
                + lambda_mmd * mmd_loss
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

            batch_examples = int(source_windows.shape[0])
            total_examples += batch_examples
            total_source_loss += float(source_loss.detach().cpu()) * batch_examples
            total_target_loss += float(target_loss.detach().cpu()) * batch_examples
            total_mmd_loss += float(mmd_loss.detach().cpu()) * batch_examples
            total_combined_loss += float(total_loss.detach().cpu()) * batch_examples

    if total_examples == 0:
        raise ValueError("No cross-domain batches were processed")

    return {
        "source_loss": total_source_loss / total_examples,
        "target_loss": total_target_loss / total_examples,
        "mmd_loss": total_mmd_loss / total_examples,
        "total_loss": total_combined_loss / total_examples,
    }


def fit_source_stage(
    args: argparse.Namespace,
    seed: int,
    model: nn.Module,
    source_loaders: dict[str, DataLoader],
    device: torch.device,
    run_dir: Path,
    target_scale: float,
    grad_clip_norm: float | None,
) -> dict[str, object]:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    best_val_rmse = float("inf")
    best_epoch = -1
    history: list[dict[str, object]] = []
    ckpt_path = run_dir / "source" / "best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    from train_supervised import run_epoch

    for epoch in range(1, args.source_epochs + 1):
        train_mse = run_epoch(
            model,
            source_loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            max_batches=args.max_source_train_batches,
            target_scale=target_scale,
            grad_clip_norm=grad_clip_norm,
        )
        val_metrics = evaluate(model, source_loaders["val"], device, target_scale)
        record = {
            "stage": "source",
            "seed": seed,
            "epoch": epoch,
            "train_mse": train_mse,
            "train_rmse": float(np.sqrt(train_mse) * target_scale),
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_score": val_metrics["score"],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = float(val_metrics["rmse"])
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_val_rmse": best_val_rmse,
                },
                ckpt_path,
            )

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return {
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "checkpoint": str(ckpt_path),
        "history": history,
    }


def fit_cd_stage(
    args: argparse.Namespace,
    seed: int,
    model: nn.Module,
    source_train_loader: DataLoader,
    target_loaders: dict[str, DataLoader],
    device: torch.device,
    run_dir: Path,
    target_scale: float,
    grad_clip_norm: float | None,
) -> dict[str, object]:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.target_lr, weight_decay=args.target_weight_decay)
    best_val_rmse = float("inf")
    best_epoch = -1
    history: list[dict[str, object]] = []
    ckpt_path = run_dir / "cd_stage" / "best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.target_epochs + 1):
        train_stats = run_cd_epoch(
            model,
            source_train_loader,
            target_loaders["labeled"],
            target_loaders["unlabeled"],
            optimizer,
            device,
            source_loss_weight=args.source_loss_weight,
            target_loss_weight=args.target_loss_weight,
            lambda_mmd=args.lambda_mmd,
            target_scale=target_scale,
            grad_clip_norm=grad_clip_norm,
            max_batches=args.max_target_train_batches,
            mmd_sigmas=tuple(float(token) for token in args.mmd_sigmas.split(",")),
        )
        val_metrics = evaluate(model, target_loaders["val"], device, target_scale)
        record = {
            "stage": "cd_mmd",
            "seed": seed,
            "epoch": epoch,
            "train_total_loss": train_stats["total_loss"],
            "train_source_loss": train_stats["source_loss"],
            "train_target_loss": train_stats["target_loss"],
            "train_mmd_loss": train_stats["mmd_loss"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_score": val_metrics["score"],
            "lambda_mmd": args.lambda_mmd,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = float(val_metrics["rmse"])
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_val_rmse": best_val_rmse,
                },
                ckpt_path,
            )

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return {
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "checkpoint": str(ckpt_path),
        "history": history,
    }


def train_one_seed(
    args: argparse.Namespace,
    task_name: str,
    source_subset: str,
    target_subset: str,
    seed: int,
    device: torch.device,
    task_output_dir: Path,
    source_train_full: CMAPSSSplit,
    source_test_raw: CMAPSSSplit,
    target_train_full: CMAPSSSplit,
    target_test_raw: CMAPSSSplit,
) -> dict[str, object]:
    set_seed(seed)
    run_dir = task_output_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    target_scale = float(args.target_scale)
    grad_clip_norm = None if args.grad_clip_norm <= 0 else float(args.grad_clip_norm)

    source_split = load_or_create_source_split(args, source_train_full, task_output_dir, seed)
    source_loaders, source_meta = build_source_stage_data(args, source_train_full, source_test_raw, source_split)

    base_model = build_model(args, int(source_meta["train_windows_shape"][2])).to(device)
    source_stage = fit_source_stage(args, seed, base_model, source_loaders, device, run_dir, target_scale, grad_clip_norm)
    source_test_metrics = evaluate(base_model, source_loaders["test"], device, target_scale)
    target_direct_loader, target_direct_meta = build_target_direct_test_loader(args, target_train_full, target_test_raw)
    target_direct_metrics = evaluate(base_model, target_direct_loader, device, target_scale)

    target_partition = load_or_create_target_partition(args, target_train_full, task_output_dir, seed)
    target_loaders, target_meta = build_target_cd_data(args, target_train_full, target_test_raw, target_partition)

    cd_model = build_model(args, int(source_meta["train_windows_shape"][2])).to(device)
    source_checkpoint = torch.load(source_stage["checkpoint"], map_location=device)
    cd_model.load_state_dict(source_checkpoint["model_state_dict"])
    cd_stage = fit_cd_stage(
        args,
        seed,
        cd_model,
        source_loaders["train"],
        target_loaders,
        device,
        run_dir,
        target_scale,
        grad_clip_norm,
    )
    cd_target_test_metrics = evaluate(cd_model, target_loaders["test"], device, target_scale)

    result = {
        "seed": seed,
        "task": task_name,
        "source_subset": source_subset,
        "target_subset": target_subset,
        "source_split_path": source_split["path"],
        "source_split_seed": int(source_split["split_seed"]),
        "target_partition_path": target_partition["path"],
        "few_shot_seed": int(target_partition["few_shot_seed"]),
        "source_stage": {
            "best_epoch": int(source_stage["best_epoch"]),
            "best_val_rmse": float(source_stage["best_val_rmse"]),
            "checkpoint": str(source_stage["checkpoint"]),
            "train_windows_shape": source_meta["train_windows_shape"],
            "val_windows_shape": source_meta["val_windows_shape"],
            "test_windows_shape": source_meta["test_windows_shape"],
            "train_unit_count": int(source_meta["train_unit_count"]),
            "val_unit_count": int(source_meta["val_unit_count"]),
        },
        "source_test": {
            "rmse": float(source_test_metrics["rmse"]),
            "mae": float(source_test_metrics["mae"]),
            "score": float(source_test_metrics["score"]),
        },
        "target_direct": {
            "rmse": float(target_direct_metrics["rmse"]),
            "mae": float(target_direct_metrics["mae"]),
            "score": float(target_direct_metrics["score"]),
            "test_windows_shape": target_direct_meta["test_windows_shape"],
            "test_unit_count": int(target_direct_meta["test_unit_count"]),
        },
        "cd_stage": {
            "best_epoch": int(cd_stage["best_epoch"]),
            "best_val_rmse": float(cd_stage["best_val_rmse"]),
            "checkpoint": str(cd_stage["checkpoint"]),
            "lambda_mmd": float(args.lambda_mmd),
            "source_loss_weight": float(args.source_loss_weight),
            "target_loss_weight": float(args.target_loss_weight),
            "labeled_windows_shape": target_meta["labeled_windows_shape"],
            "unlabeled_windows_shape": target_meta["unlabeled_windows_shape"],
            "val_windows_shape": target_meta["val_windows_shape"],
            "test_windows_shape": target_meta["test_windows_shape"],
            "labeled_unit_count": int(target_meta["labeled_unit_count"]),
            "val_unit_count": int(target_meta["val_unit_count"]),
            "unlabeled_unit_count": int(target_meta["unlabeled_unit_count"]),
            "test_rmse": float(cd_target_test_metrics["rmse"]),
            "test_mae": float(cd_target_test_metrics["mae"]),
            "test_score": float(cd_target_test_metrics["score"]),
        },
    }
    with (run_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def summarize_results(run_results: list[dict[str, object]], args: argparse.Namespace, start_time: float) -> dict[str, object]:
    direct_rmses = [float(item["target_direct"]["rmse"]) for item in run_results]
    cd_rmses = [float(item["cd_stage"]["test_rmse"]) for item in run_results]
    cd_scores = [float(item["cd_stage"]["test_score"]) for item in run_results]
    best_direct_run = min(run_results, key=lambda item: float(item["target_direct"]["rmse"]))
    best_cd_run = min(run_results, key=lambda item: float(item["cd_stage"]["best_val_rmse"]))

    return {
        "task": str(run_results[0]["task"]),
        "source_subset": str(run_results[0]["source_subset"]),
        "target_subset": str(run_results[0]["target_subset"]),
        "seeds": [int(item["seed"]) for item in run_results],
        "num_runs": len(run_results),
        "target_shots": int(args.target_shots),
        "target_val_units": int(args.target_val_units),
        "lambda_mmd": float(args.lambda_mmd),
        "source_loss_weight": float(args.source_loss_weight),
        "target_loss_weight": float(args.target_loss_weight),
        "best_direct_seed": int(best_direct_run["seed"]),
        "best_direct_target_rmse": float(best_direct_run["target_direct"]["rmse"]),
        "mean_direct_target_rmse": float(np.mean(direct_rmses)),
        "std_direct_target_rmse": float(np.std(direct_rmses)),
        "best_cd_seed": int(best_cd_run["seed"]),
        "best_cd_val_rmse": float(best_cd_run["cd_stage"]["best_val_rmse"]),
        "best_cd_test_rmse": float(best_cd_run["cd_stage"]["test_rmse"]),
        "mean_cd_test_rmse": float(np.mean(cd_rmses)),
        "std_cd_test_rmse": float(np.std(cd_rmses)),
        "mean_cd_test_score": float(np.mean(cd_scores)),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CD-MambAtt v1: source supervision + target few-shot supervision + MMD alignment.")
    parser.add_argument("--root", default="/home/shelterpl/data/CMAPSS", help="CMAPSS root directory")
    parser.add_argument("--task", default="FD001_TO_FD003", help="Registered cross-domain task name")
    parser.add_argument("--source-subset", default=None, help="Optional source subset override")
    parser.add_argument("--target-subset", default=None, help="Optional target subset override")
    parser.add_argument("--window-size", type=int, default=20, help="Sliding window size")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride")
    parser.add_argument("--rul-clip", type=int, default=125, help="Piece-wise linear RUL cap")
    parser.add_argument("--target-scale", type=float, default=1.0, help="Optional label scale divisor used during training")
    parser.add_argument("--grad-clip-norm", type=float, default=0.0, help="Clip gradient norm during training when > 0")
    parser.add_argument("--source-train-ratio", type=float, default=0.8, help="Engine-level source train/validation split ratio")
    parser.add_argument("--source-split-seed", type=int, default=42, help="Seed used to create the source engine split")
    parser.add_argument("--source-split-path", default=None, help="Optional JSON path for a fixed source engine split")
    parser.add_argument("--resample-source-split-per-seed", action="store_true", help="Regenerate source train/validation units for each run seed")
    parser.add_argument("--source-normalizer-fit-scope", choices=("train_only", "trainval"), default="train_only", help="Fit source normalization on source train only or full source train split")
    parser.add_argument("--target-shots", type=int, default=5, help="Number of labeled target engines used for adaptation")
    parser.add_argument("--target-val-units", type=int, default=10, help="Number of target training engines reserved for validation")
    parser.add_argument("--few-shot-seed", type=int, default=42, help="Seed used to sample target few-shot partitions")
    parser.add_argument("--target-partition-path", default=None, help="Optional JSON path for a fixed target few-shot partition")
    parser.add_argument("--resample-few-shot-per-seed", action="store_true", help="Resample the target few-shot partition for each run seed")
    parser.add_argument("--seeds", default=None, help="Optional comma-separated seed list; overrides --num-runs")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of run seeds when --seeds is not provided")
    parser.add_argument("--base-seed", type=int, default=42, help="First seed used when --seeds is not provided")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--source-epochs", type=int, default=50, help="Source supervised pretraining epochs")
    parser.add_argument("--target-epochs", type=int, default=20, help="Cross-domain adaptation epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Source-stage learning rate")
    parser.add_argument("--target-lr", type=float, default=5e-4, help="Adaptation-stage learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Source-stage weight decay")
    parser.add_argument("--target-weight-decay", type=float, default=0.0, help="Adaptation-stage weight decay")
    parser.add_argument("--source-loss-weight", type=float, default=1.0, help="Weight for source supervised loss during adaptation")
    parser.add_argument("--target-loss-weight", type=float, default=1.0, help="Weight for target few-shot supervised loss during adaptation")
    parser.add_argument("--lambda-mmd", type=float, default=0.1, help="Weight for MMD alignment loss")
    parser.add_argument("--mmd-sigmas", default="1,2,4,8,16", help="Comma-separated Gaussian kernel bandwidths for MMD")
    parser.add_argument("--d-model", type=int, default=None, help="Optional exploratory model width")
    parser.add_argument("--d-state", type=int, default=16, help="Mamba d_state")
    parser.add_argument("--d-conv", type=int, default=8, help="Mamba d_conv")
    parser.add_argument("--expand", type=int, default=2, help="Mamba expand factor")
    parser.add_argument("--num-mamba-layers", type=int, default=1, help="Number of Mamba blocks")
    parser.add_argument("--num-transformer-layers", type=int, default=3, help="Number of Transformer layers")
    parser.add_argument("--num-heads", type=int, default=7, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.5, help="Final dropout rate before the regression head")
    parser.add_argument("--dim-feedforward", type=int, default=84, help="Transformer FFN hidden size")
    parser.add_argument("--transformer-impl", choices=("custom", "torch"), default="custom", help="Transformer implementation")
    parser.add_argument("--transformer-norm-mode", choices=("pre", "post"), default="pre", help="Transformer normalization order")
    parser.add_argument("--transformer-inner-dropout", type=float, default=0.0, help="Dropout applied inside Transformer residual branches")
    parser.add_argument("--mamba-block-mode", choices=("bare", "prenorm_residual"), default="bare", help="Whether to wrap each Mamba layer with pre-norm residual")
    parser.add_argument("--source-val-all-windows", action="store_true", help="Validate source stage on all source validation windows")
    parser.add_argument("--target-val-all-windows", action="store_true", help="Validate target adaptation stage on all target validation windows")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--max-source-train-batches", type=int, default=None, help="Optional cap for source-stage smoke tests")
    parser.add_argument("--max-target-train-batches", type=int, default=None, help="Optional cap for adaptation-stage smoke tests")
    parser.add_argument("--output-dir", default="/home/shelterpl/cd_mambatt/runs/cd_mambatt_v1", help="Run output directory")
    args = parser.parse_args()

    if args.target_shots <= 0:
        raise ValueError("CD-MambAtt v1 expects target_shots > 0")

    from cd_mambatt.cross_domain import resolve_cross_domain_task

    task = resolve_cross_domain_task(args.task, source_subset=args.source_subset, target_subset=args.target_subset)
    device = infer_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("CD-MambAtt v1 requires CUDA in the configured environment.")

    task_output_dir = Path(args.output_dir).expanduser().resolve() / task.name
    task_output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds, args.num_runs, args.base_seed)

    source_train_full = load_cmapss_split(args.root, task.source_subset, "train", rul_clip=args.rul_clip)
    source_test_raw = load_cmapss_split(args.root, task.source_subset, "test", rul_clip=args.rul_clip)
    target_train_full = load_cmapss_split(args.root, task.target_subset, "train", rul_clip=args.rul_clip)
    target_test_raw = load_cmapss_split(args.root, task.target_subset, "test", rul_clip=args.rul_clip)

    run_results: list[dict[str, object]] = []
    start_time = time.time()
    for seed in seeds:
        result = train_one_seed(
            args,
            task.name,
            task.source_subset,
            task.target_subset,
            seed,
            device,
            task_output_dir,
            source_train_full,
            source_test_raw,
            target_train_full,
            target_test_raw,
        )
        print(json.dumps(result, ensure_ascii=False))
        run_results.append(result)

    summary = summarize_results(run_results, args, start_time)
    with (task_output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"runs": run_results, "summary": summary}, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
