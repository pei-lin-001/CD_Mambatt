from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cd_mambatt.cross_domain import build_few_shot_target_partition, resolve_cross_domain_task
from cd_mambatt.data import (
    CMAPSSSplit,
    CMAPSSWindowDataset,
    build_windows,
    fit_normalizer,
    get_train_validation_unit_ids,
    load_cmapss_split,
    select_units,
)
from train_supervised import (
    build_loader,
    build_model,
    evaluate,
    infer_device,
    parse_seeds,
    run_epoch,
    set_seed,
)


def resolve_seeded_path(base_path: Path, seed: int, resample_per_seed: bool) -> Path:
    if not resample_per_seed:
        return base_path
    return base_path.with_name(f"{base_path.stem}_seed{seed}{base_path.suffix}")


def ensure_complete_partition(
    full_split: CMAPSSSplit,
    train_units: np.ndarray,
    val_units: np.ndarray,
) -> None:
    expected_units = np.unique(full_split.unit_ids)
    actual_units = np.sort(np.concatenate([train_units, val_units]))
    if expected_units.shape != actual_units.shape or not np.array_equal(expected_units, actual_units):
        raise ValueError("Unit split definition does not match the full training subset")


def ensure_complete_few_shot_partition(
    full_split: CMAPSSSplit,
    labeled_units: np.ndarray,
    validation_units: np.ndarray,
    unlabeled_units: np.ndarray,
) -> None:
    expected_units = np.unique(full_split.unit_ids)
    actual_units = np.sort(np.concatenate([labeled_units, validation_units, unlabeled_units]))
    if expected_units.shape != actual_units.shape or not np.array_equal(expected_units, actual_units):
        raise ValueError("Few-shot partition does not cover the full target training subset")
    if np.intersect1d(labeled_units, validation_units).size > 0:
        raise ValueError("Labeled and validation target units overlap")
    if np.intersect1d(labeled_units, unlabeled_units).size > 0:
        raise ValueError("Labeled and unlabeled target units overlap")
    if np.intersect1d(validation_units, unlabeled_units).size > 0:
        raise ValueError("Validation and unlabeled target units overlap")


def load_or_create_source_split(
    args: argparse.Namespace,
    source_train_full: CMAPSSSplit,
    output_dir: Path,
    seed: int,
) -> dict[str, object]:
    base_path = (
        Path(args.source_split_path).expanduser().resolve()
        if args.source_split_path
        else output_dir / "splits" / "source_engine_split.json"
    )
    split_seed = seed if args.resample_source_split_per_seed else args.source_split_seed
    split_path = resolve_seeded_path(base_path, seed, args.resample_source_split_per_seed)

    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        train_units = np.asarray(payload["train_units"], dtype=np.int32)
        val_units = np.asarray(payload["val_units"], dtype=np.int32)
    else:
        train_units, val_units = get_train_validation_unit_ids(
            source_train_full,
            train_ratio=args.source_train_ratio,
            seed=split_seed,
        )
        payload = {
            "subset": source_train_full.subset,
            "train_ratio": args.source_train_ratio,
            "split_seed": split_seed,
            "source_seed": seed,
            "train_units": train_units.tolist(),
            "val_units": val_units.tolist(),
        }
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with split_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    ensure_complete_partition(source_train_full, train_units, val_units)
    return {
        "path": str(split_path),
        "split_seed": int(split_seed),
        "train_units": train_units,
        "val_units": val_units,
    }


def load_or_create_target_partition(
    args: argparse.Namespace,
    target_train_full: CMAPSSSplit,
    output_dir: Path,
    seed: int,
) -> dict[str, object]:
    base_path = (
        Path(args.target_partition_path).expanduser().resolve()
        if args.target_partition_path
        else output_dir / "splits" / "target_few_shot.json"
    )
    partition_seed = seed if args.resample_few_shot_per_seed else args.few_shot_seed
    partition_path = resolve_seeded_path(base_path, seed, args.resample_few_shot_per_seed)

    if partition_path.exists():
        with partition_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        labeled_units = np.asarray(payload["labeled_units"], dtype=np.int32)
        validation_units = np.asarray(payload["validation_units"], dtype=np.int32)
        unlabeled_units = np.asarray(payload["unlabeled_units"], dtype=np.int32)
    else:
        partition = build_few_shot_target_partition(
            target_train_full,
            num_shots=args.target_shots,
            num_val_units=args.target_val_units,
            seed=partition_seed,
        )
        labeled_units = partition.labeled_units
        validation_units = partition.validation_units
        unlabeled_units = partition.unlabeled_units
        payload = {
            "subset": target_train_full.subset,
            "shots": args.target_shots,
            "val_units": args.target_val_units,
            "few_shot_seed": partition_seed,
            **partition.to_dict(),
        }
        partition_path.parent.mkdir(parents=True, exist_ok=True)
        with partition_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    ensure_complete_few_shot_partition(target_train_full, labeled_units, validation_units, unlabeled_units)
    return {
        "path": str(partition_path),
        "few_shot_seed": int(partition_seed),
        "labeled_units": labeled_units,
        "validation_units": validation_units,
        "unlabeled_units": unlabeled_units,
    }


def build_window_loader_from_split(
    split: CMAPSSSplit,
    *,
    window_size: int,
    stride: int,
    last_only: bool,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> tuple[DataLoader, tuple[int, int, int]]:
    windows = build_windows(split, window_size, stride=stride, last_only=last_only)
    loader = build_loader(CMAPSSWindowDataset(windows), batch_size, shuffle, num_workers)
    return loader, windows.shape


def build_source_stage_data(
    args: argparse.Namespace,
    source_train_full: CMAPSSSplit,
    source_test_raw: CMAPSSSplit,
    source_split: dict[str, object],
) -> tuple[dict[str, DataLoader], dict[str, object]]:
    train_raw = select_units(source_train_full, source_split["train_units"])
    val_raw = select_units(source_train_full, source_split["val_units"])

    normalizer_source = train_raw if args.source_normalizer_fit_scope == "train_only" else source_train_full
    normalizer = fit_normalizer(normalizer_source)
    train_split = normalizer.transform(train_raw)
    val_split = normalizer.transform(val_raw)
    test_split = normalizer.transform(source_test_raw)

    train_loader, train_shape = build_window_loader_from_split(
        train_split,
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
        last_only=not args.source_val_all_windows,
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
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        },
        {
            "train_windows_shape": train_shape,
            "val_windows_shape": val_shape,
            "test_windows_shape": test_shape,
            "train_unit_count": int(train_raw.num_units),
            "val_unit_count": int(val_raw.num_units),
        },
    )


def build_target_direct_test_loader(
    args: argparse.Namespace,
    target_train_full: CMAPSSSplit,
    target_test_raw: CMAPSSSplit,
) -> tuple[DataLoader, dict[str, object]]:
    normalizer = fit_normalizer(target_train_full)
    test_split = normalizer.transform(target_test_raw)
    test_loader, test_shape = build_window_loader_from_split(
        test_split,
        window_size=args.window_size,
        stride=args.stride,
        last_only=True,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return test_loader, {"test_windows_shape": test_shape, "test_unit_count": int(target_test_raw.num_units)}


def build_target_finetune_data(
    args: argparse.Namespace,
    target_train_full: CMAPSSSplit,
    target_test_raw: CMAPSSSplit,
    partition: dict[str, object],
) -> tuple[dict[str, DataLoader], dict[str, object]]:
    labeled_raw = select_units(target_train_full, partition["labeled_units"])
    val_raw = select_units(target_train_full, partition["validation_units"])
    normalizer = fit_normalizer(target_train_full)
    labeled_split = normalizer.transform(labeled_raw)
    val_split = normalizer.transform(val_raw)
    test_split = normalizer.transform(target_test_raw)

    train_loader, train_shape = build_window_loader_from_split(
        labeled_split,
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
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        },
        {
            "train_windows_shape": train_shape,
            "val_windows_shape": val_shape,
            "test_windows_shape": test_shape,
            "labeled_unit_count": int(labeled_raw.num_units),
            "val_unit_count": int(val_raw.num_units),
            "unlabeled_unit_count": int(np.asarray(partition["unlabeled_units"], dtype=np.int32).shape[0]),
        },
    )


def set_finetune_mode(model: nn.Module, mode: str) -> None:
    if mode not in {"none", "head", "full"}:
        raise ValueError("finetune mode must be one of: none, head, full")
    for parameter in model.parameters():
        parameter.requires_grad = mode == "full"
    if mode == "head":
        for parameter in model.head.parameters():
            parameter.requires_grad = True


def fit_stage(
    *,
    stage_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    target_scale: float,
    grad_clip_norm: float | None,
    max_train_batches: int | None,
    checkpoint_path: Path,
    extra_log_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError(f"No trainable parameters found for stage '{stage_name}'")

    optimizer = torch.optim.Adam(trainable_parameters, lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    best_val_rmse = float("inf")
    best_epoch = -1
    history: list[dict[str, object]] = []
    extra_log_fields = extra_log_fields or {}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_mse = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            max_batches=max_train_batches,
            target_scale=target_scale,
            grad_clip_norm=grad_clip_norm,
        )
        val_metrics = evaluate(model, val_loader, device, target_scale)
        epoch_record = {
            "stage": stage_name,
            "epoch": epoch,
            "train_mse": train_mse,
            "train_rmse": float(np.sqrt(train_mse) * target_scale),
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_score": val_metrics["score"],
            **extra_log_fields,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False))

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = float(val_metrics["rmse"])
            best_epoch = int(epoch)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "stage": stage_name,
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_val_rmse": best_val_rmse,
                    "extra_log_fields": extra_log_fields,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return {
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }


def train_one_seed(
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    task_output_dir: Path,
    task_name: str,
    source_train_full: CMAPSSSplit,
    source_test_raw: CMAPSSSplit,
    target_train_full: CMAPSSSplit,
    target_test_raw: CMAPSSSplit,
) -> dict[str, object]:
    set_seed(seed)
    target_scale = float(args.target_scale)
    grad_clip_norm = None if args.grad_clip_norm <= 0 else float(args.grad_clip_norm)
    run_dir = task_output_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    source_split = load_or_create_source_split(args, source_train_full, task_output_dir, seed)
    source_loaders, source_meta = build_source_stage_data(args, source_train_full, source_test_raw, source_split)
    model = build_model(args, int(source_meta["train_windows_shape"][2])).to(device)

    source_stage = fit_stage(
        stage_name="source",
        model=model,
        train_loader=source_loaders["train"],
        val_loader=source_loaders["val"],
        device=device,
        epochs=args.source_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        target_scale=target_scale,
        grad_clip_norm=grad_clip_norm,
        max_train_batches=args.max_source_train_batches,
        checkpoint_path=run_dir / "source" / "best.pt",
        extra_log_fields={"seed": seed, "task": task_name},
    )
    source_test_metrics = evaluate(model, source_loaders["test"], device, target_scale)

    target_direct_loader, target_direct_meta = build_target_direct_test_loader(args, target_train_full, target_test_raw)
    target_direct_metrics = evaluate(model, target_direct_loader, device, target_scale)

    result: dict[str, object] = {
        "seed": seed,
        "task": task_name,
        "source_split_path": source_split["path"],
        "source_split_seed": int(source_split["split_seed"]),
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
    }

    if args.finetune_mode != "none":
        if args.target_shots <= 0:
            raise ValueError("target_shots must be > 0 when finetune_mode is not 'none'")

        target_partition = load_or_create_target_partition(args, target_train_full, task_output_dir, seed)
        target_loaders, target_meta = build_target_finetune_data(args, target_train_full, target_test_raw, target_partition)
        finetune_model = build_model(args, int(source_meta["train_windows_shape"][2])).to(device)
        source_checkpoint = torch.load(source_stage["checkpoint"], map_location=device)
        finetune_model.load_state_dict(source_checkpoint["model_state_dict"])
        set_finetune_mode(finetune_model, args.finetune_mode)

        finetune_stage = fit_stage(
            stage_name=f"target_{args.finetune_mode}",
            model=finetune_model,
            train_loader=target_loaders["train"],
            val_loader=target_loaders["val"],
            device=device,
            epochs=args.target_epochs,
            lr=args.target_lr if args.target_lr is not None else args.lr,
            weight_decay=args.target_weight_decay if args.target_weight_decay is not None else args.weight_decay,
            target_scale=target_scale,
            grad_clip_norm=grad_clip_norm,
            max_train_batches=args.max_target_train_batches,
            checkpoint_path=run_dir / "target_finetune" / "best.pt",
            extra_log_fields={"seed": seed, "task": task_name, "mode": args.finetune_mode},
        )
        finetune_test_metrics = evaluate(finetune_model, target_loaders["test"], device, target_scale)
        result["target_partition_path"] = target_partition["path"]
        result["few_shot_seed"] = int(target_partition["few_shot_seed"])
        result["target_finetune"] = {
            "mode": args.finetune_mode,
            "best_epoch": int(finetune_stage["best_epoch"]),
            "best_val_rmse": float(finetune_stage["best_val_rmse"]),
            "checkpoint": str(finetune_stage["checkpoint"]),
            "train_windows_shape": target_meta["train_windows_shape"],
            "val_windows_shape": target_meta["val_windows_shape"],
            "test_windows_shape": target_meta["test_windows_shape"],
            "labeled_unit_count": int(target_meta["labeled_unit_count"]),
            "val_unit_count": int(target_meta["val_unit_count"]),
            "unlabeled_unit_count": int(target_meta["unlabeled_unit_count"]),
            "test_rmse": float(finetune_test_metrics["rmse"]),
            "test_mae": float(finetune_test_metrics["mae"]),
            "test_score": float(finetune_test_metrics["score"]),
        }

    with (run_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def summarize_results(
    args: argparse.Namespace,
    task_name: str,
    source_subset: str,
    target_subset: str,
    run_results: list[dict[str, object]],
    start_time: float,
) -> dict[str, object]:
    direct_rmses = [float(item["target_direct"]["rmse"]) for item in run_results]
    direct_scores = [float(item["target_direct"]["score"]) for item in run_results]
    source_test_rmses = [float(item["source_test"]["rmse"]) for item in run_results]
    best_source_run = min(run_results, key=lambda item: float(item["source_stage"]["best_val_rmse"]))
    best_direct_run = min(run_results, key=lambda item: float(item["target_direct"]["rmse"]))

    summary: dict[str, object] = {
        "task": task_name,
        "source_subset": source_subset,
        "target_subset": target_subset,
        "seeds": [int(item["seed"]) for item in run_results],
        "num_runs": len(run_results),
        "finetune_mode": args.finetune_mode,
        "target_shots": int(args.target_shots),
        "target_val_units": int(args.target_val_units),
        "best_source_seed": int(best_source_run["seed"]),
        "best_source_val_rmse": float(best_source_run["source_stage"]["best_val_rmse"]),
        "best_direct_target_seed": int(best_direct_run["seed"]),
        "best_direct_target_rmse": float(best_direct_run["target_direct"]["rmse"]),
        "mean_source_test_rmse": float(np.mean(source_test_rmses)),
        "std_source_test_rmse": float(np.std(source_test_rmses)),
        "mean_direct_target_rmse": float(np.mean(direct_rmses)),
        "std_direct_target_rmse": float(np.std(direct_rmses)),
        "mean_direct_target_score": float(np.mean(direct_scores)),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }

    finetune_run_results = [item for item in run_results if "target_finetune" in item]
    if finetune_run_results:
        finetune_runs = [item["target_finetune"] for item in finetune_run_results]
        finetune_rmses = [float(item["test_rmse"]) for item in finetune_runs]
        finetune_scores = [float(item["test_score"]) for item in finetune_runs]
        best_finetune_run = min(finetune_run_results, key=lambda item: float(item["target_finetune"]["best_val_rmse"]))
        summary.update(
            {
                "best_finetune_seed": int(best_finetune_run["seed"]),
                "best_finetune_val_rmse": float(best_finetune_run["target_finetune"]["best_val_rmse"]),
                "best_finetune_test_rmse": float(best_finetune_run["target_finetune"]["test_rmse"]),
                "mean_finetune_test_rmse": float(np.mean(finetune_rmses)),
                "std_finetune_test_rmse": float(np.std(finetune_rmses)),
                "mean_finetune_test_score": float(np.mean(finetune_scores)),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal cross-domain baseline runner for CD-MambAtt phase 1.")
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
    parser.add_argument("--target-shots", type=int, default=0, help="Number of labeled target engines used for few-shot fine-tuning")
    parser.add_argument("--target-val-units", type=int, default=10, help="Number of target training engines reserved for few-shot validation")
    parser.add_argument("--few-shot-seed", type=int, default=42, help="Seed used to sample target few-shot partitions")
    parser.add_argument("--target-partition-path", default=None, help="Optional JSON path for a fixed target few-shot partition")
    parser.add_argument("--resample-few-shot-per-seed", action="store_true", help="Resample the target few-shot partition for each run seed")
    parser.add_argument("--finetune-mode", choices=("none", "head", "full"), default="none", help="Target adaptation mode")
    parser.add_argument("--seeds", default=None, help="Optional comma-separated seed list; overrides --num-runs")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of run seeds when --seeds is not provided")
    parser.add_argument("--base-seed", type=int, default=42, help="First seed used when --seeds is not provided")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--source-epochs", type=int, default=50, help="Source supervised training epochs")
    parser.add_argument("--target-epochs", type=int, default=20, help="Target few-shot fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Source-stage learning rate")
    parser.add_argument("--target-lr", type=float, default=None, help="Optional target-stage learning rate override")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Source-stage weight decay")
    parser.add_argument("--target-weight-decay", type=float, default=None, help="Optional target-stage weight decay override")
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
    parser.add_argument("--target-val-all-windows", action="store_true", help="Validate target few-shot stage on all target validation windows")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--max-source-train-batches", type=int, default=None, help="Optional cap for source-stage smoke tests")
    parser.add_argument("--max-target-train-batches", type=int, default=None, help="Optional cap for target-stage smoke tests")
    parser.add_argument("--output-dir", default="/home/shelterpl/cd_mambatt/runs/cross_domain_baseline", help="Run output directory")
    args = parser.parse_args()

    task = resolve_cross_domain_task(
        args.task,
        source_subset=args.source_subset,
        target_subset=args.target_subset,
    )
    device = infer_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("The cross-domain MambAtt baseline requires CUDA in the configured environment.")

    task_output_dir = Path(args.output_dir).expanduser().resolve() / task.name
    task_output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds, args.num_runs, args.base_seed)
    if not seeds:
        raise ValueError("At least one seed must be provided")

    source_train_full = load_cmapss_split(args.root, task.source_subset, "train", rul_clip=args.rul_clip)
    source_test_raw = load_cmapss_split(args.root, task.source_subset, "test", rul_clip=args.rul_clip)
    target_train_full = load_cmapss_split(args.root, task.target_subset, "train", rul_clip=args.rul_clip)
    target_test_raw = load_cmapss_split(args.root, task.target_subset, "test", rul_clip=args.rul_clip)

    run_results: list[dict[str, object]] = []
    start_time = time.time()
    for seed in seeds:
        result = train_one_seed(
            args,
            seed,
            device,
            task_output_dir,
            task.name,
            source_train_full,
            source_test_raw,
            target_train_full,
            target_test_raw,
        )
        print(json.dumps(result, ensure_ascii=False))
        run_results.append(result)

    summary = summarize_results(args, task.name, task.source_subset, task.target_subset, run_results, start_time)
    with (task_output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"runs": run_results, "summary": summary}, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
