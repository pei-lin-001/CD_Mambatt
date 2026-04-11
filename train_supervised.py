from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cd_mambatt.data import (
    CMAPSSWindowDataset,
    build_windows,
    fit_normalizer,
    get_train_validation_unit_ids,
    load_cmapss_split,
    select_units,
)
from cd_mambatt.metrics import mae, nasa_score, rmse
from cd_mambatt.models import MambAttRegressor


def parse_seeds(seed_text: str | None, num_runs: int, base_seed: int) -> list[int]:
    if seed_text:
        return [int(token.strip()) for token in seed_text.split(",") if token.strip()]
    if num_runs <= 0:
        raise ValueError("num_runs must be > 0 when seeds are not provided explicitly")
    return [base_seed + offset for offset in range(num_runs)]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_loader(dataset: CMAPSSWindowDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
    target_scale: float,
    grad_clip_norm: float | None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_items = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, (windows, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            windows = windows.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True) / target_scale
            predictions = model(windows)
            loss = criterion(predictions, targets)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

            batch_size = int(windows.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_items += batch_size

    if total_items == 0:
        raise ValueError("No batches were processed")
    return total_loss / total_items


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_scale: float,
    *,
    domain_label_value: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    tgts: list[np.ndarray] = []
    with torch.no_grad():
        for windows, targets in loader:
            windows = windows.to(device, non_blocking=True)
            domain_label = None
            if domain_label_value is not None:
                domain_label = torch.full(
                    (windows.shape[0],),
                    int(domain_label_value),
                    dtype=torch.long,
                    device=device,
                )
            outputs = (model(windows, domain_label=domain_label) * target_scale).detach().cpu().numpy()
            preds.append(outputs.astype(np.float32))
            tgts.append(targets.numpy().astype(np.float32))
    return np.concatenate(preds), np.concatenate(tgts)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_scale: float,
    *,
    domain_label_value: int | None = None,
) -> dict[str, float]:
    predictions, targets = predict(
        model,
        loader,
        device,
        target_scale,
        domain_label_value=domain_label_value,
    )
    return {
        "rmse": rmse(predictions, targets),
        "mae": mae(predictions, targets),
        "score": nasa_score(predictions, targets),
    }


def resolve_split_path(base_path: Path, seed: int, resample_per_seed: bool) -> Path:
    if not resample_per_seed:
        return base_path
    return base_path.with_name(f"{base_path.stem}_seed{seed}{base_path.suffix}")


def resolve_split_definition(args: argparse.Namespace, train_full, output_dir: Path, seed: int) -> dict[str, object]:
    base_split_path = Path(args.split_path).expanduser().resolve() if args.split_path else output_dir / "engine_split.json"
    split_path = resolve_split_path(base_split_path, seed, args.resample_split_per_seed)
    split_seed = seed if args.resample_split_per_seed else args.split_seed
    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        train_units = np.asarray(payload["train_units"], dtype=np.int32)
        val_units = np.asarray(payload["val_units"], dtype=np.int32)
    else:
        train_units, val_units = get_train_validation_unit_ids(
            train_full,
            train_ratio=args.train_ratio,
            seed=split_seed,
        )
        payload = {
            "subset": args.subset.upper(),
            "train_ratio": args.train_ratio,
            "split_seed": split_seed,
            "source_seed": seed,
            "train_units": train_units.tolist(),
            "val_units": val_units.tolist(),
        }
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with split_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    expected_units = np.unique(train_full.unit_ids)
    actual_units = np.sort(np.concatenate([train_units, val_units]))
    if expected_units.shape != actual_units.shape or not np.array_equal(expected_units, actual_units):
        raise ValueError("Split definition does not match the training subset units")

    return {
        "path": str(split_path),
        "split_seed": split_seed,
        "train_units": train_units,
        "val_units": val_units,
    }


def build_dataloaders(args: argparse.Namespace, split_definition: dict[str, object]) -> tuple[dict[str, DataLoader], dict[str, object]]:
    train_full = load_cmapss_split(args.root, args.subset, "train", rul_clip=args.rul_clip)
    test_raw = load_cmapss_split(args.root, args.subset, "test", rul_clip=args.rul_clip)
    train_raw = select_units(train_full, split_definition["train_units"])
    val_raw = select_units(train_full, split_definition["val_units"])

    normalizer_source = train_raw if args.normalizer_fit_scope == "train_only" else train_full
    normalizer = fit_normalizer(normalizer_source)
    train_split = normalizer.transform(train_raw)
    val_split = normalizer.transform(val_raw)
    test_split = normalizer.transform(test_raw)

    train_windows = build_windows(train_split, args.window_size, stride=args.stride, last_only=False)
    val_windows = build_windows(val_split, args.window_size, stride=args.stride, last_only=not args.val_all_windows)
    test_windows = build_windows(test_split, args.window_size, stride=args.stride, last_only=True)

    loaders = {
        "train": build_loader(CMAPSSWindowDataset(train_windows), args.batch_size, True, args.num_workers),
        "val": build_loader(CMAPSSWindowDataset(val_windows), args.batch_size, False, args.num_workers),
        "test": build_loader(CMAPSSWindowDataset(test_windows), args.batch_size, False, args.num_workers),
    }
    metadata = {
        "normalizer": normalizer,
        "train_windows_shape": train_windows.shape,
        "val_windows_shape": val_windows.shape,
        "test_windows_shape": test_windows.shape,
        "train_unit_count": train_raw.num_units,
        "val_unit_count": val_raw.num_units,
        "split_path": split_definition["path"],
    }
    return loaders, metadata


def build_model(args: argparse.Namespace, input_dim: int) -> MambAttRegressor:
    return MambAttRegressor(
        input_dim=input_dim,
        d_model=args.d_model or input_dim,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        num_mamba_layers=args.num_mamba_layers,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward,
        transformer_impl=args.transformer_impl,
        transformer_norm_mode=args.transformer_norm_mode,
        transformer_inner_dropout=args.transformer_inner_dropout,
        mamba_block_mode=args.mamba_block_mode,
        spd_gate_init_bias=float(getattr(args, "spd_gate_init_bias", -2.0)),
        spd_scan_mode=str(getattr(args, "spd_scan_mode", "mixed")),
        spd_gate_mode=str(getattr(args, "spd_gate_mode", "token")),
        spd_gate_scheme=str(getattr(args, "spd_gate_scheme", "shared")),
        spd_predictor_mode=str(getattr(args, "spd_predictor_mode", "shared_head")),
        domain_conditioned_gate=bool(getattr(args, "domain_conditioned_gate", False)),
        frontend_adapter_mode=str(getattr(args, "frontend_adapter_mode", "none")),
        transformer_domain_adapter_mode=str(getattr(args, "transformer_domain_adapter_mode", "none")),
    )


def train_one_seed(
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    output_dir: Path,
    train_full,
) -> dict[str, object]:
    set_seed(seed)
    split_definition = resolve_split_definition(args, train_full, output_dir, seed)
    loaders, metadata = build_dataloaders(args, split_definition)
    input_dim = int(metadata["train_windows_shape"][2])
    model = build_model(args, input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    target_scale = float(args.target_scale)
    grad_clip_norm = None if args.grad_clip_norm <= 0 else float(args.grad_clip_norm)

    best_val_rmse = float("inf")
    best_epoch = -1
    history: list[dict[str, float | int]] = []
    run_dir = output_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_mse = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
            target_scale=target_scale,
            grad_clip_norm=grad_clip_norm,
        )
        val_metrics = evaluate(model, loaders["val"], device, target_scale)
        epoch_record = {
            "seed": seed,
            "epoch": epoch,
            "train_mse": train_mse,
            "train_rmse": float(np.sqrt(train_mse) * target_scale),
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_score": val_metrics["score"],
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False))

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "seed": seed,
                    "history": history,
                    "normalizer": metadata["normalizer"].to_dict(),
                },
                ckpt_path,
            )

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, loaders["test"], device, target_scale)
    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_score": test_metrics["score"],
        "checkpoint": str(ckpt_path),
        "train_windows_shape": metadata["train_windows_shape"],
        "val_windows_shape": metadata["val_windows_shape"],
        "test_windows_shape": metadata["test_windows_shape"],
        "train_unit_count": metadata["train_unit_count"],
        "val_unit_count": metadata["val_unit_count"],
        "split_path": metadata["split_path"],
        "split_seed": int(split_definition["split_seed"]),
        "val_all_windows": bool(args.val_all_windows),
    }
    with (run_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised MambAtt baseline runner aligned to the paper setup.")
    parser.add_argument("--root", default="/home/shelterpl/data/CMAPSS", help="CMAPSS root directory")
    parser.add_argument("--subset", default="FD001", help="One of FD001, FD002, FD003, FD004")
    parser.add_argument("--window-size", type=int, default=20, help="Sliding window size from the paper")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride")
    parser.add_argument("--rul-clip", type=int, default=125, help="Piece-wise linear RUL cap")
    parser.add_argument("--target-scale", type=float, default=1.0, help="Optional label scale divisor used during training; predictions are rescaled for evaluation")
    parser.add_argument("--grad-clip-norm", type=float, default=0.0, help="Clip gradient norm during training when > 0")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Engine-level train/validation split")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed used once to generate the fixed engine-level train/validation split")
    parser.add_argument("--split-path", default=None, help="Optional JSON path for a fixed engine split; created if missing")
    parser.add_argument("--resample-split-per-seed", action="store_true", help="Resample the engine-level 80/20 split for each run using that run's seed; closer to the paper protocol than reusing one fixed split")
    parser.add_argument("--normalizer-fit-scope", choices=("train_only", "trainval"), default="train_only", help="Whether normalization statistics are fit on the training subset only or the full train split before the 80/20 validation split")
    parser.add_argument("--seeds", default=None, help="Optional comma-separated seed list; overrides --num-runs")
    parser.add_argument("--num-runs", type=int, default=50, help="Number of random-seed trials from the paper")
    parser.add_argument("--base-seed", type=int, default=42, help="First seed used when --seeds is not provided")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size from the paper")
    parser.add_argument("--epochs", type=int, default=50, help="Per-run training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay")
    parser.add_argument("--d-model", type=int, default=None, help="Optional exploratory model width; paper-aligned supervised reproduction keeps the external Mamba/Transformer width at input_dim=21")
    parser.add_argument("--d-state", type=int, default=16, help="Mamba d_state")
    parser.add_argument("--d-conv", type=int, default=8, help="Mamba d_conv")
    parser.add_argument("--expand", type=int, default=2, help="Mamba expand factor")
    parser.add_argument("--num-mamba-layers", type=int, default=1, help="Number of Mamba blocks")
    parser.add_argument("--num-transformer-layers", type=int, default=3, help="Number of Transformer layers")
    parser.add_argument("--num-heads", type=int, default=7, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.5, help="Final dropout rate before the regression head")
    parser.add_argument("--dim-feedforward", type=int, default=2048, help="Transformer FFN hidden size")
    parser.add_argument("--transformer-impl", choices=("custom", "torch"), default="custom", help="Transformer head implementation")
    parser.add_argument("--transformer-norm-mode", choices=("pre", "post"), default="pre", help="Transformer normalization order")
    parser.add_argument("--transformer-inner-dropout", type=float, default=0.0, help="Dropout applied inside Transformer residual branches")
    parser.add_argument("--mamba-block-mode", choices=("bare", "prenorm_residual", "dd_spd"), default="bare", help="Mamba block mode: original bare block, pre-norm residual wrapper, or SPD-style DD-Mamba block")
    parser.add_argument("--spd-gate-init-bias", type=float, default=-2.0, help="Initial bias for the SPD gate when --mamba-block-mode=dd_spd")
    parser.add_argument("--spd-scan-mode", choices=("mixed", "dual_state"), default="mixed", help="SPD scan mode: original parameter-mixing scan or dual-state isolated scan")
    parser.add_argument("--spd-gate-mode", choices=("token", "window"), default="token", help="SPD gate mode: token-wise gate or window-level shared gate")
    parser.add_argument("--spd-gate-scheme", choices=("shared", "dt_bc"), default="shared", help="SPD gate scheme: one shared gate or separate dt/bc gates")
    parser.add_argument("--domain-conditioned-gate", action="store_true", help="Enable domain-conditioned scalar gate shift inside DD-Mamba")
    parser.add_argument("--frontend-adapter-mode", choices=("none", "target_affine", "target_residual"), default="none", help="Optional domain-conditioned frontend adapter inserted after conv1d inside DD-Mamba")
    parser.add_argument("--transformer-domain-adapter-mode", choices=("none", "target_shift", "target_film"), default="none", help="Optional target-conditioned affine adapter injected inside each custom Transformer block")
    parser.add_argument("--val-all-windows", action="store_true", help="Validate on all sliding windows instead of only the last window per validation engine")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional cap for smoke tests")
    parser.add_argument("--output-dir", default="/home/shelterpl/cd_mambatt/runs/supervised", help="Run output directory")
    args = parser.parse_args()

    device = infer_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("The supervised MambAtt baseline requires CUDA in the configured environment.")

    output_dir = Path(args.output_dir).expanduser().resolve() / args.subset.upper()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds, args.num_runs, args.base_seed)
    if not seeds:
        raise ValueError("At least one seed must be provided")
    train_full = load_cmapss_split(args.root, args.subset, "train", rul_clip=args.rul_clip)

    run_results: list[dict[str, object]] = []
    start_time = time.time()
    for seed in seeds:
        run_result = train_one_seed(args, seed, device, output_dir, train_full)
        print(json.dumps(run_result, ensure_ascii=False))
        run_results.append(run_result)

    best_run = min(run_results, key=lambda item: float(item["best_val_rmse"]))
    split_mode = "per_seed" if args.resample_split_per_seed else "fixed"
    summary = {
        "subset": args.subset.upper(),
        "device": str(device),
        "seeds": seeds,
        "num_runs": len(run_results),
        "split_mode": split_mode,
        "split_seed": None if args.resample_split_per_seed else args.split_seed,
        "split_path": None if args.resample_split_per_seed else str(best_run["split_path"]),
        "split_paths": [str(item["split_path"]) for item in run_results],
        "split_seeds": [int(item["split_seed"]) for item in run_results],
        "train_unit_count": int(run_results[0]["train_unit_count"]),
        "val_unit_count": int(run_results[0]["val_unit_count"]),
        "best_seed": best_run["seed"],
        "best_val_rmse": best_run["best_val_rmse"],
        "best_test_rmse": best_run["test_rmse"],
        "best_test_score": best_run["test_score"],
        "val_all_windows": bool(args.val_all_windows),
        "transformer_domain_adapter_mode": str(getattr(args, "transformer_domain_adapter_mode", "none")),
        "mean_test_rmse": float(np.mean([float(item["test_rmse"]) for item in run_results])),
        "std_test_rmse": float(np.std([float(item["test_rmse"]) for item in run_results])),
        "mean_test_score": float(np.mean([float(item["test_score"]) for item in run_results])),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"runs": run_results, "summary": summary}, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
