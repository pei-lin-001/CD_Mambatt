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
from cd_mambatt.models import MambAttRegressor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def rmse_from_loss(loss_value: float) -> float:
    return float(loss_value ** 0.5)


def resolve_split_seed(split_seed: int, run_seed: int, resample_per_seed: bool) -> int:
    return run_seed if resample_per_seed else split_seed


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
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_items = 0

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for step, (windows, targets) in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            windows = windows.to(device)
            targets = targets.to(device) / target_scale

            predictions = model(windows)
            loss = criterion(predictions, targets)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

            batch_size = windows.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_items += int(batch_size)

    if total_items == 0:
        raise ValueError("No batches were processed")
    return total_loss / total_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal supervised MambAtt training entrypoint.")
    parser.add_argument("--root", default="/home/shelterpl/data/CMAPSS", help="CMAPSS root directory")
    parser.add_argument("--subset", default="FD001", help="One of FD001, FD002, FD003, FD004")
    parser.add_argument("--window-size", type=int, default=20, help="Paper-aligned sliding window size")
    parser.add_argument("--stride", type=int, default=1, help="Training window stride")
    parser.add_argument("--rul-clip", type=int, default=125, help="Piece-wise linear RUL cap")
    parser.add_argument("--target-scale", type=float, default=1.0, help="Optional label scale divisor used during training")
    parser.add_argument("--grad-clip-norm", type=float, default=0.0, help="Clip gradient norm during training when > 0")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/validation engine split")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed used to generate the fixed engine-level train/validation split")
    parser.add_argument("--resample-split-per-seed", action="store_true", help="Use the run seed itself to resample the engine-level 80/20 split")
    parser.add_argument("--normalizer-fit-scope", choices=("train_only", "trainval"), default="train_only", help="Whether normalization statistics are fit on the training subset only or the full train split before the 80/20 validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay")
    parser.add_argument("--d-model", type=int, default=None, help="Optional exploratory model width; paper-aligned supervised reproduction keeps the external Mamba/Transformer width at input_dim=21")
    parser.add_argument("--d-state", type=int, default=16, help="Mamba d_state")
    parser.add_argument("--d-conv", type=int, default=8, help="Mamba d_conv")
    parser.add_argument("--expand", type=int, default=2, help="Mamba expand factor")
    parser.add_argument("--num-mamba-layers", type=int, default=1, help="Number of residual Mamba blocks")
    parser.add_argument("--num-transformer-layers", type=int, default=3, help="Number of Transformer layers")
    parser.add_argument("--num-heads", type=int, default=7, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.5, help="Final dropout rate before the regression head")
    parser.add_argument("--dim-feedforward", type=int, default=2048, help="Transformer FFN hidden size")
    parser.add_argument("--transformer-impl", choices=("custom", "torch"), default="custom", help="Transformer head implementation")
    parser.add_argument("--transformer-norm-mode", choices=("pre", "post"), default="pre", help="Transformer normalization order")
    parser.add_argument("--transformer-inner-dropout", type=float, default=0.0, help="Dropout applied inside Transformer residual branches")
    parser.add_argument("--mamba-block-mode", choices=("bare", "prenorm_residual"), default="bare", help="Whether to wrap each Mamba layer with pre-norm residual")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional cap for smoke tests")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Optional cap for smoke tests")
    parser.add_argument("--output-dir", default="/home/shelterpl/cd_mambatt/runs/minimal", help="Output directory")
    args = parser.parse_args()

    set_seed(args.seed)
    device = infer_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This minimal MambAtt skeleton requires CUDA because Mamba is GPU-only in the configured environment.")

    train_full = load_cmapss_split(args.root, args.subset, "train", rul_clip=args.rul_clip)
    split_seed = resolve_split_seed(args.split_seed, args.seed, args.resample_split_per_seed)
    train_units, val_units = get_train_validation_unit_ids(train_full, train_ratio=args.train_ratio, seed=split_seed)
    train_raw = select_units(train_full, train_units)
    val_raw = select_units(train_full, val_units)
    normalizer_source = train_raw if args.normalizer_fit_scope == "train_only" else train_full
    normalizer = fit_normalizer(normalizer_source)
    train_split = normalizer.transform(train_raw)
    val_split = normalizer.transform(val_raw)

    train_windows = build_windows(train_split, args.window_size, stride=args.stride, last_only=False)
    val_windows = build_windows(val_split, args.window_size, stride=args.stride, last_only=True)

    train_dataset = CMAPSSWindowDataset(train_windows)
    val_dataset = CMAPSSWindowDataset(val_windows)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = MambAttRegressor(
        input_dim=train_split.feature_dim,
        d_model=args.d_model or train_split.feature_dim,
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
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    target_scale = float(args.target_scale)
    grad_clip_norm = None if args.grad_clip_norm <= 0 else float(args.grad_clip_norm)

    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / f"{args.subset.upper()}_best.pt"

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
            target_scale=target_scale,
            grad_clip_norm=grad_clip_norm,
        )
        val_loss = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            max_batches=args.max_val_batches,
            target_scale=target_scale,
            grad_clip_norm=None,
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_mse": train_loss,
            "train_rmse": rmse_from_loss(train_loss) * target_scale,
            "val_mse": val_loss,
            "val_rmse": rmse_from_loss(val_loss) * target_scale,
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "normalizer": normalizer.to_dict(),
                    "history": history,
                },
                best_path,
            )

    summary = {
        "subset": args.subset.upper(),
        "device": str(device),
        "split_mode": "per_seed" if args.resample_split_per_seed else "fixed",
        "split_seed": args.split_seed,
        "split_seed_used": split_seed,
        "train_unit_count": int(train_raw.num_units),
        "val_unit_count": int(val_raw.num_units),
        "train_windows": train_windows.shape,
        "val_windows": val_windows.shape,
        "best_val_rmse": rmse_from_loss(best_val_loss),
        "best_val_rmse_scaled": rmse_from_loss(best_val_loss) * target_scale,
        "best_checkpoint": str(best_path),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
