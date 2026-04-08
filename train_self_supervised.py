from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cd_mambatt.cross_domain import build_few_shot_target_partition
from cd_mambatt.data import CMAPSSWindowDataset, build_windows, fit_normalizer, load_cmapss_split, select_units
from cd_mambatt.metrics import mae, nasa_score, rmse
from cd_mambatt.self_supervised import (
    IndexedWindowPairDataset,
    IndexedWindowTripletDataset,
    MambAttSelfSupervisedPretrainer,
    WindowPseudoLabelDataset,
    assign_window_sensor_pseudo_labels,
    build_consecutive_pair_indices,
    build_temporal_triplet_indices,
    fit_sensor_pseudo_label_statistics,
    ntuplet_loss,
)
from train_supervised import build_loader, build_model, infer_device, parse_seeds, run_epoch, set_seed


def build_ssl_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def endless_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, target_scale: float) -> dict[str, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for windows, batch_targets in loader:
            windows = windows.to(device, non_blocking=True)
            outputs = (model(windows) * target_scale).detach().cpu().numpy()
            predictions.append(outputs.astype(np.float32))
            targets.append(batch_targets.numpy().astype(np.float32))
    prediction_array = np.concatenate(predictions)
    target_array = np.concatenate(targets)
    return {
        "rmse": rmse(prediction_array, target_array),
        "mae": mae(prediction_array, target_array),
        "score": nasa_score(prediction_array, target_array),
    }


def freeze_mambatt_encoder(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if name.startswith("input_proj.") or name.startswith("mamba_blocks."):
            parameter.requires_grad = False


def load_encoder_weights(model: nn.Module, encoder_state_dict: dict[str, torch.Tensor]) -> list[str]:
    current_state = model.state_dict()
    matched = {
        key: value
        for key, value in encoder_state_dict.items()
        if key in current_state and current_state[key].shape == value.shape
    }
    model.load_state_dict(matched, strict=False)
    return sorted(matched)


def apply_ssl_preset(args: argparse.Namespace) -> None:
    preset = args.ssl_preset
    if preset == "custom":
        return
    if preset == "paper_full":
        args.lambda_temporal = 0.25
        args.lambda_ntuplet = 0.25
        args.lambda_pseudo = 0.5
    elif preset == "paper_temp_ntuplet":
        args.lambda_temporal = 0.5
        args.lambda_ntuplet = 0.5
        args.lambda_pseudo = 0.0
    elif preset == "temporal_only":
        args.lambda_temporal = 1.0
        args.lambda_ntuplet = 0.0
        args.lambda_pseudo = 0.0
    else:
        raise ValueError(f"Unknown ssl preset: {preset}")


def build_ssl_pretrainer(args: argparse.Namespace, input_dim: int, pseudo_sensor_indices: np.ndarray) -> MambAttSelfSupervisedPretrainer:
    model_width = args.d_model or input_dim
    return MambAttSelfSupervisedPretrainer(
        input_dim=input_dim,
        d_model=model_width,
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
        window_size=args.window_size,
        pseudo_sensor_indices=pseudo_sensor_indices,
        temporal_hidden_dim=args.temporal_hidden_dim,
        pseudo_hidden_dim=args.pseudo_hidden_dim,
        ntuplet_mode=args.ntuplet_mode,
        ntuplet_normalize=args.ntuplet_normalize,
    )


def pretrain_ssl(
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
    train_scaled,
) -> dict[str, object]:
    unlabeled_windows = build_windows(train_scaled, args.window_size, stride=args.stride, last_only=False)
    input_dim = int(unlabeled_windows.windows.shape[2])
    if (args.d_model or input_dim) != input_dim:
        raise ValueError(
            "Self-supervised reproduction currently expects d_model == input_dim "
            "because the pseudo-label loss is defined on the original 21-sensor width."
        )

    triplet_indices, triplet_labels = build_temporal_triplet_indices(unlabeled_windows)
    pair_indices = build_consecutive_pair_indices(unlabeled_windows)
    pseudo_statistics = fit_sensor_pseudo_label_statistics(train_scaled.sensors)
    pseudo_labels = assign_window_sensor_pseudo_labels(
        unlabeled_windows.windows,
        pseudo_statistics,
        value_mode=args.pseudo_value_mode,
    )

    triplet_loader = build_ssl_loader(
        IndexedWindowTripletDataset(unlabeled_windows.windows, triplet_indices, triplet_labels),
        batch_size=args.pretrain_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    pair_loader = build_ssl_loader(
        IndexedWindowPairDataset(unlabeled_windows.windows, pair_indices),
        batch_size=args.pretrain_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    pseudo_loader = None
    if args.lambda_pseudo > 0:
        pseudo_loader = build_ssl_loader(
            WindowPseudoLabelDataset(unlabeled_windows.windows, pseudo_labels),
            batch_size=args.pretrain_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

    model = build_ssl_pretrainer(args, input_dim, pseudo_statistics.sensor_indices).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr, weight_decay=args.weight_decay)
    bce_criterion = nn.BCEWithLogitsLoss()
    ce_criterion = nn.CrossEntropyLoss()
    grad_clip_norm = None if args.grad_clip_norm <= 0 else float(args.grad_clip_norm)

    loaders = [triplet_loader, pair_loader]
    if pseudo_loader is not None:
        loaders.append(pseudo_loader)
    num_steps = max(len(loader) for loader in loaders)
    if args.max_pretrain_batches is not None:
        num_steps = min(num_steps, args.max_pretrain_batches)

    triplet_iter = endless_loader(triplet_loader)
    pair_iter = endless_loader(pair_loader)
    pseudo_iter = None if pseudo_loader is None else endless_loader(pseudo_loader)

    best_loss = float("inf")
    best_epoch = -1
    history: list[dict[str, float | int]] = []
    ckpt_path = run_dir / "ssl_pretrain_best.pt"

    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        total_loss = 0.0
        total_temporal = 0.0
        total_ntuplet = 0.0
        total_pseudo = 0.0

        for _ in range(num_steps):
            triplets, temporal_labels = next(triplet_iter)
            query_windows, positive_windows = next(pair_iter)

            triplets = triplets.to(device, non_blocking=True)
            temporal_labels = temporal_labels.to(device, non_blocking=True)
            query_windows = query_windows.to(device, non_blocking=True)
            positive_windows = positive_windows.to(device, non_blocking=True)

            temporal_logits = model.forward_temporal_logits(triplets)
            temporal_loss = bce_criterion(temporal_logits, temporal_labels)

            query_embeddings = model.forward_ntuplet_embeddings(query_windows)
            positive_embeddings = model.forward_ntuplet_embeddings(positive_windows)
            nt_loss = ntuplet_loss(query_embeddings, positive_embeddings)

            if pseudo_iter is not None:
                pseudo_windows, pseudo_targets = next(pseudo_iter)
                pseudo_windows = pseudo_windows.to(device, non_blocking=True)
                pseudo_targets = pseudo_targets.to(device, non_blocking=True)
                pseudo_logits = model.forward_pseudo_logits(pseudo_windows)
                pseudo_loss = ce_criterion(pseudo_logits.view(-1, 3), pseudo_targets.view(-1))
            else:
                pseudo_loss = query_embeddings.new_zeros(())

            loss = (
                args.lambda_temporal * temporal_loss
                + args.lambda_ntuplet * nt_loss
                + args.lambda_pseudo * pseudo_loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite self-supervised loss encountered at epoch={epoch}: "
                    f"temporal={float(temporal_loss.detach().cpu())}, "
                    f"ntuplet={float(nt_loss.detach().cpu())}, "
                    f"pseudo={float(pseudo_loss.detach().cpu())}"
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_temporal += float(temporal_loss.detach().cpu())
            total_ntuplet += float(nt_loss.detach().cpu())
            total_pseudo += float(pseudo_loss.detach().cpu())

        epoch_record = {
            "epoch": epoch,
            "ssl_loss": total_loss / num_steps,
            "temporal_loss": total_temporal / num_steps,
            "ntuplet_loss": total_ntuplet / num_steps,
            "pseudo_loss": total_pseudo / num_steps,
        }
        history.append(epoch_record)
        print(json.dumps({"stage": "ssl_pretrain", **epoch_record}, ensure_ascii=False))

        if epoch_record["ssl_loss"] < best_loss:
            best_loss = float(epoch_record["ssl_loss"])
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.export_encoder_state_dict(),
                    "history": history,
                    "pseudo_sensor_indices": pseudo_statistics.sensor_indices.tolist(),
                    "pseudo_boundary_b1": pseudo_statistics.boundary_b1.tolist(),
                    "pseudo_boundary_b2": pseudo_statistics.boundary_b2.tolist(),
                    "pseudo_value_mode": args.pseudo_value_mode,
                },
                ckpt_path,
            )

    checkpoint = torch.load(ckpt_path, map_location=device)
    return {
        "checkpoint": str(ckpt_path),
        "encoder_state_dict": checkpoint["encoder_state_dict"],
        "best_epoch": best_epoch,
        "best_ssl_loss": best_loss,
        "history": history,
        "num_unlabeled_windows": int(unlabeled_windows.windows.shape[0]),
        "num_temporal_triplets": int(triplet_indices.shape[0]),
        "num_ntuplet_pairs": int(pair_indices.shape[0]),
        "num_pseudo_sensors": int(pseudo_statistics.sensor_indices.shape[0]),
    }


def train_one_shot(
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    run_dir: Path,
    train_full,
    test_raw,
    encoder_state_dict: dict[str, torch.Tensor] | None,
) -> dict[str, object]:
    partition_seed = seed if args.resample_few_shot_per_seed else args.few_shot_seed
    partition = build_few_shot_target_partition(
        train_full,
        num_shots=args.shots,
        num_val_units=args.val_units,
        seed=partition_seed,
    )

    normalizer = fit_normalizer(train_full)
    train_scaled = normalizer.transform(train_full)
    test_scaled = normalizer.transform(test_raw)
    labeled_scaled = select_units(train_scaled, partition.labeled_units)
    validation_scaled = select_units(train_scaled, partition.validation_units)

    labeled_windows = build_windows(labeled_scaled, args.window_size, stride=args.stride, last_only=False)
    validation_windows = build_windows(
        validation_scaled,
        args.window_size,
        stride=args.stride,
        last_only=not args.val_all_windows,
    )
    test_windows = build_windows(test_scaled, args.window_size, stride=args.stride, last_only=True)

    loaders = {
        "train": build_loader(CMAPSSWindowDataset(labeled_windows), args.finetune_batch_size, True, args.num_workers),
        "val": build_loader(CMAPSSWindowDataset(validation_windows), args.finetune_batch_size, False, args.num_workers),
        "test": build_loader(CMAPSSWindowDataset(test_windows), args.finetune_batch_size, False, args.num_workers),
    }

    model = build_model(args, int(labeled_windows.windows.shape[2])).to(device)
    loaded_encoder_keys: list[str] = []
    if encoder_state_dict is not None:
        loaded_encoder_keys = load_encoder_weights(model, encoder_state_dict)
    if args.freeze_encoder:
        freeze_mambatt_encoder(model)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters remain for one-shot supervised training")

    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    target_scale = float(args.target_scale)
    grad_clip_norm = None if args.grad_clip_norm <= 0 else float(args.grad_clip_norm)

    best_val_rmse = float("inf")
    best_epoch = -1
    history: list[dict[str, float | int]] = []
    ckpt_path = run_dir / "one_shot_best.pt"

    for epoch in range(1, args.finetune_epochs + 1):
        train_mse = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            max_batches=args.max_finetune_batches,
            target_scale=target_scale,
            grad_clip_norm=grad_clip_norm,
        )
        val_metrics = evaluate(model, loaders["val"], device, target_scale)
        epoch_record = {
            "epoch": epoch,
            "train_mse": train_mse,
            "train_rmse": float(np.sqrt(train_mse) * target_scale),
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_score": val_metrics["score"],
        }
        history.append(epoch_record)
        print(json.dumps({"stage": "one_shot", **epoch_record}, ensure_ascii=False))
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "history": history,
                    "encoder_loaded_keys": loaded_encoder_keys,
                    "normalizer": normalizer.to_dict(),
                },
                ckpt_path,
            )

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, loaders["test"], device, target_scale)
    result = {
        "shots": int(args.shots),
        "val_units": int(args.val_units),
        "few_shot_seed": int(partition_seed),
        "labeled_units": partition.labeled_units.tolist(),
        "validation_units": partition.validation_units.tolist(),
        "unlabeled_units": partition.unlabeled_units.tolist(),
        "train_window_count": int(labeled_windows.windows.shape[0]),
        "val_window_count": int(validation_windows.windows.shape[0]),
        "test_window_count": int(test_windows.windows.shape[0]),
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_score": test_metrics["score"],
        "checkpoint": str(ckpt_path),
        "encoder_frozen": bool(args.freeze_encoder),
        "encoder_loaded_key_count": len(loaded_encoder_keys),
    }
    with (run_dir / "one_shot_result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def train_one_seed(args: argparse.Namespace, seed: int, device: torch.device, output_dir: Path) -> dict[str, object]:
    set_seed(seed)
    subset = args.subset.upper()
    run_dir = output_dir / subset / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_full = load_cmapss_split(args.root, subset, "train", rul_clip=args.rul_clip)
    test_raw = load_cmapss_split(args.root, subset, "test", rul_clip=args.rul_clip)

    ssl_result: dict[str, object] | None = None
    encoder_state_dict: dict[str, torch.Tensor] | None = None
    if not args.skip_pretrain:
        train_scaled = fit_normalizer(train_full).transform(train_full)
        ssl_result = pretrain_ssl(args, device, run_dir, train_scaled)
        encoder_state_dict = ssl_result["encoder_state_dict"]

    one_shot_result = train_one_shot(args, seed, device, run_dir, train_full, test_raw, encoder_state_dict)
    result = {
        "seed": seed,
        "subset": subset,
        "ssl_preset": args.ssl_preset,
        "lambda_temporal": float(args.lambda_temporal),
        "lambda_ntuplet": float(args.lambda_ntuplet),
        "lambda_pseudo": float(args.lambda_pseudo),
        "ntuplet_mode": args.ntuplet_mode,
        "ntuplet_normalize": bool(args.ntuplet_normalize),
        "ssl": None if ssl_result is None else {
            key: value for key, value in ssl_result.items() if key != "encoder_state_dict"
        },
        "one_shot": one_shot_result,
    }
    with (run_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper-style self-supervised MambAtt pretraining + one-shot evaluation on the same C-MAPSS subset."
    )
    parser.add_argument("--root", default="/home/shelterpl/data/CMAPSS", help="CMAPSS root directory")
    parser.add_argument("--subset", default="FD001", help="One of FD001, FD002, FD003, FD004")
    parser.add_argument("--window-size", type=int, default=20, help="Sliding window size")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride")
    parser.add_argument("--rul-clip", type=int, default=125, help="Piece-wise linear RUL cap")
    parser.add_argument("--target-scale", type=float, default=1.0, help="Optional target divisor during supervised fine-tuning")
    parser.add_argument("--grad-clip-norm", type=float, default=0.0, help="Clip gradient norm when > 0")
    parser.add_argument("--seeds", default=None, help="Optional comma-separated seed list")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of runs if --seeds is not provided")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed when --seeds is omitted")
    parser.add_argument("--few-shot-seed", type=int, default=42, help="Seed for one-shot engine selection when not resampling per run")
    parser.add_argument(
        "--resample-few-shot-per-seed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resample the one-shot labeled/validation engine selection for each run using that run's seed",
    )
    parser.add_argument("--shots", type=int, default=1, help="Number of labeled engines used for supervised fine-tuning")
    parser.add_argument("--val-units", type=int, default=1, help="Number of labeled validation engines")
    parser.add_argument("--pretrain-batch-size", type=int, default=64, help="Batch size for self-supervised pretraining")
    parser.add_argument("--finetune-batch-size", type=int, default=64, help="Batch size for one-shot supervised fine-tuning")
    parser.add_argument("--pretrain-epochs", type=int, default=50, help="Self-supervised pretraining epochs")
    parser.add_argument("--finetune-epochs", type=int, default=50, help="One-shot supervised fine-tuning epochs")
    parser.add_argument("--pretrain-lr", type=float, default=1e-3, help="Learning rate for self-supervised pretraining")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for one-shot supervised fine-tuning")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay")
    parser.add_argument("--ssl-preset", choices=("paper_full", "paper_temp_ntuplet", "temporal_only", "custom"), default="paper_full", help="Loss-weight preset mirroring the paper ablations")
    parser.add_argument("--lambda-temporal", type=float, default=0.25, help="Weight for temporal ordering loss (used when --ssl-preset=custom)")
    parser.add_argument("--lambda-ntuplet", type=float, default=0.25, help="Weight for N-tuplet loss (used when --ssl-preset=custom)")
    parser.add_argument("--lambda-pseudo", type=float, default=0.5, help="Weight for pseudo-label loss (used when --ssl-preset=custom)")
    parser.add_argument("--pseudo-value-mode", choices=("last", "mean"), default="last", help="How to convert a window into scalar sensor values for pseudo-label assignment")
    parser.add_argument("--temporal-hidden-dim", type=int, default=128, help="Hidden width of the temporal-order classifier")
    parser.add_argument("--pseudo-hidden-dim", type=int, default=64, help="Hidden width of the pseudo-label classifier")
    parser.add_argument("--ntuplet-mode", choices=("flatten", "last", "mean"), default="flatten", help="How to convert the encoder sequence into an N-tuplet embedding")
    parser.add_argument("--ntuplet-normalize", action=argparse.BooleanOptionalAction, default=False, help="Whether to L2-normalize N-tuplet embeddings before the paper Eq.12 inner-product loss")
    parser.add_argument(
        "--freeze-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze the Mamba encoder during one-shot supervised training (paper one-shot default: enabled)",
    )
    parser.add_argument("--skip-pretrain", action="store_true", help="Skip the self-supervised stage and run one-shot training directly")
    parser.add_argument(
        "--val-all-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate using all windows from the validation engine(s) (recommended for stable one-shot checkpoint selection)",
    )
    parser.add_argument("--d-model", type=int, default=None, help="Model width; paper-faithful self-supervised reproduction expects 21")
    parser.add_argument("--d-state", type=int, default=16, help="Mamba d_state")
    parser.add_argument("--d-conv", type=int, default=8, help="Mamba d_conv")
    parser.add_argument("--expand", type=int, default=2, help="Mamba expand factor")
    parser.add_argument("--num-mamba-layers", type=int, default=1, help="Number of Mamba blocks")
    parser.add_argument("--num-transformer-layers", type=int, default=3, help="Number of Transformer decoder blocks")
    parser.add_argument("--num-heads", type=int, default=7, help="Number of Transformer heads")
    parser.add_argument("--dropout", type=float, default=0.5, help="Final dropout rate")
    parser.add_argument("--dim-feedforward", type=int, default=2048, help="Transformer FFN hidden size")
    parser.add_argument("--transformer-impl", choices=("custom", "torch"), default="custom", help="Transformer implementation")
    parser.add_argument("--transformer-norm-mode", choices=("pre", "post"), default="pre", help="Transformer normalization order")
    parser.add_argument("--transformer-inner-dropout", type=float, default=0.0, help="Dropout inside Transformer residual branches")
    parser.add_argument("--mamba-block-mode", choices=("bare", "prenorm_residual", "dd_spd"), default="bare", help="Mamba block mode")
    parser.add_argument("--spd-gate-init-bias", type=float, default=-2.0, help="Passed through when building the supervised regressor")
    parser.add_argument("--spd-scan-mode", choices=("mixed", "dual_state"), default="mixed", help="Passed through when building the supervised regressor")
    parser.add_argument("--spd-gate-mode", choices=("token", "window"), default="token", help="Passed through when building the supervised regressor")
    parser.add_argument("--spd-gate-scheme", choices=("shared", "dt_bc"), default="shared", help="Passed through when building the supervised regressor")
    parser.add_argument("--spd-predictor-mode", choices=("shared_head", "decomposed_residual", "shared_aux_residual"), default="shared_head", help="Passed through when building the supervised regressor")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--max-pretrain-batches", type=int, default=None, help="Optional cap for self-supervised pretraining batches per epoch")
    parser.add_argument("--max-finetune-batches", type=int, default=None, help="Optional cap for one-shot fine-tuning batches per epoch")
    parser.add_argument("--output-dir", default="/home/shelterpl/cd_mambatt/runs/self_supervised", help="Output directory")
    args = parser.parse_args()

    apply_ssl_preset(args)
    if args.skip_pretrain and args.freeze_encoder:
        raise ValueError("Freezing the encoder while skipping pretraining is not meaningful for the paper-style one-shot setup")

    device = infer_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("The self-supervised MambAtt runner requires CUDA in the configured environment.")

    seeds = parse_seeds(args.seeds, args.num_runs, args.base_seed)
    if not seeds:
        raise ValueError("At least one seed must be provided")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_results: list[dict[str, object]] = []
    start_time = time.time()
    for seed in seeds:
        result = train_one_seed(args, seed, device, output_dir)
        print(json.dumps(result, ensure_ascii=False))
        run_results.append(result)

    one_shot_rmse = [float(item["one_shot"]["test_rmse"]) for item in run_results]
    summary = {
        "subset": args.subset.upper(),
        "device": str(device),
        "seeds": seeds,
        "num_runs": len(run_results),
        "ssl_preset": args.ssl_preset,
        "lambda_temporal": float(args.lambda_temporal),
        "lambda_ntuplet": float(args.lambda_ntuplet),
        "lambda_pseudo": float(args.lambda_pseudo),
        "ntuplet_mode": args.ntuplet_mode,
        "ntuplet_normalize": bool(args.ntuplet_normalize),
        "shots": int(args.shots),
        "val_units": int(args.val_units),
        "freeze_encoder": bool(args.freeze_encoder),
        "mean_test_rmse": float(np.mean(one_shot_rmse)),
        "std_test_rmse": float(np.std(one_shot_rmse)),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    subset_dir = output_dir / args.subset.upper()
    subset_dir.mkdir(parents=True, exist_ok=True)
    with (subset_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"runs": run_results, "summary": summary}, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
