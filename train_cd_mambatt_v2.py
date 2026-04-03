from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cd_mambatt.data import (
    CMAPSSSplit,
    CMAPSSWindowPairDataset,
    build_monotonic_window_pairs,
    fit_normalizer,
    load_cmapss_split,
    select_units,
)
from cd_mambatt.losses import cross_domain_contrastive_loss, gaussian_mmd_loss, local_monotonicity_loss
from cd_mambatt.pseudo_labeling import (
    SourceStageStatistics,
    assign_pseudo_stage_labels,
    assign_rul_stage_labels,
    compute_source_stage_statistics,
)
from train_cd_mambatt_v1 import build_target_cd_data, fit_source_stage
from train_cross_domain_baseline import (
    build_source_stage_data,
    build_target_direct_test_loader,
    load_or_create_source_split,
    load_or_create_target_partition,
)
from train_supervised import build_loader, build_model, evaluate, infer_device, parse_seeds, set_seed


def endless_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def resolve_pseudo_quantile(
    epoch: int,
    total_epochs: int,
    *,
    start_quantile: float,
    end_quantile: float,
) -> float:
    if total_epochs <= 1:
        return float(end_quantile)
    progress = float(epoch - 1) / float(total_epochs - 1)
    return float(start_quantile + progress * (end_quantile - start_quantile))


def collect_source_stage_statistics(
    model: nn.Module,
    source_loader: DataLoader,
    device: torch.device,
    *,
    rul_clip: float,
    num_stages: int,
    quantile: float,
    normalize_features: bool,
) -> SourceStageStatistics:
    model.eval()
    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    rul_batches: list[torch.Tensor] = []

    with torch.no_grad():
        for windows, targets in source_loader:
            windows = windows.to(device, non_blocking=True)
            features = model.forward_features(windows).detach().cpu()
            rul_values = targets.detach().cpu()
            stage_labels = assign_rul_stage_labels(rul_values, rul_clip=rul_clip, num_stages=num_stages).cpu()
            feature_batches.append(features)
            label_batches.append(stage_labels)
            rul_batches.append(rul_values)

    all_features = torch.cat(feature_batches, dim=0)
    all_labels = torch.cat(label_batches, dim=0)
    all_rul = torch.cat(rul_batches, dim=0)
    stats = compute_source_stage_statistics(
        all_features,
        all_labels,
        all_rul,
        num_stages=num_stages,
        quantile=quantile,
        normalize_features=normalize_features,
    )
    return SourceStageStatistics(
        centroids=stats.centroids.to(device),
        thresholds=stats.thresholds.to(device),
        counts=stats.counts.to(device),
        mean_rul=stats.mean_rul.to(device),
        quantile=stats.quantile,
        normalize_features=stats.normalize_features,
    )


def build_target_monotonic_loader(
    args: argparse.Namespace,
    target_train_full: CMAPSSSplit,
    partition: dict[str, object],
) -> tuple[DataLoader, dict[str, object]]:
    train_units = np.sort(
        np.concatenate(
            [
                np.asarray(partition["labeled_units"], dtype=np.int32),
                np.asarray(partition["unlabeled_units"], dtype=np.int32),
            ]
        )
    )
    train_raw = select_units(target_train_full, train_units)
    normalizer = fit_normalizer(target_train_full)
    train_split = normalizer.transform(train_raw)
    pair_data = build_monotonic_window_pairs(
        train_split,
        args.window_size,
        stride=args.stride,
        pair_gap=args.monotonic_pair_gap,
        pair_stride=args.monotonic_pair_stride,
    )
    loader = build_loader(CMAPSSWindowPairDataset(pair_data), args.batch_size, True, args.num_workers)
    return loader, {
        "pair_count": int(pair_data.num_pairs),
        "pair_unit_count": int(train_raw.num_units),
        "pair_gap": int(args.monotonic_pair_gap),
        "pair_stride": int(args.monotonic_pair_stride),
    }


def run_cd_pseudo_epoch(
    model: nn.Module,
    stage_head: nn.Module,
    source_loader: DataLoader,
    target_labeled_loader: DataLoader,
    target_unlabeled_loader: DataLoader,
    target_monotonic_loader: DataLoader | None,
    stage_statistics: SourceStageStatistics,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    source_loss_weight: float,
    target_loss_weight: float,
    lambda_mmd: float,
    lambda_source_stage: float,
    lambda_pseudo: float,
    lambda_contrastive: float,
    lambda_monotonic: float,
    rul_clip: float,
    num_pseudo_stages: int,
    target_scale: float,
    grad_clip_norm: float | None,
    max_batches: int | None,
    mmd_sigmas: tuple[float, ...],
    contrastive_temperature: float,
    contrastive_normalize_features: bool,
    monotonic_margin: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    stage_head.train(is_train)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    source_iter = endless_loader(source_loader)
    target_labeled_iter = endless_loader(target_labeled_loader)
    target_unlabeled_iter = endless_loader(target_unlabeled_loader)
    target_monotonic_iter = None if target_monotonic_loader is None else endless_loader(target_monotonic_loader)
    num_steps = max(len(source_loader), len(target_labeled_loader), len(target_unlabeled_loader))
    if target_monotonic_loader is not None:
        num_steps = max(num_steps, len(target_monotonic_loader))
    if max_batches is not None:
        num_steps = min(num_steps, max_batches)

    total_examples = 0
    total_source_loss = 0.0
    total_target_loss = 0.0
    total_mmd_loss = 0.0
    total_source_stage_loss = 0.0
    total_pseudo_loss = 0.0
    total_contrastive_loss = 0.0
    total_monotonic_loss = 0.0
    total_combined_loss = 0.0
    total_pseudo_candidates = 0
    total_pseudo_accepted = 0
    total_pseudo_distance = 0.0
    total_contrastive_valid_anchor_ratio = 0.0
    total_contrastive_positive_count = 0.0
    total_contrastive_target_bank_size = 0.0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for _ in range(num_steps):
            source_windows, source_targets_raw = next(source_iter)
            target_labeled_windows, target_labeled_targets_raw = next(target_labeled_iter)
            target_unlabeled_windows, _ = next(target_unlabeled_iter)
            if target_monotonic_iter is not None:
                monotonic_earlier_windows, monotonic_later_windows = next(target_monotonic_iter)
                monotonic_earlier_windows = monotonic_earlier_windows.to(device, non_blocking=True)
                monotonic_later_windows = monotonic_later_windows.to(device, non_blocking=True)
            else:
                monotonic_earlier_windows = None
                monotonic_later_windows = None

            source_windows = source_windows.to(device, non_blocking=True)
            source_targets_raw = source_targets_raw.to(device, non_blocking=True)
            source_targets = source_targets_raw / target_scale

            target_labeled_windows = target_labeled_windows.to(device, non_blocking=True)
            target_labeled_targets_raw = target_labeled_targets_raw.to(device, non_blocking=True)
            target_labeled_targets = target_labeled_targets_raw / target_scale
            target_unlabeled_windows = target_unlabeled_windows.to(device, non_blocking=True)

            source_features = model.forward_features(source_windows)
            source_predictions = model.predict_from_features(source_features)
            source_loss = mse_loss(source_predictions, source_targets)

            source_stage_labels = assign_rul_stage_labels(
                source_targets_raw,
                rul_clip=rul_clip,
                num_stages=num_pseudo_stages,
            )
            source_stage_logits = stage_head(source_features)
            source_stage_loss = ce_loss(source_stage_logits, source_stage_labels)

            target_labeled_features = model.forward_features(target_labeled_windows)
            target_predictions = model.predict_from_features(target_labeled_features)
            target_loss = mse_loss(target_predictions, target_labeled_targets)
            target_labeled_stage_labels = assign_rul_stage_labels(
                target_labeled_targets_raw,
                rul_clip=rul_clip,
                num_stages=num_pseudo_stages,
            )

            target_unlabeled_features = model.forward_features(target_unlabeled_windows)
            target_alignment_features = torch.cat([target_labeled_features, target_unlabeled_features], dim=0)
            mmd_loss = gaussian_mmd_loss(source_features, target_alignment_features, sigmas=mmd_sigmas)

            pseudo_stage_labels, accepted_mask, pseudo_distances = assign_pseudo_stage_labels(
                target_unlabeled_features,
                stage_statistics,
            )
            accepted_count = int(accepted_mask.sum().item())
            candidate_count = int(target_unlabeled_features.shape[0])
            total_pseudo_candidates += candidate_count
            total_pseudo_accepted += accepted_count
            if accepted_count > 0:
                accepted_logits = stage_head(target_unlabeled_features[accepted_mask])
                pseudo_loss = ce_loss(accepted_logits, pseudo_stage_labels[accepted_mask])
                total_pseudo_distance += float(pseudo_distances[accepted_mask].sum().detach().cpu())
            else:
                pseudo_loss = source_features.new_zeros(())

            contrastive_target_features = [target_labeled_features]
            contrastive_target_labels = [target_labeled_stage_labels]
            if accepted_count > 0:
                contrastive_target_features.append(target_unlabeled_features[accepted_mask])
                contrastive_target_labels.append(pseudo_stage_labels[accepted_mask])
            contrastive_loss, contrastive_stats = cross_domain_contrastive_loss(
                source_features,
                source_stage_labels,
                torch.cat(contrastive_target_features, dim=0),
                torch.cat(contrastive_target_labels, dim=0),
                temperature=contrastive_temperature,
                normalize_features=contrastive_normalize_features,
            )

            if monotonic_earlier_windows is not None and monotonic_later_windows is not None:
                earlier_predictions = model(monotonic_earlier_windows)
                later_predictions = model(monotonic_later_windows)
                monotonic_loss = local_monotonicity_loss(
                    earlier_predictions,
                    later_predictions,
                    margin=monotonic_margin,
                )
            else:
                monotonic_loss = source_features.new_zeros(())

            total_loss = (
                source_loss_weight * source_loss
                + target_loss_weight * target_loss
                + lambda_mmd * mmd_loss
                + lambda_source_stage * source_stage_loss
                + lambda_pseudo * pseudo_loss
                + lambda_contrastive * contrastive_loss
                + lambda_monotonic * monotonic_loss
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(stage_head.parameters()),
                        max_norm=grad_clip_norm,
                    )
                optimizer.step()

            batch_examples = int(source_windows.shape[0])
            total_examples += batch_examples
            total_source_loss += float(source_loss.detach().cpu()) * batch_examples
            total_target_loss += float(target_loss.detach().cpu()) * batch_examples
            total_mmd_loss += float(mmd_loss.detach().cpu()) * batch_examples
            total_source_stage_loss += float(source_stage_loss.detach().cpu()) * batch_examples
            total_pseudo_loss += float(pseudo_loss.detach().cpu()) * batch_examples
            total_contrastive_loss += float(contrastive_loss.detach().cpu()) * batch_examples
            total_monotonic_loss += float(monotonic_loss.detach().cpu()) * batch_examples
            total_combined_loss += float(total_loss.detach().cpu()) * batch_examples
            total_contrastive_valid_anchor_ratio += float(contrastive_stats["valid_anchor_ratio"]) * batch_examples
            total_contrastive_positive_count += float(contrastive_stats["mean_positive_count"]) * batch_examples
            total_contrastive_target_bank_size += float(contrastive_stats["target_bank_size"]) * batch_examples

    if total_examples == 0:
        raise ValueError("No cross-domain batches were processed")

    pseudo_acceptance_ratio = 0.0 if total_pseudo_candidates == 0 else float(total_pseudo_accepted) / float(total_pseudo_candidates)
    mean_pseudo_distance = 0.0 if total_pseudo_accepted == 0 else total_pseudo_distance / float(total_pseudo_accepted)
    return {
        "source_loss": total_source_loss / total_examples,
        "target_loss": total_target_loss / total_examples,
        "mmd_loss": total_mmd_loss / total_examples,
        "source_stage_loss": total_source_stage_loss / total_examples,
        "pseudo_loss": total_pseudo_loss / total_examples,
        "contrastive_loss": total_contrastive_loss / total_examples,
        "monotonic_loss": total_monotonic_loss / total_examples,
        "total_loss": total_combined_loss / total_examples,
        "pseudo_acceptance_ratio": pseudo_acceptance_ratio,
        "pseudo_mean_distance": mean_pseudo_distance,
        "pseudo_accepted": float(total_pseudo_accepted),
        "pseudo_candidates": float(total_pseudo_candidates),
        "contrastive_valid_anchor_ratio": total_contrastive_valid_anchor_ratio / total_examples,
        "contrastive_mean_positive_count": total_contrastive_positive_count / total_examples,
        "contrastive_target_bank_size": total_contrastive_target_bank_size / total_examples,
    }


def fit_cd_pseudo_stage(
    args: argparse.Namespace,
    seed: int,
    model: nn.Module,
    source_train_loader: DataLoader,
    target_loaders: dict[str, DataLoader],
    target_monotonic_loader: DataLoader | None,
    device: torch.device,
    run_dir: Path,
    target_scale: float,
    grad_clip_norm: float | None,
) -> dict[str, object]:
    stage_head = nn.Linear(model.head.in_features, args.num_pseudo_stages).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(stage_head.parameters()),
        lr=args.target_lr,
        weight_decay=args.target_weight_decay,
    )
    best_val_rmse = float("inf")
    best_epoch = -1
    history: list[dict[str, object]] = []
    ckpt_path = run_dir / "cd_stage" / "best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.target_epochs + 1):
        stage_quantile = resolve_pseudo_quantile(
            epoch,
            args.target_epochs,
            start_quantile=args.pseudo_start_quantile,
            end_quantile=args.pseudo_end_quantile,
        )
        stage_statistics = collect_source_stage_statistics(
            model,
            source_train_loader,
            device,
            rul_clip=float(args.rul_clip),
            num_stages=args.num_pseudo_stages,
            quantile=stage_quantile,
            normalize_features=not args.disable_pseudo_feature_normalization,
        )
        train_stats = run_cd_pseudo_epoch(
            model,
            stage_head,
            source_train_loader,
            target_loaders["labeled"],
            target_loaders["unlabeled"],
            target_monotonic_loader,
            stage_statistics,
            optimizer,
            device,
            source_loss_weight=args.source_loss_weight,
            target_loss_weight=args.target_loss_weight,
            lambda_mmd=args.lambda_mmd,
            lambda_source_stage=args.lambda_source_stage,
            lambda_pseudo=args.lambda_pseudo,
            lambda_contrastive=args.lambda_contrastive,
            lambda_monotonic=args.lambda_monotonic,
            rul_clip=float(args.rul_clip),
            num_pseudo_stages=args.num_pseudo_stages,
            target_scale=target_scale,
            grad_clip_norm=grad_clip_norm,
            max_batches=args.max_target_train_batches,
            mmd_sigmas=tuple(float(token) for token in args.mmd_sigmas.split(",")),
            contrastive_temperature=float(args.contrastive_temperature),
            contrastive_normalize_features=not args.disable_contrastive_feature_normalization,
            monotonic_margin=float(args.monotonic_margin),
        )
        val_metrics = evaluate(model, target_loaders["val"], device, target_scale)
        record = {
            "stage": "cd_mmd_pseudo",
            "seed": seed,
            "epoch": epoch,
            "train_total_loss": train_stats["total_loss"],
            "train_source_loss": train_stats["source_loss"],
            "train_target_loss": train_stats["target_loss"],
            "train_mmd_loss": train_stats["mmd_loss"],
            "train_source_stage_loss": train_stats["source_stage_loss"],
            "train_pseudo_loss": train_stats["pseudo_loss"],
            "train_contrastive_loss": train_stats["contrastive_loss"],
            "train_monotonic_loss": train_stats["monotonic_loss"],
            "pseudo_acceptance_ratio": train_stats["pseudo_acceptance_ratio"],
            "pseudo_mean_distance": train_stats["pseudo_mean_distance"],
            "pseudo_stage_quantile": stage_quantile,
            "contrastive_valid_anchor_ratio": train_stats["contrastive_valid_anchor_ratio"],
            "contrastive_mean_positive_count": train_stats["contrastive_mean_positive_count"],
            "contrastive_target_bank_size": train_stats["contrastive_target_bank_size"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_score": val_metrics["score"],
            "lambda_mmd": args.lambda_mmd,
            "lambda_source_stage": args.lambda_source_stage,
            "lambda_pseudo": args.lambda_pseudo,
            "lambda_contrastive": args.lambda_contrastive,
            "lambda_monotonic": args.lambda_monotonic,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = float(val_metrics["rmse"])
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "stage_head_state_dict": stage_head.state_dict(),
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_val_rmse": best_val_rmse,
                },
                ckpt_path,
            )

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    stage_head.load_state_dict(checkpoint["stage_head_state_dict"])
    best_record = history[best_epoch - 1] if best_epoch > 0 else None
    return {
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "checkpoint": str(ckpt_path),
        "history": history,
        "best_record": best_record,
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
    target_monotonic_loader = None
    monotonic_meta = {
        "pair_count": 0,
        "pair_unit_count": 0,
        "pair_gap": int(args.monotonic_pair_gap),
        "pair_stride": int(args.monotonic_pair_stride),
    }
    if args.lambda_monotonic > 0:
        target_monotonic_loader, monotonic_meta = build_target_monotonic_loader(args, target_train_full, target_partition)

    cd_model = build_model(args, int(source_meta["train_windows_shape"][2])).to(device)
    source_checkpoint = torch.load(source_stage["checkpoint"], map_location=device)
    cd_model.load_state_dict(source_checkpoint["model_state_dict"])
    cd_stage = fit_cd_pseudo_stage(
        args,
        seed,
        cd_model,
        source_loaders["train"],
        target_loaders,
        target_monotonic_loader,
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
            "lambda_source_stage": float(args.lambda_source_stage),
            "lambda_pseudo": float(args.lambda_pseudo),
            "lambda_contrastive": float(args.lambda_contrastive),
            "lambda_monotonic": float(args.lambda_monotonic),
            "pseudo_num_stages": int(args.num_pseudo_stages),
            "pseudo_start_quantile": float(args.pseudo_start_quantile),
            "pseudo_end_quantile": float(args.pseudo_end_quantile),
            "labeled_windows_shape": target_meta["labeled_windows_shape"],
            "unlabeled_windows_shape": target_meta["unlabeled_windows_shape"],
            "val_windows_shape": target_meta["val_windows_shape"],
            "test_windows_shape": target_meta["test_windows_shape"],
            "labeled_unit_count": int(target_meta["labeled_unit_count"]),
            "val_unit_count": int(target_meta["val_unit_count"]),
            "unlabeled_unit_count": int(target_meta["unlabeled_unit_count"]),
            "monotonic_pair_count": int(monotonic_meta["pair_count"]),
            "monotonic_pair_unit_count": int(monotonic_meta["pair_unit_count"]),
            "monotonic_pair_gap": int(monotonic_meta["pair_gap"]),
            "monotonic_pair_stride": int(monotonic_meta["pair_stride"]),
            "test_rmse": float(cd_target_test_metrics["rmse"]),
            "test_mae": float(cd_target_test_metrics["mae"]),
            "test_score": float(cd_target_test_metrics["score"]),
            "best_pseudo_acceptance_ratio": 0.0 if cd_stage["best_record"] is None else float(cd_stage["best_record"]["pseudo_acceptance_ratio"]),
            "best_contrastive_valid_anchor_ratio": 0.0 if cd_stage["best_record"] is None else float(cd_stage["best_record"]["contrastive_valid_anchor_ratio"]),
        },
    }
    with (run_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def summarize_results(run_results: list[dict[str, object]], args: argparse.Namespace, start_time: float) -> dict[str, object]:
    direct_rmses = [float(item["target_direct"]["rmse"]) for item in run_results]
    cd_rmses = [float(item["cd_stage"]["test_rmse"]) for item in run_results]
    cd_scores = [float(item["cd_stage"]["test_score"]) for item in run_results]
    pseudo_acceptances = [float(item["cd_stage"]["best_pseudo_acceptance_ratio"]) for item in run_results]
    contrastive_valid_anchor_ratios = [float(item["cd_stage"]["best_contrastive_valid_anchor_ratio"]) for item in run_results]
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
        "lambda_source_stage": float(args.lambda_source_stage),
        "lambda_pseudo": float(args.lambda_pseudo),
        "lambda_contrastive": float(args.lambda_contrastive),
        "lambda_monotonic": float(args.lambda_monotonic),
        "pseudo_num_stages": int(args.num_pseudo_stages),
        "pseudo_start_quantile": float(args.pseudo_start_quantile),
        "pseudo_end_quantile": float(args.pseudo_end_quantile),
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
        "mean_best_epoch_pseudo_acceptance_ratio": float(np.mean(pseudo_acceptances)),
        "mean_best_epoch_contrastive_valid_anchor_ratio": float(np.mean(contrastive_valid_anchor_ratios)),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CD-MambAtt: source supervision + target few-shot supervision + MMD alignment + confidence-filtered pseudo-labeling + optional cross-domain contrastive alignment + optional local monotonicity loss."
    )
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
    parser.add_argument("--lambda-source-stage", type=float, default=1.0, help="Weight for source stage classification loss")
    parser.add_argument("--lambda-pseudo", type=float, default=1.0, help="Weight for target pseudo-label classification loss")
    parser.add_argument("--lambda-contrastive", type=float, default=0.0, help="Weight for cross-domain N-tuplet / contrastive alignment loss")
    parser.add_argument("--lambda-monotonic", type=float, default=0.0, help="Weight for local target monotonicity loss")
    parser.add_argument("--contrastive-temperature", type=float, default=0.1, help="Temperature used in cross-domain contrastive loss")
    parser.add_argument("--disable-contrastive-feature-normalization", action="store_true", help="Disable L2 normalization before cross-domain contrastive similarity")
    parser.add_argument("--monotonic-margin", type=float, default=0.0, help="Margin used in the local monotonicity ranking loss")
    parser.add_argument("--monotonic-pair-gap", type=int, default=1, help="Window gap used to form local target monotonic pairs")
    parser.add_argument("--monotonic-pair-stride", type=int, default=1, help="Stride used when sampling target monotonic pairs")
    parser.add_argument("--num-pseudo-stages", type=int, default=3, help="Number of degradation stages used for pseudo-labeling")
    parser.add_argument("--pseudo-start-quantile", type=float, default=0.30, help="Initial source-distance quantile used as the pseudo-label acceptance threshold")
    parser.add_argument("--pseudo-end-quantile", type=float, default=0.70, help="Final source-distance quantile used as the pseudo-label acceptance threshold")
    parser.add_argument("--disable-pseudo-feature-normalization", action="store_true", help="Disable L2 normalization before centroid-based pseudo-label assignment")
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
    parser.add_argument("--output-dir", default="/home/shelterpl/cd_mambatt/runs/cd_mambatt_v2", help="Run output directory")
    args = parser.parse_args()

    if args.target_shots <= 0:
        raise ValueError("CD-MambAtt v2 expects target_shots > 0")

    from cd_mambatt.cross_domain import resolve_cross_domain_task

    task = resolve_cross_domain_task(args.task, source_subset=args.source_subset, target_subset=args.target_subset)
    device = infer_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("CD-MambAtt v2 requires CUDA in the configured environment.")

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
