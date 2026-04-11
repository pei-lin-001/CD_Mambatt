from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from cd_mambatt.data import (
    CMAPSSSplit,
    CMAPSSWindowPairDataset,
    build_monotonic_window_pairs,
    fit_normalizer,
    load_cmapss_split,
    resolve_sensor_subset_preset,
    select_sensor_subset,
    select_units,
)
from cd_mambatt.losses import (
    DomainDiscriminator,
    conditional_gaussian_mmd_loss,
    compute_domain_adversarial_loss,
    cross_domain_contrastive_loss,
    gaussian_mmd_loss,
    local_monotonicity_loss,
)
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


def uses_domain_conditioning(model: nn.Module) -> bool:
    if getattr(model, "transformer_domain_adapter_mode", "none") != "none":
        return True
    mamba_blocks = getattr(model, "mamba_blocks", None)
    return mamba_blocks is not None and any(
        getattr(block, "domain_conditioned_gate", False) or getattr(block, "frontend_adapter_mode", "none") != "none"
        for block in mamba_blocks
    )


def model_forward_with_aux(model: nn.Module, windows: torch.Tensor, *, domain_label: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    if hasattr(model, "forward_features_with_aux"):
        outputs = model.forward_features_with_aux(windows, domain_label=domain_label)
        if isinstance(outputs, dict):
            if hasattr(model, "predict_from_output_dict"):
                prediction_dict = model.predict_from_output_dict(outputs)
                outputs = {**outputs, **prediction_dict}
            return outputs
    features = model.forward_features(windows, domain_label=domain_label)
    return {
        "features": features,
        "domain_features": features,
        "pre_transformer_features": features,
        "frontend_features": features,
        "invariant_features": features,
        "specific_features": torch.zeros_like(features),
        "prediction": model.predict_from_features(features),
        "prediction_shared": model.predict_from_features(features),
        "prediction_inv": model.predict_from_features(features),
        "prediction_spec": features.new_zeros((features.shape[0],)),
        "prediction_decomposed": model.predict_from_features(features),
        "gate_mean": features.new_zeros(()),
        "gate_dt_mean": features.new_zeros(()),
        "gate_bc_mean": features.new_zeros(()),
    }


def select_stage_features(outputs: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    if mode == "combined":
        return outputs["features"]
    if mode == "invariant":
        return outputs.get("invariant_features", outputs["features"])
    raise ValueError(f"Unsupported stage feature mode: {mode}")


def inv_spec_orthogonality_loss(invariant_features: torch.Tensor, specific_features: torch.Tensor) -> torch.Tensor:
    if invariant_features.ndim != 2 or specific_features.ndim != 2:
        raise ValueError("Orthogonality loss expects 2D feature tensors")
    invariant_norm = F.normalize(invariant_features, p=2, dim=1)
    specific_norm = F.normalize(specific_features, p=2, dim=1)
    cosine = (invariant_norm * specific_norm).sum(dim=1)
    return (cosine * cosine).mean()


def inv_spec_cross_correlation_loss(invariant_features: torch.Tensor, specific_features: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if invariant_features.ndim != 2 or specific_features.ndim != 2:
        raise ValueError("Cross-correlation loss expects 2D feature tensors")
    if invariant_features.shape != specific_features.shape:
        raise ValueError("Cross-correlation loss expects matching invariant/specific feature shapes")
    if invariant_features.shape[0] <= 1:
        return invariant_features.new_zeros(())

    invariant_centered = invariant_features - invariant_features.mean(dim=0, keepdim=True)
    specific_centered = specific_features - specific_features.mean(dim=0, keepdim=True)
    invariant_std = invariant_centered.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    specific_std = specific_centered.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    invariant_norm = invariant_centered / invariant_std
    specific_norm = specific_centered / specific_std
    cross_corr = invariant_norm.transpose(0, 1) @ specific_norm
    cross_corr = cross_corr / float(invariant_features.shape[0])
    return cross_corr.pow(2).mean()


def residual_reconstruction_loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    prediction_shared = outputs["prediction_shared"].detach()
    prediction_inv = outputs["prediction_inv"].detach()
    prediction_spec = outputs["prediction_spec"]
    residual_target = prediction_shared - prediction_inv
    return F.mse_loss(prediction_spec, residual_target)


def stage_prototype_alignment_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    stage_statistics: SourceStageStatistics,
    *,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("stage_prototype_alignment_loss expects 2D features")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("stage_prototype_alignment_loss expects 1D labels matching features")
    if features.shape[0] == 0:
        return features.new_zeros(())

    if stage_statistics.normalize_features:
        features = F.normalize(features, p=2, dim=1)
    centroids = stage_statistics.centroids[labels]
    squared_distances = (features - centroids).pow(2).sum(dim=1)

    if sample_weights is None:
        return squared_distances.mean()

    weights = sample_weights.to(device=features.device, dtype=features.dtype).clamp(min=0.0)
    weight_sum = weights.sum()
    if float(weight_sum.detach().cpu()) <= 0.0:
        return features.new_zeros(())
    return (weights * squared_distances).sum() / weight_sum


def stage_prototype_ce_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    stage_statistics: SourceStageStatistics,
    *,
    sample_weights: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("stage_prototype_ce_loss expects 2D features")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("stage_prototype_ce_loss expects 1D labels matching features")
    if features.shape[0] == 0:
        return features.new_zeros(())
    if temperature <= 0.0:
        raise ValueError("stage_prototype_ce_loss temperature must be positive")

    if stage_statistics.normalize_features:
        features = F.normalize(features, p=2, dim=1)
    distances = torch.cdist(features, stage_statistics.centroids).pow(2)
    logits = -distances / float(temperature)
    losses = F.cross_entropy(logits, labels, reduction="none")

    if sample_weights is None:
        return losses.mean()

    weights = sample_weights.to(device=features.device, dtype=features.dtype).clamp(min=0.0)
    weight_sum = weights.sum()
    if float(weight_sum.detach().cpu()) <= 0.0:
        return features.new_zeros(())
    return (weights * losses).sum() / weight_sum


def resolve_grl_lambda(epoch: int, total_epochs: int, *, base_lambda: float, warmup_epochs: int) -> float:
    if base_lambda <= 0:
        return 0.0
    if warmup_epochs <= 0:
        return float(base_lambda)
    progress = min(max(epoch / float(warmup_epochs), 0.0), 1.0)
    return float(base_lambda) * progress


def resolve_inv_mmd_lambda(args: argparse.Namespace) -> float:
    if str(args.inv_alignment_mode) != "mmd":
        return 0.0
    if float(args.lambda_inv_mmd) > 0:
        return float(args.lambda_inv_mmd)
    return float(args.lambda_domain_adv)


def configure_adaptation_model_params(
    model: nn.Module,
    freeze_mode: str,
) -> tuple[list[nn.Parameter], dict[str, object]]:
    named_params = list(model.named_parameters())

    if freeze_mode == "none":
        for _, param in named_params:
            param.requires_grad_(True)
        trainable = [param for _, param in named_params]
        names = [name for name, _ in named_params]
    else:
        allowed_names: list[str] = []
        for name, _ in named_params:
            if freeze_mode in {"head_only", "transformer_head", "spec_gate_head", "spec_gate_transformer_head"} and name.startswith("head."):
                allowed_names.append(name)
                continue
            if freeze_mode in {"transformer_head", "spec_gate_transformer_head"} and (
                name.startswith("transformer_blocks.") or name.startswith("transformer_encoder.")
            ):
                allowed_names.append(name)
                continue
            if freeze_mode in {"spec_gate_only", "spec_gate_head", "spec_gate_transformer_head"} and (
                ".x_proj_spec." in name or ".dt_proj_spec." in name or ".gate_proj." in name
            ):
                allowed_names.append(name)

        allowed_name_set = set(allowed_names)
        trainable = []
        names = []
        for name, param in named_params:
            is_trainable = name in allowed_name_set
            param.requires_grad_(is_trainable)
            if is_trainable:
                trainable.append(param)
                names.append(name)

    if not trainable:
        raise ValueError(f"No trainable model parameters were selected for adaptation freeze mode: {freeze_mode}")

    total_param_count = int(sum(param.numel() for _, param in named_params))
    trainable_param_count = int(sum(param.numel() for param in trainable))
    return trainable, {
        "freeze_mode": freeze_mode,
        "trainable_param_count": trainable_param_count,
        "frozen_param_count": int(total_param_count - trainable_param_count),
        "trainable_param_ratio": float(trainable_param_count) / float(total_param_count),
        "trainable_param_names": names,
    }


def configure_semantic_warmup_params(model: nn.Module) -> tuple[list[nn.Parameter], dict[str, object]]:
    named_params = list(model.named_parameters())
    trainable: list[nn.Parameter] = []
    trainable_names: list[str] = []
    for name, param in named_params:
        is_trainable = name.startswith("inv_head.") or name.startswith("spec_head.")
        param.requires_grad_(is_trainable)
        if is_trainable:
            trainable.append(param)
            trainable_names.append(name)
    if not trainable:
        raise ValueError("Semantic warmup requested, but the model does not expose inv/spec heads")
    total_param_count = int(sum(param.numel() for _, param in named_params))
    trainable_param_count = int(sum(param.numel() for param in trainable))
    return trainable, {
        "trainable_param_count": trainable_param_count,
        "frozen_param_count": int(total_param_count - trainable_param_count),
        "trainable_param_ratio": float(trainable_param_count) / float(total_param_count),
        "trainable_param_names": trainable_names,
    }


def build_adaptation_optimizer(
    trainable_model_params: list[nn.Parameter],
    stage_head: nn.Module,
    domain_discriminator: nn.Module | None,
    spec_domain_classifier: nn.Module | None,
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        trainable_model_params
        + list(stage_head.parameters())
        + ([] if domain_discriminator is None else list(domain_discriminator.parameters()))
        + ([] if spec_domain_classifier is None else list(spec_domain_classifier.parameters())),
        lr=lr,
        weight_decay=weight_decay,
    )


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str,
    total_epochs: int,
    min_lr: float,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    name = str(scheduler_name).lower()
    if name == "none":
        return None
    if name == "cosine":
        t_max = max(int(total_epochs), 1)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=float(min_lr),
        )
    raise ValueError(f"Unsupported lr scheduler: {scheduler_name}")


def run_semantic_warmup_epoch(
    model: nn.Module,
    source_loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    target_scale: float,
    lambda_inv_aux: float,
    lambda_spec_residual: float,
    lambda_spec_reconstruction: float,
    max_batches: int | None,
) -> dict[str, float]:
    if lambda_inv_aux <= 0 and lambda_spec_residual <= 0 and lambda_spec_reconstruction <= 0:
        return {
            "total_loss": 0.0,
            "inv_aux_loss": 0.0,
            "spec_residual_loss": 0.0,
            "spec_reconstruction_loss": 0.0,
        }

    is_train = optimizer is not None
    model.train(is_train)
    mse_loss = nn.MSELoss()
    use_dcg_warmup = uses_domain_conditioning(model)
    total_examples = 0
    total_loss_value = 0.0
    total_inv_aux_loss = 0.0
    total_spec_residual_loss = 0.0
    total_spec_reconstruction_loss = 0.0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, (windows, targets_raw) in enumerate(source_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            windows = windows.to(device, non_blocking=True)
            targets = targets_raw.to(device, non_blocking=True) / target_scale
            src_domain = torch.zeros(windows.shape[0], dtype=torch.long, device=device) if use_dcg_warmup else None
            outputs = model_forward_with_aux(model, windows, domain_label=src_domain)

            if lambda_inv_aux > 0:
                inv_aux_loss = mse_loss(outputs["prediction_inv"], targets)
            else:
                inv_aux_loss = windows.new_zeros(())

            if lambda_spec_residual > 0:
                spec_residual_loss = outputs["prediction_spec"].pow(2).mean()
            else:
                spec_residual_loss = windows.new_zeros(())

            if lambda_spec_reconstruction > 0:
                spec_reconstruction_loss = residual_reconstruction_loss(outputs)
            else:
                spec_reconstruction_loss = windows.new_zeros(())

            total_loss = (
                lambda_inv_aux * inv_aux_loss
                + lambda_spec_residual * spec_residual_loss
                + lambda_spec_reconstruction * spec_reconstruction_loss
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

            batch_size = int(windows.shape[0])
            total_examples += batch_size
            total_loss_value += float(total_loss.detach().cpu()) * batch_size
            total_inv_aux_loss += float(inv_aux_loss.detach().cpu()) * batch_size
            total_spec_residual_loss += float(spec_residual_loss.detach().cpu()) * batch_size
            total_spec_reconstruction_loss += float(spec_reconstruction_loss.detach().cpu()) * batch_size

    if total_examples == 0:
        raise ValueError("Semantic warmup processed zero source batches")
    return {
        "total_loss": total_loss_value / total_examples,
        "inv_aux_loss": total_inv_aux_loss / total_examples,
        "spec_residual_loss": total_spec_residual_loss / total_examples,
        "spec_reconstruction_loss": total_spec_reconstruction_loss / total_examples,
    }


def get_domain_feature_dim(model: nn.Module, tap: str) -> int:
    base_dim = int(model.head.in_features)
    if tap in {"inv_mean", "pre_transformer_last"}:
        return base_dim
    if tap == "frontend_mean":
        first_block = getattr(model, "mamba_blocks", [None])[0]
        frontend_dim = getattr(first_block, "d_inner", None)
        if frontend_dim is None:
            return base_dim
        return int(frontend_dim)
    if tap == "concat_inv_pre":
        return base_dim * 2
    raise ValueError(f"Unsupported domain feature tap: {tap}")


def select_domain_features(outputs: dict[str, torch.Tensor], tap: str) -> torch.Tensor:
    if tap == "inv_mean":
        return outputs["domain_features"]
    if tap == "pre_transformer_last":
        return outputs["pre_transformer_features"]
    if tap == "frontend_mean":
        return outputs["frontend_features"]
    if tap == "concat_inv_pre":
        return torch.cat([outputs["domain_features"], outputs["pre_transformer_features"]], dim=-1)
    raise ValueError(f"Unsupported domain feature tap: {tap}")


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
    stage_feature_mode: str,
) -> SourceStageStatistics:
    model.eval()
    use_dcg_stats = uses_domain_conditioning(model)
    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    rul_batches: list[torch.Tensor] = []

    with torch.no_grad():
        for windows, targets in source_loader:
            windows = windows.to(device, non_blocking=True)
            src_domain = torch.zeros(windows.shape[0], dtype=torch.long, device=device) if use_dcg_stats else None
            outputs = model_forward_with_aux(model, windows, domain_label=src_domain)
            features = select_stage_features(outputs, stage_feature_mode).detach().cpu()
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
    sensor_indices = resolve_sensor_subset_preset(getattr(args, "sensor_subset", None))
    target_train_full = select_sensor_subset(target_train_full, sensor_indices)
    train_units = np.sort(
        np.concatenate(
            [
                np.asarray(partition["labeled_units"], dtype=np.int32),
                np.asarray(partition["unlabeled_units"], dtype=np.int32),
            ]
        )
    )
    train_raw = select_units(target_train_full, train_units)
    normalizer = fit_normalizer(target_train_full, mode=str(getattr(args, "normalization_mode", "zscore")))
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
    domain_discriminator: DomainDiscriminator | None,
    spec_domain_classifier: nn.Module | None,
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
    lambda_domain_adv: float,
    lambda_inv_mmd: float,
    lambda_spec_domain: float,
    lambda_conditional_inv_mmd: float,
    lambda_conditional_proto: float,
    lambda_conditional_proto_ce: float,
    lambda_inv_spec_orth: float,
    lambda_inv_spec_xcorr: float,
    lambda_spec_residual: float,
    lambda_inv_aux: float,
    lambda_spec_reconstruction: float,
    lambda_transformer_domain_adapter_l2: float,
    rul_clip: float,
    num_pseudo_stages: int,
    target_scale: float,
    grad_clip_norm: float | None,
    max_batches: int | None,
    mmd_sigmas: tuple[float, ...],
    contrastive_temperature: float,
    contrastive_normalize_features: bool,
    monotonic_margin: float,
    grl_lambda: float,
    inv_alignment_mode: str,
    domain_feature_tap: str,
    stage_feature_mode: str,
    conditional_target_scope: str,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    stage_head.train(is_train)
    if domain_discriminator is not None:
        domain_discriminator.train(is_train)
    if spec_domain_classifier is not None:
        spec_domain_classifier.train(is_train)
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
    total_domain_adv_loss = 0.0
    total_inv_mmd_loss = 0.0
    total_spec_domain_loss = 0.0
    total_conditional_inv_mmd_loss = 0.0
    total_conditional_proto_loss = 0.0
    total_conditional_proto_ce_loss = 0.0
    total_inv_spec_orth_loss = 0.0
    total_inv_spec_xcorr_loss = 0.0
    total_spec_residual_loss = 0.0
    total_inv_aux_loss = 0.0
    total_spec_reconstruction_loss = 0.0
    total_transformer_domain_adapter_penalty = 0.0
    total_combined_loss = 0.0
    total_pseudo_candidates = 0
    total_pseudo_accepted = 0
    total_pseudo_distance = 0.0
    total_contrastive_valid_anchor_ratio = 0.0
    total_contrastive_positive_count = 0.0
    total_contrastive_target_bank_size = 0.0
    total_domain_accuracy = 0.0
    total_source_domain_accuracy = 0.0
    total_target_domain_accuracy = 0.0
    total_gate_mean = 0.0
    total_gate_dt_mean = 0.0
    total_gate_bc_mean = 0.0

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

            use_dcg = uses_domain_conditioning(model)
            source_domain = torch.zeros(source_windows.shape[0], dtype=torch.long, device=device) if use_dcg else None
            target_labeled_domain = torch.ones(target_labeled_windows.shape[0], dtype=torch.long, device=device) if use_dcg else None
            target_unlabeled_domain = torch.ones(target_unlabeled_windows.shape[0], dtype=torch.long, device=device) if use_dcg else None
            monotonic_domain = torch.ones(monotonic_earlier_windows.shape[0], dtype=torch.long, device=device) if use_dcg and monotonic_earlier_windows is not None else None

            source_outputs = model_forward_with_aux(model, source_windows, domain_label=source_domain)
            source_features = source_outputs["features"]
            source_stage_features = select_stage_features(source_outputs, stage_feature_mode)
            source_predictions = source_outputs["prediction"]
            source_loss = mse_loss(source_predictions, source_targets)

            source_stage_labels = assign_rul_stage_labels(
                source_targets_raw,
                rul_clip=rul_clip,
                num_stages=num_pseudo_stages,
            )
            source_stage_logits = stage_head(source_stage_features)
            source_stage_loss = ce_loss(source_stage_logits, source_stage_labels)

            target_labeled_outputs = model_forward_with_aux(model, target_labeled_windows, domain_label=target_labeled_domain)
            target_labeled_features = target_labeled_outputs["features"]
            target_labeled_stage_features = select_stage_features(target_labeled_outputs, stage_feature_mode)
            target_predictions = target_labeled_outputs["prediction"]
            target_loss = mse_loss(target_predictions, target_labeled_targets)
            target_labeled_stage_labels = assign_rul_stage_labels(
                target_labeled_targets_raw,
                rul_clip=rul_clip,
                num_stages=num_pseudo_stages,
            )

            target_unlabeled_outputs = model_forward_with_aux(model, target_unlabeled_windows, domain_label=target_unlabeled_domain)
            target_unlabeled_features = target_unlabeled_outputs["features"]
            target_unlabeled_stage_features = select_stage_features(target_unlabeled_outputs, stage_feature_mode)
            target_global_alignment_features = torch.cat([target_labeled_features, target_unlabeled_features], dim=0)
            mmd_loss = gaussian_mmd_loss(source_features, target_global_alignment_features, sigmas=mmd_sigmas)

            pseudo_stage_labels, accepted_mask, pseudo_distances = assign_pseudo_stage_labels(
                target_unlabeled_stage_features,
                stage_statistics,
            )
            accepted_count = int(accepted_mask.sum().item())
            candidate_count = int(target_unlabeled_features.shape[0])
            total_pseudo_candidates += candidate_count
            total_pseudo_accepted += accepted_count
            if accepted_count > 0:
                accepted_logits = stage_head(target_unlabeled_stage_features[accepted_mask])
                pseudo_loss = ce_loss(accepted_logits, pseudo_stage_labels[accepted_mask])
                total_pseudo_distance += float(pseudo_distances[accepted_mask].sum().detach().cpu())
            else:
                pseudo_loss = source_features.new_zeros(())

            contrastive_target_features = [target_labeled_stage_features]
            contrastive_target_labels = [target_labeled_stage_labels]
            if accepted_count > 0:
                contrastive_target_features.append(target_unlabeled_stage_features[accepted_mask])
                contrastive_target_labels.append(pseudo_stage_labels[accepted_mask])
            contrastive_loss, contrastive_stats = cross_domain_contrastive_loss(
                source_stage_features,
                source_stage_labels,
                torch.cat(contrastive_target_features, dim=0),
                torch.cat(contrastive_target_labels, dim=0),
                temperature=contrastive_temperature,
                normalize_features=contrastive_normalize_features,
            )

            if monotonic_earlier_windows is not None and monotonic_later_windows is not None:
                earlier_predictions = model(monotonic_earlier_windows, domain_label=monotonic_domain)
                later_predictions = model(monotonic_later_windows, domain_label=monotonic_domain)
                monotonic_loss = local_monotonicity_loss(
                    earlier_predictions,
                    later_predictions,
                    margin=monotonic_margin,
                )
            else:
                monotonic_loss = source_features.new_zeros(())

            source_domain_features = select_domain_features(source_outputs, domain_feature_tap)
            target_domain_features = torch.cat(
                [
                    select_domain_features(target_labeled_outputs, domain_feature_tap),
                    select_domain_features(target_unlabeled_outputs, domain_feature_tap),
                ],
                dim=0,
            )

            if inv_alignment_mode == "grl" and domain_discriminator is not None and lambda_domain_adv > 0:
                domain_adv_loss, domain_adv_stats = compute_domain_adversarial_loss(
                    domain_discriminator,
                    source_domain_features,
                    target_domain_features,
                    grl_lambda=grl_lambda,
                )
                inv_mmd_loss = source_features.new_zeros(())
            elif inv_alignment_mode == "mmd" and lambda_inv_mmd > 0:
                inv_mmd_loss = gaussian_mmd_loss(source_domain_features, target_domain_features, sigmas=mmd_sigmas)
                domain_adv_loss = source_features.new_zeros(())
                domain_adv_stats = {
                    "domain_accuracy": 0.0,
                    "source_domain_accuracy": 0.0,
                    "target_domain_accuracy": 0.0,
                }
            else:
                domain_adv_loss = source_features.new_zeros(())
                inv_mmd_loss = source_features.new_zeros(())
                domain_adv_stats = {
                    "domain_accuracy": 0.0,
                    "source_domain_accuracy": 0.0,
                    "target_domain_accuracy": 0.0,
                }

            if lambda_spec_domain > 0 and spec_domain_classifier is not None:
                source_specific_features = source_outputs.get("specific_features")
                target_specific_features = target_labeled_outputs.get("specific_features")
                if source_specific_features is not None and target_specific_features is not None:
                    spec_domain_features = torch.cat([source_specific_features, target_specific_features], dim=0)
                    spec_domain_labels = torch.cat(
                        [
                            torch.zeros(source_specific_features.shape[0], dtype=torch.long, device=source_specific_features.device),
                            torch.ones(target_specific_features.shape[0], dtype=torch.long, device=target_specific_features.device),
                        ],
                        dim=0,
                    )
                    spec_domain_loss = F.cross_entropy(spec_domain_classifier(spec_domain_features), spec_domain_labels)
                else:
                    spec_domain_loss = source_features.new_zeros(())
            else:
                spec_domain_loss = source_features.new_zeros(())

            if lambda_conditional_inv_mmd > 0:
                source_invariant_features = source_stage_features
                target_conditional_features = []
                target_conditional_labels = []
                if conditional_target_scope in {"labeled_only", "labeled_accepted"}:
                    target_conditional_features.append(target_labeled_stage_features)
                    target_conditional_labels.append(target_labeled_stage_labels)
                if conditional_target_scope in {"accepted_only", "labeled_accepted"} and accepted_count > 0:
                    target_conditional_features.append(target_unlabeled_stage_features[accepted_mask])
                    target_conditional_labels.append(pseudo_stage_labels[accepted_mask])
                if target_conditional_features:
                    conditional_inv_mmd_loss = conditional_gaussian_mmd_loss(
                        source_invariant_features,
                        source_stage_labels,
                        torch.cat(target_conditional_features, dim=0),
                        torch.cat(target_conditional_labels, dim=0),
                        sigmas=mmd_sigmas,
                    )
                else:
                    conditional_inv_mmd_loss = source_features.new_zeros(())
            else:
                conditional_inv_mmd_loss = source_features.new_zeros(())

            if lambda_conditional_proto > 0:
                prototype_terms: list[torch.Tensor] = []
                if conditional_target_scope in {"labeled_only", "labeled_accepted"}:
                    prototype_terms.append(
                        stage_prototype_alignment_loss(
                            target_labeled_stage_features,
                            target_labeled_stage_labels,
                            stage_statistics,
                        )
                    )
                if conditional_target_scope in {"accepted_only", "labeled_accepted"} and accepted_count > 0:
                    target_unlabeled_selected = target_unlabeled_stage_features[accepted_mask]
                    accepted_labels = pseudo_stage_labels[accepted_mask]
                    accepted_thresholds = stage_statistics.thresholds[accepted_labels].clamp_min(1e-6)
                    accepted_confidence = (1.0 - pseudo_distances[accepted_mask] / accepted_thresholds).clamp(min=0.0, max=1.0)
                    prototype_terms.append(
                        stage_prototype_alignment_loss(
                            target_unlabeled_selected,
                            accepted_labels,
                            stage_statistics,
                            sample_weights=accepted_confidence,
                        )
                    )
                conditional_proto_loss = (
                    torch.stack(prototype_terms).mean()
                    if prototype_terms
                    else source_features.new_zeros(())
                )
            else:
                conditional_proto_loss = source_features.new_zeros(())

            if lambda_conditional_proto_ce > 0:
                prototype_ce_terms: list[torch.Tensor] = []
                if conditional_target_scope in {"labeled_only", "labeled_accepted"}:
                    prototype_ce_terms.append(
                        stage_prototype_ce_loss(
                            target_labeled_stage_features,
                            target_labeled_stage_labels,
                            stage_statistics,
                        )
                    )
                if conditional_target_scope in {"accepted_only", "labeled_accepted"} and accepted_count > 0:
                    accepted_labels = pseudo_stage_labels[accepted_mask]
                    accepted_thresholds = stage_statistics.thresholds[accepted_labels].clamp_min(1e-6)
                    accepted_confidence = (1.0 - pseudo_distances[accepted_mask] / accepted_thresholds).clamp(min=0.0, max=1.0)
                    prototype_ce_terms.append(
                        stage_prototype_ce_loss(
                            target_unlabeled_stage_features[accepted_mask],
                            accepted_labels,
                            stage_statistics,
                            sample_weights=accepted_confidence,
                        )
                    )
                conditional_proto_ce_loss = (
                    torch.stack(prototype_ce_terms).mean()
                    if prototype_ce_terms
                    else source_features.new_zeros(())
                )
            else:
                conditional_proto_ce_loss = source_features.new_zeros(())

            if lambda_inv_spec_orth > 0:
                inv_spec_orth_loss = (
                    inv_spec_orthogonality_loss(
                        source_outputs.get("invariant_features", source_features),
                        source_outputs.get("specific_features", torch.zeros_like(source_features)),
                    )
                    + inv_spec_orthogonality_loss(
                        target_labeled_outputs.get("invariant_features", target_labeled_features),
                        target_labeled_outputs.get("specific_features", torch.zeros_like(target_labeled_features)),
                    )
                    + inv_spec_orthogonality_loss(
                        target_unlabeled_outputs.get("invariant_features", target_unlabeled_features),
                        target_unlabeled_outputs.get("specific_features", torch.zeros_like(target_unlabeled_features)),
                    )
                ) / 3.0
            else:
                inv_spec_orth_loss = source_features.new_zeros(())

            if lambda_inv_spec_xcorr > 0:
                inv_spec_xcorr_loss = (
                    inv_spec_cross_correlation_loss(
                        source_outputs.get("invariant_features", source_features),
                        source_outputs.get("specific_features", torch.zeros_like(source_features)),
                    )
                    + inv_spec_cross_correlation_loss(
                        target_labeled_outputs.get("invariant_features", target_labeled_features),
                        target_labeled_outputs.get("specific_features", torch.zeros_like(target_labeled_features)),
                    )
                    + inv_spec_cross_correlation_loss(
                        target_unlabeled_outputs.get("invariant_features", target_unlabeled_features),
                        target_unlabeled_outputs.get("specific_features", torch.zeros_like(target_unlabeled_features)),
                    )
                ) / 3.0
            else:
                inv_spec_xcorr_loss = source_features.new_zeros(())

            if lambda_spec_residual > 0:
                spec_residual_loss = (
                    source_outputs["prediction_spec"].pow(2).mean()
                    + target_labeled_outputs["prediction_spec"].pow(2).mean()
                    + target_unlabeled_outputs["prediction_spec"].pow(2).mean()
                ) / 3.0
            else:
                spec_residual_loss = source_features.new_zeros(())

            if lambda_inv_aux > 0:
                inv_aux_loss = (
                    mse_loss(source_outputs["prediction_inv"], source_targets)
                    + mse_loss(target_labeled_outputs["prediction_inv"], target_labeled_targets)
                ) / 2.0
            else:
                inv_aux_loss = source_features.new_zeros(())

            if lambda_spec_reconstruction > 0:
                spec_reconstruction_loss = (
                    residual_reconstruction_loss(source_outputs)
                    + residual_reconstruction_loss(target_labeled_outputs)
                    + residual_reconstruction_loss(target_unlabeled_outputs)
                ) / 3.0
            else:
                spec_reconstruction_loss = source_features.new_zeros(())

            if lambda_transformer_domain_adapter_l2 > 0 and hasattr(model, "transformer_domain_adapter_penalty"):
                transformer_domain_adapter_penalty = model.transformer_domain_adapter_penalty()
                if transformer_domain_adapter_penalty is None:
                    transformer_domain_adapter_penalty = source_features.new_zeros(())
            else:
                transformer_domain_adapter_penalty = source_features.new_zeros(())

            total_loss = (
                source_loss_weight * source_loss
                + target_loss_weight * target_loss
                + lambda_mmd * mmd_loss
                + lambda_source_stage * source_stage_loss
                + lambda_pseudo * pseudo_loss
                + lambda_contrastive * contrastive_loss
                + lambda_monotonic * monotonic_loss
                + lambda_domain_adv * domain_adv_loss
                + lambda_inv_mmd * inv_mmd_loss
                + lambda_spec_domain * spec_domain_loss
                + lambda_conditional_inv_mmd * conditional_inv_mmd_loss
                + lambda_conditional_proto * conditional_proto_loss
                + lambda_conditional_proto_ce * conditional_proto_ce_loss
                + lambda_inv_spec_orth * inv_spec_orth_loss
                + lambda_inv_spec_xcorr * inv_spec_xcorr_loss
                + lambda_spec_residual * spec_residual_loss
                + lambda_inv_aux * inv_aux_loss
                + lambda_spec_reconstruction * spec_reconstruction_loss
                + lambda_transformer_domain_adapter_l2 * transformer_domain_adapter_penalty
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters())
                        + list(stage_head.parameters())
                        + ([] if domain_discriminator is None else list(domain_discriminator.parameters())),
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
            total_domain_adv_loss += float(domain_adv_loss.detach().cpu()) * batch_examples
            total_inv_mmd_loss += float(inv_mmd_loss.detach().cpu()) * batch_examples
            total_spec_domain_loss += float(spec_domain_loss.detach().cpu()) * batch_examples
            total_conditional_inv_mmd_loss += float(conditional_inv_mmd_loss.detach().cpu()) * batch_examples
            total_conditional_proto_loss += float(conditional_proto_loss.detach().cpu()) * batch_examples
            total_conditional_proto_ce_loss += float(conditional_proto_ce_loss.detach().cpu()) * batch_examples
            total_inv_spec_orth_loss += float(inv_spec_orth_loss.detach().cpu()) * batch_examples
            total_inv_spec_xcorr_loss += float(inv_spec_xcorr_loss.detach().cpu()) * batch_examples
            total_spec_residual_loss += float(spec_residual_loss.detach().cpu()) * batch_examples
            total_inv_aux_loss += float(inv_aux_loss.detach().cpu()) * batch_examples
            total_spec_reconstruction_loss += float(spec_reconstruction_loss.detach().cpu()) * batch_examples
            total_transformer_domain_adapter_penalty += float(transformer_domain_adapter_penalty.detach().cpu()) * batch_examples
            total_combined_loss += float(total_loss.detach().cpu()) * batch_examples
            total_contrastive_valid_anchor_ratio += float(contrastive_stats["valid_anchor_ratio"]) * batch_examples
            total_contrastive_positive_count += float(contrastive_stats["mean_positive_count"]) * batch_examples
            total_contrastive_target_bank_size += float(contrastive_stats["target_bank_size"]) * batch_examples
            total_domain_accuracy += float(domain_adv_stats["domain_accuracy"]) * batch_examples
            total_source_domain_accuracy += float(domain_adv_stats["source_domain_accuracy"]) * batch_examples
            total_target_domain_accuracy += float(domain_adv_stats["target_domain_accuracy"]) * batch_examples
            total_gate_mean += float(source_outputs.get("gate_mean", source_features.new_zeros(())).detach().cpu()) * batch_examples
            total_gate_dt_mean += float(source_outputs.get("gate_dt_mean", source_features.new_zeros(())).detach().cpu()) * batch_examples
            total_gate_bc_mean += float(source_outputs.get("gate_bc_mean", source_features.new_zeros(())).detach().cpu()) * batch_examples

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
        "domain_adv_loss": total_domain_adv_loss / total_examples,
        "inv_mmd_loss": total_inv_mmd_loss / total_examples,
        "spec_domain_loss": total_spec_domain_loss / total_examples,
        "conditional_inv_mmd_loss": total_conditional_inv_mmd_loss / total_examples,
        "conditional_proto_loss": total_conditional_proto_loss / total_examples,
        "conditional_proto_ce_loss": total_conditional_proto_ce_loss / total_examples,
        "inv_spec_orth_loss": total_inv_spec_orth_loss / total_examples,
        "inv_spec_xcorr_loss": total_inv_spec_xcorr_loss / total_examples,
        "spec_residual_loss": total_spec_residual_loss / total_examples,
        "inv_aux_loss": total_inv_aux_loss / total_examples,
        "spec_reconstruction_loss": total_spec_reconstruction_loss / total_examples,
        "transformer_domain_adapter_penalty": total_transformer_domain_adapter_penalty / total_examples,
        "total_loss": total_combined_loss / total_examples,
        "pseudo_acceptance_ratio": pseudo_acceptance_ratio,
        "pseudo_mean_distance": mean_pseudo_distance,
        "pseudo_accepted": float(total_pseudo_accepted),
        "pseudo_candidates": float(total_pseudo_candidates),
        "contrastive_valid_anchor_ratio": total_contrastive_valid_anchor_ratio / total_examples,
        "contrastive_mean_positive_count": total_contrastive_positive_count / total_examples,
        "contrastive_target_bank_size": total_contrastive_target_bank_size / total_examples,
        "domain_accuracy": total_domain_accuracy / total_examples,
        "source_domain_accuracy": total_source_domain_accuracy / total_examples,
        "target_domain_accuracy": total_target_domain_accuracy / total_examples,
        "gate_mean": total_gate_mean / total_examples,
        "gate_dt_mean": total_gate_dt_mean / total_examples,
        "gate_bc_mean": total_gate_bc_mean / total_examples,
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
    effective_lambda_inv_mmd = resolve_inv_mmd_lambda(args)
    domain_discriminator = None
    spec_domain_classifier = None
    if args.inv_alignment_mode == "grl" and args.lambda_domain_adv > 0:
        domain_discriminator = DomainDiscriminator(
            get_domain_feature_dim(model, args.domain_feature_tap),
            hidden_dim=args.domain_adv_hidden_dim,
            dropout=args.domain_adv_dropout,
        ).to(device)
    if float(getattr(args, "lambda_spec_domain", 0.0)) > 0:
        spec_domain_classifier = nn.Linear(model.head.in_features, 2).to(device)
    best_val_rmse = float("inf")
    best_epoch = -1
    history: list[dict[str, object]] = []
    ckpt_path = run_dir / "cd_stage" / "best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_epochs = max(int(getattr(args, "adaptation_freeze_epochs", 0)), 0)
    initial_freeze_mode = str(args.adaptation_freeze_mode)
    active_freeze_mode = initial_freeze_mode
    active_trainable_model_params, freeze_info = configure_adaptation_model_params(model, active_freeze_mode)
    optimizer = build_adaptation_optimizer(
        active_trainable_model_params,
        stage_head,
        domain_discriminator,
        spec_domain_classifier,
        lr=float(args.target_lr),
        weight_decay=float(args.target_weight_decay),
    )
    scheduler = build_lr_scheduler(
        optimizer,
        scheduler_name=str(getattr(args, "target_lr_scheduler", "none")),
        total_epochs=int(args.target_epochs),
        min_lr=float(getattr(args, "target_lr_min", 1e-5)),
    )
    trainable_param_path = run_dir / "cd_stage" / "trainable_model_params.txt"
    trainable_param_path.write_text("\n".join(str(name) for name in freeze_info["trainable_param_names"]) + "\n", encoding="utf-8")
    full_trainable_param_path = run_dir / "cd_stage" / "trainable_model_params_unfrozen.txt"
    switched_to_full_unfreeze = False

    for epoch in range(1, args.target_epochs + 1):
        if (
            not switched_to_full_unfreeze
            and freeze_epochs > 0
            and initial_freeze_mode != "none"
            and epoch == freeze_epochs + 1
        ):
            active_freeze_mode = "none"
            active_trainable_model_params, _ = configure_adaptation_model_params(model, active_freeze_mode)
            optimizer = build_adaptation_optimizer(
                active_trainable_model_params,
                stage_head,
                domain_discriminator,
                spec_domain_classifier,
                lr=float(args.target_lr),
                weight_decay=float(args.target_weight_decay),
            )
            scheduler = build_lr_scheduler(
                optimizer,
                scheduler_name=str(getattr(args, "target_lr_scheduler", "none")),
                total_epochs=max(int(args.target_epochs) - epoch + 1, 1),
                min_lr=float(getattr(args, "target_lr_min", 1e-5)),
            )
            full_trainable_param_path.write_text(
                "\n".join(name for name, param in model.named_parameters() if param.requires_grad) + "\n",
                encoding="utf-8",
            )
            switched_to_full_unfreeze = True
            print(
                json.dumps(
                    {
                        "stage": "cd_spd_v0_freeze_switch",
                        "seed": seed,
                        "epoch": epoch,
                        "from_freeze_mode": initial_freeze_mode,
                        "to_freeze_mode": active_freeze_mode,
                        "freeze_epochs": freeze_epochs,
                    },
                    ensure_ascii=False,
                )
            )
        stage_quantile = resolve_pseudo_quantile(
            epoch,
            args.target_epochs,
            start_quantile=args.pseudo_start_quantile,
            end_quantile=args.pseudo_end_quantile,
        )
        current_grl_lambda = resolve_grl_lambda(
            epoch,
            args.target_epochs,
            base_lambda=float(args.grl_lambda),
            warmup_epochs=int(args.grl_warmup_epochs),
        )
        stage_statistics = collect_source_stage_statistics(
            model,
            source_train_loader,
            device,
            rul_clip=float(args.rul_clip),
            num_stages=args.num_pseudo_stages,
            quantile=stage_quantile,
            normalize_features=not args.disable_pseudo_feature_normalization,
            stage_feature_mode=str(args.stage_feature_mode),
        )
        train_stats = run_cd_pseudo_epoch(
            model,
            stage_head,
            domain_discriminator,
            spec_domain_classifier,
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
            lambda_domain_adv=args.lambda_domain_adv,
            lambda_inv_mmd=effective_lambda_inv_mmd,
            lambda_spec_domain=float(getattr(args, "lambda_spec_domain", 0.0)),
            lambda_conditional_inv_mmd=float(args.lambda_conditional_inv_mmd),
            lambda_conditional_proto=float(args.lambda_conditional_proto),
            lambda_conditional_proto_ce=float(args.lambda_conditional_proto_ce),
            lambda_inv_spec_orth=float(args.lambda_inv_spec_orth),
            lambda_inv_spec_xcorr=float(getattr(args, "lambda_inv_spec_xcorr", 0.0)),
            lambda_spec_residual=float(args.lambda_spec_residual),
            lambda_inv_aux=float(args.lambda_inv_aux),
            lambda_spec_reconstruction=float(args.lambda_spec_reconstruction),
            lambda_transformer_domain_adapter_l2=float(getattr(args, "lambda_transformer_domain_adapter_l2", 0.0)),
            rul_clip=float(args.rul_clip),
            num_pseudo_stages=args.num_pseudo_stages,
            target_scale=target_scale,
            grad_clip_norm=grad_clip_norm,
            max_batches=args.max_target_train_batches,
            mmd_sigmas=tuple(float(token) for token in args.mmd_sigmas.split(",")),
            contrastive_temperature=float(args.contrastive_temperature),
            contrastive_normalize_features=not args.disable_contrastive_feature_normalization,
            monotonic_margin=float(args.monotonic_margin),
            grl_lambda=current_grl_lambda,
            inv_alignment_mode=str(args.inv_alignment_mode),
            domain_feature_tap=str(args.domain_feature_tap),
            stage_feature_mode=str(args.stage_feature_mode),
            conditional_target_scope=str(args.conditional_target_scope),
        )
        val_metrics = evaluate(
            model,
            target_loaders["val"],
            device,
            target_scale,
            domain_label_value=1 if uses_domain_conditioning(model) else None,
        )
        current_lr = float(optimizer.param_groups[0]["lr"])
        record = {
            "stage": "cd_spd_v0",
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
            "train_domain_adv_loss": train_stats["domain_adv_loss"],
            "train_inv_mmd_loss": train_stats["inv_mmd_loss"],
            "train_spec_domain_loss": train_stats["spec_domain_loss"],
            "train_conditional_inv_mmd_loss": train_stats["conditional_inv_mmd_loss"],
            "train_conditional_proto_loss": train_stats["conditional_proto_loss"],
            "train_conditional_proto_ce_loss": train_stats["conditional_proto_ce_loss"],
            "train_inv_spec_orth_loss": train_stats["inv_spec_orth_loss"],
            "train_inv_spec_xcorr_loss": train_stats["inv_spec_xcorr_loss"],
            "train_spec_residual_loss": train_stats["spec_residual_loss"],
            "train_inv_aux_loss": train_stats["inv_aux_loss"],
            "train_spec_reconstruction_loss": train_stats["spec_reconstruction_loss"],
            "train_transformer_domain_adapter_penalty": train_stats["transformer_domain_adapter_penalty"],
            "pseudo_acceptance_ratio": train_stats["pseudo_acceptance_ratio"],
            "pseudo_mean_distance": train_stats["pseudo_mean_distance"],
            "pseudo_stage_quantile": stage_quantile,
            "contrastive_valid_anchor_ratio": train_stats["contrastive_valid_anchor_ratio"],
            "contrastive_mean_positive_count": train_stats["contrastive_mean_positive_count"],
            "contrastive_target_bank_size": train_stats["contrastive_target_bank_size"],
            "domain_accuracy": train_stats["domain_accuracy"],
            "source_domain_accuracy": train_stats["source_domain_accuracy"],
            "target_domain_accuracy": train_stats["target_domain_accuracy"],
            "gate_mean": train_stats["gate_mean"],
            "gate_dt_mean": train_stats["gate_dt_mean"],
            "gate_bc_mean": train_stats["gate_bc_mean"],
            "grl_lambda": current_grl_lambda,
            "inv_alignment_mode": str(args.inv_alignment_mode),
            "active_adaptation_freeze_mode": active_freeze_mode,
            "target_lr": current_lr,
            "target_lr_scheduler": str(getattr(args, "target_lr_scheduler", "none")),
            "domain_feature_tap": str(args.domain_feature_tap),
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_score": val_metrics["score"],
            "lambda_mmd": args.lambda_mmd,
            "lambda_source_stage": args.lambda_source_stage,
            "lambda_pseudo": args.lambda_pseudo,
            "lambda_contrastive": args.lambda_contrastive,
            "lambda_monotonic": args.lambda_monotonic,
            "lambda_domain_adv": args.lambda_domain_adv,
            "lambda_inv_mmd": effective_lambda_inv_mmd,
            "lambda_spec_domain": float(getattr(args, "lambda_spec_domain", 0.0)),
            "lambda_conditional_inv_mmd": args.lambda_conditional_inv_mmd,
            "lambda_conditional_proto": float(args.lambda_conditional_proto),
            "lambda_conditional_proto_ce": float(args.lambda_conditional_proto_ce),
            "conditional_target_scope": str(args.conditional_target_scope),
            "lambda_inv_spec_orth": args.lambda_inv_spec_orth,
            "lambda_inv_spec_xcorr": float(getattr(args, "lambda_inv_spec_xcorr", 0.0)),
            "lambda_spec_residual": args.lambda_spec_residual,
            "lambda_inv_aux": args.lambda_inv_aux,
            "lambda_spec_reconstruction": args.lambda_spec_reconstruction,
            "lambda_transformer_domain_adapter_l2": float(getattr(args, "lambda_transformer_domain_adapter_l2", 0.0)),
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
                    "domain_discriminator_state_dict": None if domain_discriminator is None else domain_discriminator.state_dict(),
                    "spec_domain_classifier_state_dict": None if spec_domain_classifier is None else spec_domain_classifier.state_dict(),
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_val_rmse": best_val_rmse,
                },
                ckpt_path,
            )
        if scheduler is not None:
            scheduler.step()

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    stage_head.load_state_dict(checkpoint["stage_head_state_dict"])
    if domain_discriminator is not None and checkpoint["domain_discriminator_state_dict"] is not None:
        domain_discriminator.load_state_dict(checkpoint["domain_discriminator_state_dict"])
    if spec_domain_classifier is not None and checkpoint.get("spec_domain_classifier_state_dict") is not None:
        spec_domain_classifier.load_state_dict(checkpoint["spec_domain_classifier_state_dict"])
    best_record = history[best_epoch - 1] if best_epoch > 0 else None
    return {
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "checkpoint": str(ckpt_path),
        "history": history,
        "best_record": best_record,
        "uses_domain_adv": bool(domain_discriminator is not None),
        "uses_spec_domain": bool(spec_domain_classifier is not None),
        "freeze_info": {
            **freeze_info,
            "trainable_param_path": str(trainable_param_path),
            "full_trainable_param_path": str(full_trainable_param_path) if switched_to_full_unfreeze else None,
            "freeze_epochs": int(freeze_epochs),
            "switched_to_full_unfreeze": bool(switched_to_full_unfreeze),
        },
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
    source_model_args = args
    if str(args.mamba_block_mode) == "dd_spd" and str(args.spd_predictor_mode) in {"decomposed_residual", "shared_aux_residual"}:
        source_model_args = argparse.Namespace(**vars(args))
        source_model_args.spd_predictor_mode = "shared_head"

    source_split = load_or_create_source_split(args, source_train_full, task_output_dir, seed)
    source_loaders, source_meta = build_source_stage_data(args, source_train_full, source_test_raw, source_split)

    base_model = build_model(source_model_args, int(source_meta["train_windows_shape"][2])).to(device)
    source_stage = fit_source_stage(args, seed, base_model, source_loaders, device, run_dir, target_scale, grad_clip_norm)
    source_test_metrics = evaluate(
        base_model,
        source_loaders["test"],
        device,
        target_scale,
        domain_label_value=0 if uses_domain_conditioning(base_model) else None,
    )
    target_direct_loader, target_direct_meta = build_target_direct_test_loader(args, target_train_full, target_test_raw)
    target_direct_metrics = evaluate(
        base_model,
        target_direct_loader,
        device,
        target_scale,
        domain_label_value=1 if uses_domain_conditioning(base_model) else None,
    )

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

    reuse_source_model_for_adaptation = bool(getattr(args, "reuse_source_model_for_adaptation", False))
    source_checkpoint = torch.load(source_stage["checkpoint"], map_location=device)
    if reuse_source_model_for_adaptation:
        cd_model = base_model
    else:
        cd_model = build_model(args, int(source_meta["train_windows_shape"][2])).to(device)
    semantic_warmup_history: list[dict[str, object]] = []
    semantic_warmup_info: dict[str, object] | None = None
    if not reuse_source_model_for_adaptation:
        if str(args.spd_predictor_mode) in {"decomposed_residual", "shared_aux_residual"}:
            cd_model.load_state_dict(source_checkpoint["model_state_dict"], strict=False)
            if getattr(cd_model, "inv_head", None) is not None and "head.weight" in source_checkpoint["model_state_dict"]:
                with torch.no_grad():
                    cd_model.inv_head.weight.copy_(source_checkpoint["model_state_dict"]["head.weight"])
                    cd_model.inv_head.bias.copy_(source_checkpoint["model_state_dict"]["head.bias"])
                    nn.init.zeros_(cd_model.spec_head.weight)
                    nn.init.zeros_(cd_model.spec_head.bias)
        else:
            cd_model.load_state_dict(source_checkpoint["model_state_dict"])

    if str(args.spd_predictor_mode) == "shared_aux_residual" and int(args.semantic_warmup_epochs) > 0:
        semantic_warmup_params, semantic_warmup_freeze_info = configure_semantic_warmup_params(cd_model)
        semantic_warmup_optimizer = torch.optim.Adam(
            semantic_warmup_params,
            lr=float(args.semantic_warmup_lr),
            weight_decay=float(args.target_weight_decay),
        )
        semantic_warmup_dir = run_dir / "semantic_warmup"
        semantic_warmup_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, int(args.semantic_warmup_epochs) + 1):
            warmup_stats = run_semantic_warmup_epoch(
                cd_model,
                source_loaders["train"],
                semantic_warmup_optimizer,
                device,
                target_scale=target_scale,
                lambda_inv_aux=float(args.lambda_inv_aux),
                lambda_spec_residual=float(args.lambda_spec_residual),
                lambda_spec_reconstruction=float(args.lambda_spec_reconstruction),
                max_batches=args.max_source_train_batches,
            )
            record = {
                "stage": "semantic_warmup",
                "seed": seed,
                "epoch": epoch,
                "train_total_loss": warmup_stats["total_loss"],
                "train_inv_aux_loss": warmup_stats["inv_aux_loss"],
                "train_spec_residual_loss": warmup_stats["spec_residual_loss"],
                "train_spec_reconstruction_loss": warmup_stats["spec_reconstruction_loss"],
            }
            semantic_warmup_history.append(record)
            print(json.dumps(record, ensure_ascii=False))
        semantic_warmup_info = {
            **semantic_warmup_freeze_info,
            "epochs": int(args.semantic_warmup_epochs),
            "lr": float(args.semantic_warmup_lr),
            "history_path": str(semantic_warmup_dir / "history.json"),
        }
        with (semantic_warmup_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(semantic_warmup_history, handle, indent=2, ensure_ascii=False)

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
    cd_target_test_metrics = evaluate(
        cd_model,
        target_loaders["test"],
        device,
        target_scale,
        domain_label_value=1 if uses_domain_conditioning(cd_model) else None,
    )

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
        "semantic_warmup": None
        if semantic_warmup_info is None
        else {
            "epochs": int(semantic_warmup_info["epochs"]),
            "lr": float(semantic_warmup_info["lr"]),
            "trainable_param_count": int(semantic_warmup_info["trainable_param_count"]),
            "frozen_param_count": int(semantic_warmup_info["frozen_param_count"]),
            "trainable_param_ratio": float(semantic_warmup_info["trainable_param_ratio"]),
            "history_path": str(semantic_warmup_info["history_path"]),
            "last_train_inv_aux_loss": float(semantic_warmup_history[-1]["train_inv_aux_loss"]) if semantic_warmup_history else 0.0,
            "last_train_spec_residual_loss": float(semantic_warmup_history[-1]["train_spec_residual_loss"]) if semantic_warmup_history else 0.0,
            "last_train_spec_reconstruction_loss": float(semantic_warmup_history[-1]["train_spec_reconstruction_loss"]) if semantic_warmup_history else 0.0,
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
            "lambda_domain_adv": float(args.lambda_domain_adv),
            "lambda_inv_mmd": float(resolve_inv_mmd_lambda(args)),
            "lambda_spec_domain": float(getattr(args, "lambda_spec_domain", 0.0)),
            "lambda_conditional_inv_mmd": float(args.lambda_conditional_inv_mmd),
            "lambda_conditional_proto": float(args.lambda_conditional_proto),
            "lambda_conditional_proto_ce": float(args.lambda_conditional_proto_ce),
            "lambda_inv_spec_orth": float(args.lambda_inv_spec_orth),
            "lambda_inv_spec_xcorr": float(getattr(args, "lambda_inv_spec_xcorr", 0.0)),
            "lambda_spec_residual": float(args.lambda_spec_residual),
            "lambda_inv_aux": float(args.lambda_inv_aux),
            "lambda_spec_reconstruction": float(args.lambda_spec_reconstruction),
            "inv_alignment_mode": str(args.inv_alignment_mode),
            "semantic_warmup_epochs": int(args.semantic_warmup_epochs),
            "semantic_warmup_lr": float(args.semantic_warmup_lr),
            "adaptation_freeze_mode": str(args.adaptation_freeze_mode),
            "adaptation_freeze_epochs": int(getattr(args, "adaptation_freeze_epochs", 0)),
            "reuse_source_model_for_adaptation": bool(getattr(args, "reuse_source_model_for_adaptation", False)),
            "target_lr": float(args.target_lr),
            "target_lr_scheduler": str(getattr(args, "target_lr_scheduler", "none")),
            "target_lr_min": float(getattr(args, "target_lr_min", 1e-5)),
            "stage_feature_mode": str(args.stage_feature_mode),
            "conditional_target_scope": str(args.conditional_target_scope),
            "spd_predictor_mode": str(args.spd_predictor_mode),
            "domain_conditioned_gate": bool(getattr(args, "domain_conditioned_gate", False)),
            "frontend_adapter_mode": str(getattr(args, "frontend_adapter_mode", "none")),
            "transformer_domain_adapter_mode": str(getattr(args, "transformer_domain_adapter_mode", "none")),
            "domain_feature_tap": str(args.domain_feature_tap),
            "grl_lambda": float(args.grl_lambda),
            "grl_warmup_epochs": int(args.grl_warmup_epochs),
            "domain_adv_hidden_dim": int(args.domain_adv_hidden_dim),
            "trainable_model_param_count": int(cd_stage["freeze_info"]["trainable_param_count"]),
            "frozen_model_param_count": int(cd_stage["freeze_info"]["frozen_param_count"]),
            "trainable_model_param_ratio": float(cd_stage["freeze_info"]["trainable_param_ratio"]),
            "trainable_model_param_path": str(cd_stage["freeze_info"]["trainable_param_path"]),
            "full_trainable_model_param_path": cd_stage["freeze_info"]["full_trainable_param_path"],
            "switched_to_full_unfreeze": bool(cd_stage["freeze_info"]["switched_to_full_unfreeze"]),
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
            "best_domain_accuracy": 0.0 if cd_stage["best_record"] is None else float(cd_stage["best_record"]["domain_accuracy"]),
            "best_gate_mean": 0.0 if cd_stage["best_record"] is None else float(cd_stage["best_record"].get("gate_mean", 0.0)),
            "best_gate_dt_mean": 0.0 if cd_stage["best_record"] is None else float(cd_stage["best_record"].get("gate_dt_mean", 0.0)),
            "best_gate_bc_mean": 0.0 if cd_stage["best_record"] is None else float(cd_stage["best_record"].get("gate_bc_mean", 0.0)),
            "best_inv_mmd_loss": 0.0 if cd_stage["best_record"] is None else float(cd_stage["best_record"].get("train_inv_mmd_loss", 0.0)),
            "best_spec_domain_loss": 0.0 if cd_stage["best_record"] is None else float(cd_stage["best_record"].get("train_spec_domain_loss", 0.0)),
            "best_conditional_inv_mmd_loss": 0.0 if cd_stage["best_record"] is None else float(cd_stage["best_record"].get("train_conditional_inv_mmd_loss", 0.0)),
            "best_conditional_proto_loss": 0.0 if cd_stage["best_record"] is None else float(cd_stage["best_record"].get("train_conditional_proto_loss", 0.0)),
            "best_conditional_proto_ce_loss": 0.0 if cd_stage["best_record"] is None else float(cd_stage["best_record"].get("train_conditional_proto_ce_loss", 0.0)),
            "best_inv_spec_orth_loss": 0.0 if cd_stage["best_record"] is None else float(cd_stage["best_record"].get("train_inv_spec_orth_loss", 0.0)),
            "best_inv_spec_xcorr_loss": 0.0 if cd_stage["best_record"] is None else float(cd_stage["best_record"].get("train_inv_spec_xcorr_loss", 0.0)),
            "best_spec_residual_loss": 0.0 if cd_stage["best_record"] is None else float(cd_stage["best_record"].get("train_spec_residual_loss", 0.0)),
            "best_inv_aux_loss": 0.0 if cd_stage["best_record"] is None else float(cd_stage["best_record"].get("train_inv_aux_loss", 0.0)),
            "best_spec_reconstruction_loss": 0.0 if cd_stage["best_record"] is None else float(cd_stage["best_record"].get("train_spec_reconstruction_loss", 0.0)),
            "best_transformer_domain_adapter_penalty": 0.0 if cd_stage["best_record"] is None else float(cd_stage["best_record"].get("train_transformer_domain_adapter_penalty", 0.0)),
            "best_active_adaptation_freeze_mode": None if cd_stage["best_record"] is None else str(cd_stage["best_record"].get("active_adaptation_freeze_mode", "")),
            "uses_domain_adv": bool(cd_stage["uses_domain_adv"]),
            "uses_spec_domain": bool(cd_stage.get("uses_spec_domain", False)),
            "uses_inv_mmd_alignment": bool(str(args.inv_alignment_mode) == "mmd" and resolve_inv_mmd_lambda(args) > 0),
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
    domain_accuracies = [float(item["cd_stage"]["best_domain_accuracy"]) for item in run_results]
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
        "lambda_domain_adv": float(args.lambda_domain_adv),
        "lambda_inv_mmd": float(resolve_inv_mmd_lambda(args)),
        "lambda_spec_domain": float(getattr(args, "lambda_spec_domain", 0.0)),
        "lambda_conditional_inv_mmd": float(args.lambda_conditional_inv_mmd),
        "lambda_inv_spec_orth": float(args.lambda_inv_spec_orth),
        "lambda_inv_spec_xcorr": float(getattr(args, "lambda_inv_spec_xcorr", 0.0)),
        "lambda_spec_residual": float(args.lambda_spec_residual),
        "lambda_inv_aux": float(args.lambda_inv_aux),
        "lambda_spec_reconstruction": float(args.lambda_spec_reconstruction),
        "lambda_transformer_domain_adapter_l2": float(getattr(args, "lambda_transformer_domain_adapter_l2", 0.0)),
        "inv_alignment_mode": str(args.inv_alignment_mode),
        "semantic_warmup_epochs": int(args.semantic_warmup_epochs),
        "semantic_warmup_lr": float(args.semantic_warmup_lr),
        "adaptation_freeze_mode": str(args.adaptation_freeze_mode),
        "adaptation_freeze_epochs": int(getattr(args, "adaptation_freeze_epochs", 0)),
        "reuse_source_model_for_adaptation": bool(getattr(args, "reuse_source_model_for_adaptation", False)),
        "target_lr": float(args.target_lr),
        "target_lr_scheduler": str(getattr(args, "target_lr_scheduler", "none")),
        "target_lr_min": float(getattr(args, "target_lr_min", 1e-5)),
        "stage_feature_mode": str(args.stage_feature_mode),
        "spd_predictor_mode": str(args.spd_predictor_mode),
        "domain_conditioned_gate": bool(getattr(args, "domain_conditioned_gate", False)),
        "frontend_adapter_mode": str(getattr(args, "frontend_adapter_mode", "none")),
        "transformer_domain_adapter_mode": str(getattr(args, "transformer_domain_adapter_mode", "none")),
        "domain_feature_tap": str(args.domain_feature_tap),
        "grl_lambda": float(args.grl_lambda),
        "grl_warmup_epochs": int(args.grl_warmup_epochs),
        "domain_adv_hidden_dim": int(args.domain_adv_hidden_dim),
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
        "mean_best_epoch_domain_accuracy": float(np.mean(domain_accuracies)),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CD-MambAtt v3 / SPD-only: source supervision + target few-shot supervision + MMD/pseudo/monotonic scaffold + SPD-style DD-Mamba + configurable invariant-path alignment (GRL or MMD)."
    )
    parser.add_argument("--root", default="/home/shelterpl/data/CMAPSS", help="CMAPSS root directory")
    parser.add_argument("--task", default="FD001_TO_FD003", help="Registered cross-domain task name")
    parser.add_argument("--source-subset", default=None, help="Optional source subset override")
    parser.add_argument("--target-subset", default=None, help="Optional target subset override")
    parser.add_argument("--window-size", type=int, default=20, help="Sliding window size")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride")
    parser.add_argument("--rul-clip", type=int, default=125, help="Piece-wise linear RUL cap")
    parser.add_argument("--sensor-subset", choices=("none", "paper14"), default="none", help="Optional input sensor subset preset")
    parser.add_argument("--normalization-mode", choices=("zscore", "minmax"), default="zscore", help="Input normalization mode")
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
    parser.add_argument("--resample-few-shot-per-seed", action=argparse.BooleanOptionalAction, default=True, help="Resample the target few-shot partition for each run seed (default: True to match v2 best-config protocol)")
    parser.add_argument("--seeds", default=None, help="Optional comma-separated seed list; overrides --num-runs")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of run seeds when --seeds is not provided")
    parser.add_argument("--base-seed", type=int, default=42, help="First seed used when --seeds is not provided")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--source-epochs", type=int, default=50, help="Source supervised pretraining epochs")
    parser.add_argument("--target-epochs", type=int, default=20, help="Cross-domain adaptation epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Source-stage learning rate")
    parser.add_argument("--target-lr", type=float, default=5e-4, help="Adaptation-stage learning rate")
    parser.add_argument("--target-lr-scheduler", choices=("none", "cosine"), default="none", help="Optional scheduler for the adaptation-stage optimizer")
    parser.add_argument("--target-lr-min", type=float, default=1e-5, help="Minimum learning rate used by the adaptation-stage scheduler")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Source-stage weight decay")
    parser.add_argument("--target-weight-decay", type=float, default=0.0, help="Adaptation-stage weight decay")
    parser.add_argument("--source-loss-weight", type=float, default=1.0, help="Weight for source supervised loss during adaptation")
    parser.add_argument("--target-loss-weight", type=float, default=1.0, help="Weight for target few-shot supervised loss during adaptation")
    parser.add_argument("--lambda-mmd", type=float, default=0.1, help="Weight for MMD alignment loss")
    parser.add_argument("--mmd-sigmas", default="1,2,4,8,16", help="Comma-separated Gaussian kernel bandwidths for MMD")
    parser.add_argument("--lambda-source-stage", type=float, default=1.0, help="Weight for source stage classification loss")
    parser.add_argument("--lambda-pseudo", type=float, default=0.5, help="Weight for target pseudo-label classification loss")
    parser.add_argument("--lambda-contrastive", type=float, default=0.0, help="Weight for cross-domain N-tuplet / contrastive alignment loss")
    parser.add_argument("--lambda-monotonic", type=float, default=0.05, help="Weight for local target monotonicity loss")
    parser.add_argument("--inv-alignment-mode", choices=("grl", "mmd", "none"), default="grl", help="Invariant-path alignment mode for SPD: gradient reversal, direct MMD, or disabled")
    parser.add_argument("--lambda-domain-adv", type=float, default=0.1, help="Weight for SPD invariant-path domain-adversarial loss")
    parser.add_argument("--lambda-inv-mmd", type=float, default=0.0, help="Weight for SPD invariant-path MMD alignment; when <= 0 and --inv-alignment-mode=mmd, reuse --lambda-domain-adv")
    parser.add_argument("--lambda-spec-domain", type=float, default=0.0, help="Weight for encouraging the specific branch to be domain-predictive via a lightweight domain classifier")
    parser.add_argument("--lambda-conditional-inv-mmd", type=float, default=0.0, help="Weight for stage-conditional MMD on the invariant branch")
    parser.add_argument("--lambda-conditional-proto", type=float, default=0.0, help="Weight for stage-prototype alignment on target invariant features")
    parser.add_argument("--lambda-conditional-proto-ce", type=float, default=0.0, help="Weight for prototype-based stage classification on target conditional features without forcing centroid collapse")
    parser.add_argument(
        "--conditional-target-scope",
        choices=("labeled_accepted", "labeled_only", "accepted_only"),
        default="labeled_accepted",
        help="Which target samples participate in stage-conditional invariant alignment",
    )
    parser.add_argument("--lambda-inv-spec-orth", type=float, default=0.0, help="Weight for invariant/specific feature orthogonality regularization")
    parser.add_argument("--lambda-inv-spec-xcorr", type=float, default=0.0, help="Weight for invariant/specific cross-correlation suppression (a stable MI surrogate)")
    parser.add_argument("--lambda-spec-residual", type=float, default=0.0, help="Weight for residual-size regularization on the specific prediction branch")
    parser.add_argument("--lambda-inv-aux", type=float, default=0.0, help="Weight for supervised invariant-branch RUL prediction")
    parser.add_argument("--lambda-spec-reconstruction", type=float, default=0.0, help="Weight for residual reconstruction of the shared prediction by the specific branch")
    parser.add_argument("--semantic-warmup-epochs", type=int, default=0, help="Optional source-only semantic warmup epochs for inv/spec heads before cross-domain adaptation")
    parser.add_argument("--semantic-warmup-lr", type=float, default=5e-4, help="Learning rate used during semantic warmup")
    parser.add_argument(
        "--adaptation-freeze-mode",
        choices=("none", "head_only", "transformer_head", "spec_gate_only", "spec_gate_head", "spec_gate_transformer_head"),
        default="none",
        help="Freeze strategy for the SPD adaptation stage",
    )
    parser.add_argument(
        "--adaptation-freeze-epochs",
        type=int,
        default=0,
        help="When > 0 and --adaptation-freeze-mode is not none, keep the requested freeze mode for this many adaptation epochs and then unfreeze the full model",
    )
    parser.add_argument("--reuse-source-model-for-adaptation", action="store_true", help="Skip rebuilding/reloading a fresh model between source and adaptation stages and continue adapting the in-memory source-best model")
    parser.add_argument("--grl-lambda", type=float, default=1.0, help="Gradient-reversal strength used by the domain discriminator")
    parser.add_argument("--grl-warmup-epochs", type=int, default=5, help="Linearly warm up the GRL strength over this many adaptation epochs; 0 disables warmup")
    parser.add_argument("--domain-feature-tap", choices=("inv_mean", "pre_transformer_last", "frontend_mean", "concat_inv_pre"), default="pre_transformer_last", help="Feature tap point used by the SPD invariant-alignment branch")
    parser.add_argument("--domain-adv-hidden-dim", type=int, default=16, help="Hidden dimension of the SPD domain discriminator")
    parser.add_argument("--domain-adv-dropout", type=float, default=0.0, help="Dropout used inside the SPD domain discriminator")
    parser.add_argument("--stage-feature-mode", choices=("combined", "invariant"), default="combined", help="Feature branch used for stage statistics and stage classification")
    parser.add_argument("--contrastive-temperature", type=float, default=0.1, help="Temperature used in cross-domain contrastive loss")
    parser.add_argument("--disable-contrastive-feature-normalization", action="store_true", help="Disable L2 normalization before cross-domain contrastive similarity")
    parser.add_argument("--monotonic-margin", type=float, default=0.0, help="Margin used in the local monotonicity ranking loss")
    parser.add_argument("--monotonic-pair-gap", type=int, default=1, help="Window gap used to form local target monotonic pairs")
    parser.add_argument("--monotonic-pair-stride", type=int, default=1, help="Stride used when sampling target monotonic pairs")
    parser.add_argument("--num-pseudo-stages", type=int, default=3, help="Number of degradation stages used for pseudo-labeling")
    parser.add_argument("--pseudo-start-quantile", type=float, default=0.50, help="Initial source-distance quantile used as the pseudo-label acceptance threshold")
    parser.add_argument("--pseudo-end-quantile", type=float, default=0.90, help="Final source-distance quantile used as the pseudo-label acceptance threshold")
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
    parser.add_argument("--mamba-block-mode", choices=("bare", "prenorm_residual", "dd_spd"), default="dd_spd", help="Mamba block mode used in v3; dd_spd is the intended SPD setting")
    parser.add_argument("--spd-gate-init-bias", type=float, default=-2.0, help="Initial bias for the SPD specific-path gate")
    parser.add_argument("--spd-scan-mode", choices=("mixed", "dual_state"), default="mixed", help="SPD scan mode: original parameter-mixing scan or dual-state isolated scan")
    parser.add_argument("--spd-gate-mode", choices=("token", "window"), default="token", help="SPD gate mode: token-wise gate or window-level shared gate")
    parser.add_argument("--spd-gate-scheme", choices=("shared", "dt_bc"), default="shared", help="SPD gate scheme: one shared gate or separate dt/bc gates")
    parser.add_argument("--spd-predictor-mode", choices=("shared_head", "decomposed_residual", "shared_aux_residual"), default="shared_head", help="Prediction head mode for SPD: original shared head, hard decomposed residual predictor, or shared-head-anchored auxiliary decomposition")
    parser.add_argument("--domain-conditioned-gate", action="store_true", help="Enable domain-conditioned gate: add a learnable shift to the SPD gate based on domain label (0=source, 1=target)")
    parser.add_argument("--frontend-adapter-mode", choices=("none", "target_affine", "target_residual"), default="none", help="Optional target-conditioned frontend adapter inserted after conv1d inside DD-Mamba")
    parser.add_argument("--transformer-domain-adapter-mode", choices=("none", "target_shift", "target_film"), default="none", help="Optional target-conditioned affine adapter inside each custom Transformer block")
    parser.add_argument("--lambda-transformer-domain-adapter-l2", type=float, default=0.0, help="L2 regularization weight for the target-conditioned Transformer adapter parameters")
    parser.add_argument("--source-val-all-windows", action=argparse.BooleanOptionalAction, default=True, help="Validate source stage on all source validation windows (default: True to match v2 best-config protocol)")
    parser.add_argument("--target-val-all-windows", action=argparse.BooleanOptionalAction, default=True, help="Validate target adaptation stage on all target validation windows (default: True to match v2 best-config protocol)")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--max-source-train-batches", type=int, default=None, help="Optional cap for source-stage smoke tests")
    parser.add_argument("--max-target-train-batches", type=int, default=None, help="Optional cap for adaptation-stage smoke tests")
    parser.add_argument("--output-dir", default="/home/shelterpl/cd_mambatt/runs/cd_mambatt_v3", help="Run output directory")
    args = parser.parse_args()

    if args.target_shots <= 0:
        raise ValueError("CD-MambAtt v3 expects target_shots > 0")

    from cd_mambatt.cross_domain import resolve_cross_domain_task

    task = resolve_cross_domain_task(args.task, source_subset=args.source_subset, target_subset=args.target_subset)
    device = infer_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("CD-MambAtt v3 requires CUDA in the configured environment.")

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
