from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SourceStageStatistics:
    centroids: torch.Tensor
    thresholds: torch.Tensor
    counts: torch.Tensor
    mean_rul: torch.Tensor
    quantile: float
    normalize_features: bool


def assign_rul_stage_labels(
    rul_values: torch.Tensor,
    *,
    rul_clip: float,
    num_stages: int = 3,
) -> torch.Tensor:
    if num_stages < 2:
        raise ValueError("num_stages must be >= 2")
    boundaries = torch.linspace(
        0.0,
        float(rul_clip),
        steps=num_stages + 1,
        device=rul_values.device,
        dtype=rul_values.dtype,
    )[1:-1]
    return torch.bucketize(rul_values, boundaries=boundaries, right=False).long()


def _prepare_features(features: torch.Tensor, normalize_features: bool) -> torch.Tensor:
    if normalize_features:
        return F.normalize(features, p=2, dim=1)
    return features


def compute_source_stage_statistics(
    features: torch.Tensor,
    stage_labels: torch.Tensor,
    rul_values: torch.Tensor,
    *,
    num_stages: int,
    quantile: float,
    normalize_features: bool = True,
    min_threshold: float = 1e-6,
) -> SourceStageStatistics:
    if features.ndim != 2:
        raise ValueError("features must be a 2D tensor")
    if stage_labels.ndim != 1 or rul_values.ndim != 1:
        raise ValueError("stage_labels and rul_values must be 1D tensors")
    if features.shape[0] != stage_labels.shape[0] or features.shape[0] != rul_values.shape[0]:
        raise ValueError("features, stage_labels, and rul_values must have matching first dimension")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")

    prepared_features = _prepare_features(features, normalize_features)
    feature_dim = prepared_features.shape[1]
    centroids = torch.empty((num_stages, feature_dim), dtype=prepared_features.dtype, device=prepared_features.device)
    thresholds = torch.empty((num_stages,), dtype=prepared_features.dtype, device=prepared_features.device)
    counts = torch.empty((num_stages,), dtype=torch.long, device=prepared_features.device)
    mean_rul = torch.empty((num_stages,), dtype=rul_values.dtype, device=rul_values.device)

    global_centroid = prepared_features.mean(dim=0)
    if normalize_features:
        global_centroid = F.normalize(global_centroid.unsqueeze(0), p=2, dim=1).squeeze(0)

    for stage_idx in range(num_stages):
        mask = stage_labels == stage_idx
        count = int(mask.sum().item())
        counts[stage_idx] = count
        if count == 0:
            centroids[stage_idx] = global_centroid
            thresholds[stage_idx] = torch.as_tensor(min_threshold, dtype=prepared_features.dtype, device=prepared_features.device)
            mean_rul[stage_idx] = rul_values.mean()
            continue

        stage_features = prepared_features[mask]
        centroid = stage_features.mean(dim=0)
        if normalize_features:
            centroid = F.normalize(centroid.unsqueeze(0), p=2, dim=1).squeeze(0)
        centroids[stage_idx] = centroid

        distances = torch.norm(stage_features - centroid.unsqueeze(0), dim=1)
        if distances.numel() == 1:
            threshold = distances.max()
        else:
            threshold = torch.quantile(distances, q=quantile)
        thresholds[stage_idx] = torch.clamp(threshold, min=min_threshold)
        mean_rul[stage_idx] = rul_values[mask].mean()

    return SourceStageStatistics(
        centroids=centroids,
        thresholds=thresholds,
        counts=counts,
        mean_rul=mean_rul,
        quantile=float(quantile),
        normalize_features=normalize_features,
    )


def assign_pseudo_stage_labels(
    features: torch.Tensor,
    stage_statistics: SourceStageStatistics,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if features.ndim != 2:
        raise ValueError("features must be a 2D tensor")

    prepared_features = _prepare_features(features, stage_statistics.normalize_features)
    distances = torch.cdist(prepared_features, stage_statistics.centroids)
    min_distances, assigned_labels = torch.min(distances, dim=1)
    accepted_mask = min_distances <= stage_statistics.thresholds[assigned_labels]
    return assigned_labels, accepted_mask, min_distances

