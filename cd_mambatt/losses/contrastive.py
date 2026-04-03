from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_domain_contrastive_loss(
    source_features: torch.Tensor,
    source_labels: torch.Tensor,
    target_features: torch.Tensor,
    target_labels: torch.Tensor,
    *,
    temperature: float = 0.1,
    normalize_features: bool = True,
    min_positives_per_anchor: int = 1,
) -> tuple[torch.Tensor, dict[str, float]]:
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("contrastive loss expects 2D feature tensors")
    if source_labels.ndim != 1 or target_labels.ndim != 1:
        raise ValueError("contrastive loss expects 1D label tensors")
    if source_features.shape[0] != source_labels.shape[0]:
        raise ValueError("source_features and source_labels must align")
    if target_features.shape[0] != target_labels.shape[0]:
        raise ValueError("target_features and target_labels must align")
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("source and target feature dimensions must match")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if min_positives_per_anchor < 1:
        raise ValueError("min_positives_per_anchor must be >= 1")

    zero = source_features.new_zeros(())
    if source_features.shape[0] == 0 or target_features.shape[0] == 0:
        return zero, {
            "valid_anchor_ratio": 0.0,
            "mean_positive_count": 0.0,
            "target_bank_size": float(target_features.shape[0]),
        }

    if normalize_features:
        source_features = F.normalize(source_features, p=2, dim=1)
        target_features = F.normalize(target_features, p=2, dim=1)

    logits = source_features @ target_features.transpose(0, 1)
    logits = logits / temperature

    positive_mask = source_labels.unsqueeze(1) == target_labels.unsqueeze(0)
    negative_mask = ~positive_mask
    positive_counts = positive_mask.sum(dim=1)
    valid_anchor_mask = (positive_counts >= min_positives_per_anchor) & negative_mask.any(dim=1)

    if not bool(valid_anchor_mask.any().item()):
        return zero, {
            "valid_anchor_ratio": 0.0,
            "mean_positive_count": 0.0,
            "target_bank_size": float(target_features.shape[0]),
        }

    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_mask_float = positive_mask.to(dtype=source_features.dtype)
    per_anchor_loss = -(log_prob * positive_mask_float).sum(dim=1) / positive_counts.clamp(min=1).to(source_features.dtype)
    loss = per_anchor_loss[valid_anchor_mask].mean()

    return loss, {
        "valid_anchor_ratio": float(valid_anchor_mask.to(dtype=source_features.dtype).mean().detach().cpu()),
        "mean_positive_count": float(positive_counts[valid_anchor_mask].to(dtype=source_features.dtype).mean().detach().cpu()),
        "target_bank_size": float(target_features.shape[0]),
    }
