from __future__ import annotations

import torch


def _pairwise_squared_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1, keepdim=True).transpose(0, 1)
    distances = x_norm + y_norm - 2.0 * (x @ y.transpose(0, 1))
    return torch.clamp(distances, min=0.0)


def _gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    distances = _pairwise_squared_distances(x, y).unsqueeze(0)
    gamma = 1.0 / (2.0 * sigmas.view(-1, 1, 1) ** 2)
    kernels = torch.exp(-gamma * distances)
    return kernels.sum(dim=0)


def gaussian_mmd_loss(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    *,
    sigmas: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0),
) -> torch.Tensor:
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("MMD expects 2D feature tensors of shape (batch, feature_dim)")
    if source_features.shape[0] == 0 or target_features.shape[0] == 0:
        raise ValueError("MMD expects non-empty source and target batches")

    sigmas_tensor = torch.as_tensor(sigmas, dtype=source_features.dtype, device=source_features.device)
    k_xx = _gaussian_kernel(source_features, source_features, sigmas_tensor)
    k_yy = _gaussian_kernel(target_features, target_features, sigmas_tensor)
    k_xy = _gaussian_kernel(source_features, target_features, sigmas_tensor)
    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
