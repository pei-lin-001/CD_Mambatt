from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function


class _GradientReverseFn(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


def gradient_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradientReverseFn.apply(x, float(lambda_))


class DomainDiscriminator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, *, grl_lambda: float = 1.0) -> torch.Tensor:
        return self.net(gradient_reverse(features, grl_lambda))


def compute_domain_adversarial_loss(
    discriminator: DomainDiscriminator,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    *,
    grl_lambda: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if source_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("domain adversarial loss expects 2D feature tensors")
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("source and target domain features must share the same dimension")
    if source_features.shape[0] == 0 or target_features.shape[0] == 0:
        raise ValueError("domain adversarial loss expects non-empty source and target batches")

    features = torch.cat([source_features, target_features], dim=0)
    logits = discriminator(features, grl_lambda=grl_lambda)
    labels = torch.cat(
        [
            torch.zeros(source_features.shape[0], dtype=torch.long, device=features.device),
            torch.ones(target_features.shape[0], dtype=torch.long, device=features.device),
        ],
        dim=0,
    )
    loss = F.cross_entropy(logits, labels)
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == labels).to(dtype=features.dtype).mean()
    source_accuracy = (predictions[: source_features.shape[0]] == 0).to(dtype=features.dtype).mean()
    target_accuracy = (predictions[source_features.shape[0] :] == 1).to(dtype=features.dtype).mean()
    return loss, {
        "domain_accuracy": float(accuracy.detach().cpu()),
        "source_domain_accuracy": float(source_accuracy.detach().cpu()),
        "target_domain_accuracy": float(target_accuracy.detach().cpu()),
        "source_domain_count": float(source_features.shape[0]),
        "target_domain_count": float(target_features.shape[0]),
    }
