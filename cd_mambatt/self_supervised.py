from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from cd_mambatt.data import PSEUDO_LABEL_SENSOR_DROP, WindowedData
from cd_mambatt.models import MambAttRegressor


DEFAULT_TEMPORAL_ORDERED_PATTERNS: tuple[tuple[int, int, int], ...] = (
    (-1, 0, 1),
    (-2, 0, 3),
    (-3, -1, 1),
    (-3, 0, 2),
    (-2, -1, 1),
    (-1, 1, 2),
)

DEFAULT_TEMPORAL_DISORDERED_PATTERNS: tuple[tuple[int, int, int], ...] = (
    (2, 0, 1),
    (1, -1, 0),
)


@dataclass(frozen=True)
class SensorPseudoLabelStatistics:
    sensor_indices: np.ndarray
    centroids: np.ndarray
    boundary_b1: np.ndarray
    boundary_b2: np.ndarray
    min_values: np.ndarray


class IndexedWindowPairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, windows: np.ndarray, pair_indices: np.ndarray) -> None:
        self.windows = torch.from_numpy(windows.astype(np.float32))
        self.pair_indices = torch.from_numpy(pair_indices.astype(np.int64))

    def __len__(self) -> int:
        return int(self.pair_indices.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        query_idx, positive_idx = self.pair_indices[index]
        return self.windows[query_idx], self.windows[positive_idx]


class IndexedWindowTripletDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, windows: np.ndarray, triplet_indices: np.ndarray, labels: np.ndarray) -> None:
        self.windows = torch.from_numpy(windows.astype(np.float32))
        self.triplet_indices = torch.from_numpy(triplet_indices.astype(np.int64))
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self) -> int:
        return int(self.triplet_indices.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        triplet = self.windows[self.triplet_indices[index]]
        return triplet, self.labels[index]


class WindowPseudoLabelDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, windows: np.ndarray, pseudo_labels: np.ndarray) -> None:
        self.windows = torch.from_numpy(windows.astype(np.float32))
        self.pseudo_labels = torch.from_numpy(pseudo_labels.astype(np.int64))

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[index], self.pseudo_labels[index]


def get_pseudo_label_sensor_indices(feature_dim: int) -> np.ndarray:
    dropped = {sensor_id - 1 for sensor_id in PSEUDO_LABEL_SENSOR_DROP}
    return np.asarray([idx for idx in range(feature_dim) if idx not in dropped], dtype=np.int64)


def build_consecutive_pair_indices(windowed: WindowedData) -> np.ndarray:
    pair_indices: list[tuple[int, int]] = []
    for unit_id in np.unique(windowed.unit_ids):
        unit_mask = windowed.unit_ids == unit_id
        ordered_indices = np.where(unit_mask)[0]
        ordered_indices = ordered_indices[np.argsort(windowed.end_cycles[ordered_indices])]
        for first_idx, second_idx in zip(ordered_indices[:-1], ordered_indices[1:]):
            pair_indices.append((int(first_idx), int(second_idx)))

    if not pair_indices:
        raise ValueError("No consecutive pair indices were generated")
    return np.asarray(pair_indices, dtype=np.int64)


def build_temporal_triplet_indices(
    windowed: WindowedData,
    *,
    ordered_patterns: tuple[tuple[int, int, int], ...] = DEFAULT_TEMPORAL_ORDERED_PATTERNS,
    disordered_patterns: tuple[tuple[int, int, int], ...] = DEFAULT_TEMPORAL_DISORDERED_PATTERNS,
) -> tuple[np.ndarray, np.ndarray]:
    triplet_indices: list[tuple[int, int, int]] = []
    labels: list[float] = []
    pattern_groups = tuple((pattern, 1.0) for pattern in ordered_patterns) + tuple((pattern, 0.0) for pattern in disordered_patterns)

    for unit_id in np.unique(windowed.unit_ids):
        unit_mask = windowed.unit_ids == unit_id
        ordered_indices = np.where(unit_mask)[0]
        ordered_indices = ordered_indices[np.argsort(windowed.end_cycles[ordered_indices])]
        unit_count = int(ordered_indices.shape[0])
        for center_pos in range(unit_count):
            for offsets, label in pattern_groups:
                triplet_positions = [center_pos + offset for offset in offsets]
                if min(triplet_positions) < 0 or max(triplet_positions) >= unit_count:
                    continue
                triplet = tuple(int(ordered_indices[pos]) for pos in triplet_positions)
                triplet_indices.append(triplet)
                labels.append(float(label))

    if not triplet_indices:
        raise ValueError("No temporal triplet indices were generated")
    return np.asarray(triplet_indices, dtype=np.int64), np.asarray(labels, dtype=np.float32)


def _kmeans_1d(values: np.ndarray, *, k: int = 3, max_iters: int = 100) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("values must be a 1D array")
    if values.size == 0:
        raise ValueError("values must not be empty")

    values = values.astype(np.float64, copy=False)
    quantiles = np.linspace(0.0, 1.0, num=k + 2, endpoint=True)[1:-1]
    centroids = np.quantile(values, quantiles)
    if np.unique(centroids).size == 1:
        min_value = float(values.min())
        max_value = float(values.max())
        if max_value - min_value < 1e-8:
            return np.linspace(min_value - 1e-3, max_value + 1e-3, num=k, dtype=np.float64)
        centroids = np.linspace(min_value, max_value, num=k, dtype=np.float64)

    for _ in range(max_iters):
        distances = np.abs(values[:, None] - centroids[None, :])
        assignments = np.argmin(distances, axis=1)
        updated = centroids.copy()
        for cluster_idx in range(k):
            mask = assignments == cluster_idx
            if np.any(mask):
                updated[cluster_idx] = float(values[mask].mean())
        if np.allclose(updated, centroids, atol=1e-6):
            centroids = updated
            break
        centroids = updated
    return np.sort(centroids.astype(np.float32))


def fit_sensor_pseudo_label_statistics(normalized_sensor_rows: np.ndarray) -> SensorPseudoLabelStatistics:
    if normalized_sensor_rows.ndim != 2:
        raise ValueError("normalized_sensor_rows must be a 2D array")

    feature_dim = int(normalized_sensor_rows.shape[1])
    sensor_indices = get_pseudo_label_sensor_indices(feature_dim)
    centroids: list[np.ndarray] = []
    boundary_b1: list[float] = []
    boundary_b2: list[float] = []
    min_values: list[float] = []

    for sensor_idx in sensor_indices:
        values = normalized_sensor_rows[:, sensor_idx].astype(np.float32, copy=False)
        clusters = _kmeans_1d(values, k=3)
        b1 = float((clusters[1] + clusters[2]) / 2.0)
        minimum = float(values.min())
        b2 = float((minimum + b1) / 2.0)
        centroids.append(clusters)
        boundary_b1.append(b1)
        boundary_b2.append(b2)
        min_values.append(minimum)

    return SensorPseudoLabelStatistics(
        sensor_indices=sensor_indices.astype(np.int64),
        centroids=np.stack(centroids).astype(np.float32),
        boundary_b1=np.asarray(boundary_b1, dtype=np.float32),
        boundary_b2=np.asarray(boundary_b2, dtype=np.float32),
        min_values=np.asarray(min_values, dtype=np.float32),
    )


def assign_window_sensor_pseudo_labels(
    windows: np.ndarray,
    statistics: SensorPseudoLabelStatistics,
    *,
    value_mode: str = "last",
) -> np.ndarray:
    if windows.ndim != 3:
        raise ValueError("windows must be a 3D array")
    if value_mode not in {"last", "mean"}:
        raise ValueError("value_mode must be 'last' or 'mean'")

    if value_mode == "last":
        sensor_values = windows[:, -1, statistics.sensor_indices]
    else:
        sensor_values = windows[:, :, statistics.sensor_indices].mean(axis=1)

    labels = np.zeros(sensor_values.shape, dtype=np.int64)
    labels[sensor_values >= statistics.boundary_b2[None, :]] = 1
    labels[sensor_values >= statistics.boundary_b1[None, :]] = 2
    return labels


def ntuplet_loss(query_embeddings: torch.Tensor, positive_embeddings: torch.Tensor) -> torch.Tensor:
    if query_embeddings.ndim != 2 or positive_embeddings.ndim != 2:
        raise ValueError("query_embeddings and positive_embeddings must be 2D tensors")
    if query_embeddings.shape != positive_embeddings.shape:
        raise ValueError("query_embeddings and positive_embeddings must have the same shape")
    batch_size = int(query_embeddings.shape[0])
    if batch_size <= 1:
        return query_embeddings.new_zeros(())

    similarity = query_embeddings @ positive_embeddings.transpose(0, 1)
    positive_scores = similarity.diag().unsqueeze(1)
    margin_scores = similarity - positive_scores
    negative_mask = ~torch.eye(batch_size, dtype=torch.bool, device=query_embeddings.device)
    negatives = margin_scores[negative_mask].view(batch_size, batch_size - 1)
    zeros = torch.zeros((batch_size, 1), dtype=negatives.dtype, device=negatives.device)
    return torch.logsumexp(torch.cat([zeros, negatives], dim=1), dim=1).mean()


class MambAttSelfSupervisedPretrainer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        num_mamba_layers: int,
        num_transformer_layers: int,
        num_heads: int,
        dropout: float,
        dim_feedforward: int,
        transformer_impl: str,
        transformer_norm_mode: str,
        transformer_inner_dropout: float,
        mamba_block_mode: str,
        window_size: int,
        pseudo_sensor_indices: np.ndarray,
        temporal_hidden_dim: int = 128,
        pseudo_hidden_dim: int = 64,
        ntuplet_mode: str = "flatten",
        ntuplet_normalize: bool = False,
    ) -> None:
        super().__init__()
        if ntuplet_mode not in {"flatten", "last", "mean"}:
            raise ValueError("ntuplet_mode must be 'flatten', 'last', or 'mean'")
        self.backbone = MambAttRegressor(
            input_dim=input_dim,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_mamba_layers=num_mamba_layers,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            transformer_impl=transformer_impl,
            transformer_norm_mode=transformer_norm_mode,
            transformer_inner_dropout=transformer_inner_dropout,
            mamba_block_mode=mamba_block_mode,
        )
        self.window_size = int(window_size)
        self.d_model = int(d_model)
        self.input_dim = int(input_dim)
        self.ntuplet_mode = ntuplet_mode
        self.ntuplet_normalize = bool(ntuplet_normalize)
        self.register_buffer(
            "pseudo_sensor_indices",
            torch.as_tensor(pseudo_sensor_indices, dtype=torch.long),
            persistent=False,
        )
        temporal_input_dim = 3 * self.window_size * self.d_model
        self.temporal_classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(temporal_input_dim, temporal_hidden_dim),
            nn.ReLU(),
            nn.Linear(temporal_hidden_dim, 1),
        )
        if self.ntuplet_mode == "flatten":
            self.ntuplet_projector = nn.Identity()
        else:
            self.ntuplet_projector = nn.Identity()
        self.pseudo_classifier = nn.Sequential(
            nn.Linear(self.window_size, pseudo_hidden_dim),
            nn.ReLU(),
            nn.Linear(pseudo_hidden_dim, 3),
        )

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_mamba_sequence(x)

    def forward_temporal_logits(self, triplets: torch.Tensor) -> torch.Tensor:
        batch_size, triplet_count, window_size, feature_dim = triplets.shape
        if triplet_count != 3:
            raise ValueError("Temporal triplets must contain exactly 3 windows")
        if window_size != self.window_size:
            raise ValueError(f"Expected temporal triplets with window size {self.window_size}, got {window_size}")
        if feature_dim != self.input_dim:
            raise ValueError(f"Expected temporal triplets with input_dim {self.input_dim}, got {feature_dim}")
        encoded = self.encode_sequence(triplets.reshape(batch_size * triplet_count, window_size, feature_dim))
        encoded = encoded.reshape(batch_size, triplet_count, window_size, self.d_model)
        return self.temporal_classifier(encoded).squeeze(-1)

    def forward_ntuplet_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_sequence(x)
        if self.ntuplet_mode == "flatten":
            embedding = encoded.reshape(encoded.shape[0], -1)
        elif self.ntuplet_mode == "mean":
            embedding = encoded.mean(dim=1)
        else:
            embedding = encoded[:, -1, :]
        projected = self.ntuplet_projector(embedding)
        if self.ntuplet_normalize:
            return F.normalize(projected, p=2, dim=-1)
        return projected

    def forward_pseudo_logits(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_sequence(x)
        selected = encoded[:, :, self.pseudo_sensor_indices]
        traces = selected.transpose(1, 2).contiguous()
        logits = self.pseudo_classifier(traces.view(-1, self.window_size))
        return logits.view(x.shape[0], self.pseudo_sensor_indices.numel(), 3)

    def export_encoder_state_dict(self) -> dict[str, torch.Tensor]:
        return MambAttRegressor.extract_encoder_state_dict(self.backbone.state_dict())
