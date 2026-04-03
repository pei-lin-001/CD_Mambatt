from __future__ import annotations

import numpy as np


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    errors = predictions.astype(np.float64) - targets.astype(np.float64)
    return float(np.sqrt(np.mean(errors ** 2)))


def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    errors = predictions.astype(np.float64) - targets.astype(np.float64)
    return float(np.mean(np.abs(errors)))


def nasa_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    errors = predictions.astype(np.float64) - targets.astype(np.float64)
    score_terms = np.where(
        errors < 0.0,
        np.exp(-errors / 13.0) - 1.0,
        np.exp(errors / 10.0) - 1.0,
    )
    return float(np.sum(score_terms))
