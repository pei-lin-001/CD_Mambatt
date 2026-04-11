from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


SUBSETS = {"FD001", "FD002", "FD003", "FD004"}
CONDITION_NORMALIZED_SUBSETS = {"FD002", "FD004"}
CONDITION_KEY_DECIMALS = 0
PSEUDO_LABEL_SENSOR_DROP = (1, 5, 6, 10, 16, 18, 19)
PAPER_14_SENSOR_IDS_1_BASED = (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)
PAPER_14_SENSOR_INDICES = tuple(sensor_id - 1 for sensor_id in PAPER_14_SENSOR_IDS_1_BASED)
SENSOR_SUBSET_PRESETS: dict[str, tuple[int, ...]] = {
    "paper14": PAPER_14_SENSOR_INDICES,
}
OPERATING_SETTING_NAMES = [f"op_setting_{idx}" for idx in range(1, 4)]
SENSOR_NAMES = [f"sensor_{idx}" for idx in range(1, 22)]
FEATURE_NAMES = SENSOR_NAMES


@dataclass(frozen=True)
class CMAPSSSplit:
    subset: str
    split: str
    unit_ids: np.ndarray
    cycles: np.ndarray
    operating_settings: np.ndarray
    sensors: np.ndarray
    rul: np.ndarray
    condition_ids: np.ndarray | None = None

    @property
    def num_rows(self) -> int:
        return int(self.unit_ids.shape[0])

    @property
    def num_units(self) -> int:
        return int(np.unique(self.unit_ids).size)

    @property
    def feature_dim(self) -> int:
        return int(self.sensors.shape[1])


@dataclass(frozen=True)
class CMAPSSNormalizer:
    subset: str
    mode: str
    use_condition_normalization: bool
    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray
    condition_mean: dict[int, np.ndarray]
    condition_std: dict[int, np.ndarray]
    condition_min: dict[int, np.ndarray]
    condition_max: dict[int, np.ndarray]
    condition_key_to_id: dict[tuple[float, float, float], int]
    condition_centers: np.ndarray

    def transform(self, split: CMAPSSSplit) -> CMAPSSSplit:
        if self.use_condition_normalization:
            condition_ids = np.asarray([self._lookup_condition_id(row) for row in split.operating_settings], dtype=np.int32)
            sensors = np.empty_like(split.sensors, dtype=np.float32)
            for condition_id in np.unique(condition_ids):
                mask = condition_ids == condition_id
                if self.mode == "zscore":
                    sensors[mask] = (
                        split.sensors[mask] - self.condition_mean[int(condition_id)]
                    ) / self.condition_std[int(condition_id)]
                elif self.mode == "minmax":
                    cond_min = self.condition_min[int(condition_id)]
                    cond_max = self.condition_max[int(condition_id)]
                    sensors[mask] = 2.0 * (split.sensors[mask] - cond_min) / (cond_max - cond_min) - 1.0
                else:
                    raise ValueError(f"Unknown normalization mode: {self.mode}")
        else:
            condition_ids = split.condition_ids
            if self.mode == "zscore":
                sensors = (split.sensors - self.mean) / self.std
            elif self.mode == "minmax":
                sensors = 2.0 * (split.sensors - self.min) / (self.max - self.min) - 1.0
            else:
                raise ValueError(f"Unknown normalization mode: {self.mode}")
        return CMAPSSSplit(
            subset=split.subset,
            split=split.split,
            unit_ids=split.unit_ids,
            cycles=split.cycles,
            operating_settings=split.operating_settings,
            sensors=sensors.astype(np.float32),
            rul=split.rul.astype(np.float32),
            condition_ids=condition_ids,
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "subset": self.subset,
            "feature_names": FEATURE_NAMES,
            "mode": self.mode,
            "use_condition_normalization": self.use_condition_normalization,
        }
        if self.use_condition_normalization:
            payload["condition_stats"] = {
                str(condition_id): {
                    "mean": self.condition_mean[condition_id].tolist(),
                    "std": self.condition_std[condition_id].tolist(),
                    "min": self.condition_min[condition_id].tolist(),
                    "max": self.condition_max[condition_id].tolist(),
                }
                for condition_id in sorted(self.condition_mean)
            }
            payload["condition_key_to_id"] = {
                ",".join(f"{value:.6f}" for value in key): value
                for key, value in self.condition_key_to_id.items()
            }
            payload["condition_centers"] = self.condition_centers.tolist()
        else:
            payload["mean"] = self.mean.tolist()
            payload["std"] = self.std.tolist()
            payload["min"] = self.min.tolist()
            payload["max"] = self.max.tolist()
        return payload

    def _lookup_condition_id(self, operating_setting: np.ndarray) -> int:
        key = _condition_key(operating_setting, decimals=CONDITION_KEY_DECIMALS)
        if key in self.condition_key_to_id:
            return self.condition_key_to_id[key]
        distances = np.linalg.norm(self.condition_centers - operating_setting[None, :], axis=1)
        return int(np.argmin(distances))


@dataclass(frozen=True)
class WindowedData:
    windows: np.ndarray
    targets: np.ndarray
    unit_ids: np.ndarray
    end_cycles: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(dim) for dim in self.windows.shape)


@dataclass(frozen=True)
class WindowPairData:
    earlier_windows: np.ndarray
    later_windows: np.ndarray
    unit_ids: np.ndarray
    earlier_end_cycles: np.ndarray
    later_end_cycles: np.ndarray

    @property
    def num_pairs(self) -> int:
        return int(self.unit_ids.shape[0])


class CMAPSSWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data: WindowedData) -> None:
        self.windows = torch.from_numpy(data.windows.astype(np.float32))
        self.targets = torch.from_numpy(data.targets.astype(np.float32))

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[index], self.targets[index]


class CMAPSSWindowPairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data: WindowPairData) -> None:
        self.earlier_windows = torch.from_numpy(data.earlier_windows.astype(np.float32))
        self.later_windows = torch.from_numpy(data.later_windows.astype(np.float32))

    def __len__(self) -> int:
        return int(self.earlier_windows.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.earlier_windows[index], self.later_windows[index]


def _resolve_root(root: str | Path) -> Path:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"CMAPSS root does not exist: {root_path}")
    return root_path


def _validate_subset(subset: str) -> str:
    subset = subset.upper()
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset '{subset}'. Expected one of: {sorted(SUBSETS)}")
    return subset


def _split_file(root: Path, subset: str, split: str) -> Path:
    filename = f"{split}_{subset}.txt"
    if split == "RUL":
        filename = f"RUL_{subset}.txt"
    path = root / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file for {subset} {split}: {path}")
    return path


def _load_matrix(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    return matrix


def _compute_train_rul(unit_ids: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    max_cycles = np.zeros(int(unit_ids.max()) + 1, dtype=np.int32)
    for unit_id, cycle in zip(unit_ids, cycles):
        if cycle > max_cycles[unit_id]:
            max_cycles[unit_id] = cycle
    return (max_cycles[unit_ids] - cycles).astype(np.float32)


def _compute_test_rul(root: Path, subset: str, unit_ids: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    final_rul = np.loadtxt(_split_file(root, subset, "RUL"), dtype=np.float32)
    if final_rul.ndim == 0:
        final_rul = final_rul[None]

    observed_last_cycle = np.zeros(int(unit_ids.max()) + 1, dtype=np.int32)
    for unit_id, cycle in zip(unit_ids, cycles):
        if cycle > observed_last_cycle[unit_id]:
            observed_last_cycle[unit_id] = cycle

    total_life = np.zeros_like(observed_last_cycle, dtype=np.float32)
    total_life[1 : final_rul.shape[0] + 1] = observed_last_cycle[1 : final_rul.shape[0] + 1] + final_rul
    return total_life[unit_ids] - cycles.astype(np.float32)


def _condition_key(values: np.ndarray, decimals: int = CONDITION_KEY_DECIMALS) -> tuple[float, float, float]:
    rounded = np.round(values.astype(np.float64), decimals)
    return tuple(float(value) for value in rounded.tolist())


def load_cmapss_split(
    root: str | Path,
    subset: str,
    split: str,
    *,
    rul_clip: int = 125,
) -> CMAPSSSplit:
    root_path = _resolve_root(root)
    subset = _validate_subset(subset)
    split = split.lower()
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    raw = _load_matrix(_split_file(root_path, subset, split))
    unit_ids = raw[:, 0].astype(np.int32)
    cycles = raw[:, 1].astype(np.int32)
    operating_settings = raw[:, 2:5].astype(np.float32)
    sensors = raw[:, 5:26].astype(np.float32)

    rul = _compute_train_rul(unit_ids, cycles) if split == "train" else _compute_test_rul(root_path, subset, unit_ids, cycles)
    rul = np.minimum(rul, np.float32(rul_clip))

    return CMAPSSSplit(
        subset=subset,
        split=split,
        unit_ids=unit_ids,
        cycles=cycles,
        operating_settings=operating_settings,
        sensors=sensors,
        rul=rul,
        condition_ids=None,
    )


def select_units(split: CMAPSSSplit, unit_ids: np.ndarray | list[int]) -> CMAPSSSplit:
    selected = np.asarray(unit_ids, dtype=np.int32)
    mask = np.isin(split.unit_ids, selected)
    return CMAPSSSplit(
        subset=split.subset,
        split=split.split,
        unit_ids=split.unit_ids[mask],
        cycles=split.cycles[mask],
        operating_settings=split.operating_settings[mask],
        sensors=split.sensors[mask],
        rul=split.rul[mask],
        condition_ids=None if split.condition_ids is None else split.condition_ids[mask],
    )


def resolve_sensor_subset_preset(sensor_subset: str | None) -> tuple[int, ...] | None:
    if sensor_subset is None:
        return None
    key = sensor_subset.strip().lower()
    if key in {"", "none", "all"}:
        return None
    if key not in SENSOR_SUBSET_PRESETS:
        raise ValueError(f"Unknown sensor subset preset '{sensor_subset}'. Expected one of: {sorted(SENSOR_SUBSET_PRESETS)}")
    return SENSOR_SUBSET_PRESETS[key]


def select_sensor_subset(split: CMAPSSSplit, sensor_indices: tuple[int, ...] | list[int] | np.ndarray | None) -> CMAPSSSplit:
    if sensor_indices is None:
        return split
    indices = np.asarray(sensor_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("sensor_indices must be a non-empty 1D collection")
    if np.any(indices < 0) or np.any(indices >= split.sensors.shape[1]):
        raise ValueError(
            f"sensor_indices must be within [0, {split.sensors.shape[1] - 1}] for split with feature_dim={split.sensors.shape[1]}"
        )
    return CMAPSSSplit(
        subset=split.subset,
        split=split.split,
        unit_ids=split.unit_ids,
        cycles=split.cycles,
        operating_settings=split.operating_settings,
        sensors=split.sensors[:, indices].astype(np.float32),
        rul=split.rul,
        condition_ids=split.condition_ids,
    )


def get_train_validation_unit_ids(
    split: CMAPSSSplit,
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if split.split != "train":
        raise ValueError("get_train_validation_unit_ids expects a training split")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    unique_units = np.unique(split.unit_ids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_units)
    cutoff = max(1, int(len(shuffled) * train_ratio))
    train_units = np.sort(shuffled[:cutoff]).astype(np.int32)
    val_units = np.sort(shuffled[cutoff:]).astype(np.int32)
    if val_units.size == 0:
        raise ValueError("Validation unit split is empty; adjust train_ratio")
    return train_units, val_units


def split_train_validation_by_unit(
    split: CMAPSSSplit,
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[CMAPSSSplit, CMAPSSSplit]:
    train_units, val_units = get_train_validation_unit_ids(split, train_ratio=train_ratio, seed=seed)
    return select_units(split, train_units), select_units(split, val_units)


def fit_normalizer(split: CMAPSSSplit, eps: float = 1e-8, mode: str = "zscore") -> CMAPSSNormalizer:
    subset = _validate_subset(split.subset)
    mode = mode.lower()
    if mode not in {"zscore", "minmax"}:
        raise ValueError("mode must be either 'zscore' or 'minmax'")
    if subset in CONDITION_NORMALIZED_SUBSETS:
        condition_keys = [_condition_key(row, decimals=CONDITION_KEY_DECIMALS) for row in split.operating_settings]
        unique_keys = sorted(set(condition_keys))
        condition_key_to_id = {key: idx for idx, key in enumerate(unique_keys)}
        condition_ids = np.asarray([condition_key_to_id[key] for key in condition_keys], dtype=np.int32)
        condition_mean: dict[int, np.ndarray] = {}
        condition_std: dict[int, np.ndarray] = {}
        condition_min: dict[int, np.ndarray] = {}
        condition_max: dict[int, np.ndarray] = {}
        for condition_id in np.unique(condition_ids):
            mask = condition_ids == condition_id
            mean = split.sensors[mask].mean(axis=0)
            std = split.sensors[mask].std(axis=0)
            cond_min = split.sensors[mask].min(axis=0)
            cond_max = split.sensors[mask].max(axis=0)
            condition_mean[int(condition_id)] = mean.astype(np.float32)
            condition_std[int(condition_id)] = np.where(std < eps, 1.0, std).astype(np.float32)
            condition_min[int(condition_id)] = cond_min.astype(np.float32)
            condition_max[int(condition_id)] = np.where((cond_max - cond_min) < eps, cond_min + 1.0, cond_max).astype(np.float32)
        return CMAPSSNormalizer(
            subset=subset,
            mode=mode,
            use_condition_normalization=True,
            mean=np.zeros(split.feature_dim, dtype=np.float32),
            std=np.ones(split.feature_dim, dtype=np.float32),
            min=np.zeros(split.feature_dim, dtype=np.float32),
            max=np.ones(split.feature_dim, dtype=np.float32),
            condition_mean=condition_mean,
            condition_std=condition_std,
            condition_min=condition_min,
            condition_max=condition_max,
            condition_key_to_id=condition_key_to_id,
            condition_centers=np.asarray(unique_keys, dtype=np.float32),
        )

    mean = split.sensors.mean(axis=0)
    std = split.sensors.std(axis=0)
    data_min = split.sensors.min(axis=0)
    data_max = split.sensors.max(axis=0)
    return CMAPSSNormalizer(
        subset=subset,
        mode=mode,
        use_condition_normalization=False,
        mean=mean.astype(np.float32),
        std=np.where(std < eps, 1.0, std).astype(np.float32),
        min=data_min.astype(np.float32),
        max=np.where((data_max - data_min) < eps, data_min + 1.0, data_max).astype(np.float32),
        condition_mean={},
        condition_std={},
        condition_min={},
        condition_max={},
        condition_key_to_id={},
        condition_centers=np.zeros((0, 3), dtype=np.float32),
    )


def build_windows(
    split: CMAPSSSplit,
    window_size: int,
    *,
    stride: int = 1,
    last_only: bool = False,
    pad_short: bool = True,
) -> WindowedData:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    windows: list[np.ndarray] = []
    targets: list[float] = []
    unit_ids: list[int] = []
    end_cycles: list[int] = []

    for unit_id in np.unique(split.unit_ids):
        mask = split.unit_ids == unit_id
        unit_sensors = split.sensors[mask]
        unit_rul = split.rul[mask]
        unit_cycles = split.cycles[mask]
        seq_len = unit_sensors.shape[0]

        if seq_len < window_size:
            if not pad_short:
                continue
            pad_rows = np.repeat(unit_sensors[:1], window_size - seq_len, axis=0)
            unit_window = np.concatenate([pad_rows, unit_sensors], axis=0)
            windows.append(unit_window.astype(np.float32))
            targets.append(float(unit_rul[-1]))
            unit_ids.append(int(unit_id))
            end_cycles.append(int(unit_cycles[-1]))
            continue

        start_indices: Iterable[int]
        if last_only:
            start_indices = [seq_len - window_size]
        else:
            start_indices = range(0, seq_len - window_size + 1, stride)

        for start in start_indices:
            end = start + window_size
            windows.append(unit_sensors[start:end].astype(np.float32))
            targets.append(float(unit_rul[end - 1]))
            unit_ids.append(int(unit_id))
            end_cycles.append(int(unit_cycles[end - 1]))

    if not windows:
        raise ValueError("No windows were generated")

    return WindowedData(
        windows=np.stack(windows).astype(np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        unit_ids=np.asarray(unit_ids, dtype=np.int32),
        end_cycles=np.asarray(end_cycles, dtype=np.int32),
    )


def build_monotonic_window_pairs(
    split: CMAPSSSplit,
    window_size: int,
    *,
    stride: int = 1,
    pair_gap: int = 1,
    pair_stride: int = 1,
) -> WindowPairData:
    if pair_gap <= 0:
        raise ValueError("pair_gap must be > 0")
    if pair_stride <= 0:
        raise ValueError("pair_stride must be > 0")

    windowed = build_windows(split, window_size, stride=stride, last_only=False)
    earlier_windows: list[np.ndarray] = []
    later_windows: list[np.ndarray] = []
    unit_ids: list[int] = []
    earlier_end_cycles: list[int] = []
    later_end_cycles: list[int] = []

    for unit_id in np.unique(windowed.unit_ids):
        unit_indices = np.where(windowed.unit_ids == unit_id)[0]
        if unit_indices.size <= pair_gap:
            continue
        ordered = unit_indices[np.argsort(windowed.end_cycles[unit_indices])]
        for offset in range(0, ordered.size - pair_gap, pair_stride):
            earlier_idx = ordered[offset]
            later_idx = ordered[offset + pair_gap]
            earlier_windows.append(windowed.windows[earlier_idx].astype(np.float32))
            later_windows.append(windowed.windows[later_idx].astype(np.float32))
            unit_ids.append(int(unit_id))
            earlier_end_cycles.append(int(windowed.end_cycles[earlier_idx]))
            later_end_cycles.append(int(windowed.end_cycles[later_idx]))

    if not earlier_windows:
        raise ValueError("No monotonic window pairs were generated")

    return WindowPairData(
        earlier_windows=np.stack(earlier_windows).astype(np.float32),
        later_windows=np.stack(later_windows).astype(np.float32),
        unit_ids=np.asarray(unit_ids, dtype=np.int32),
        earlier_end_cycles=np.asarray(earlier_end_cycles, dtype=np.int32),
        later_end_cycles=np.asarray(later_end_cycles, dtype=np.int32),
    )


def summarize_split(split: CMAPSSSplit) -> dict[str, int | float]:
    return {
        "rows": split.num_rows,
        "units": split.num_units,
        "feature_dim": split.feature_dim,
        "cycle_min": int(split.cycles.min()),
        "cycle_max": int(split.cycles.max()),
        "rul_min": float(split.rul.min()),
        "rul_max": float(split.rul.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-aligned C-MAPSS loader for MambAtt.")
    parser.add_argument("--root", default="/home/shelterpl/data/CMAPSS", help="CMAPSS root directory")
    parser.add_argument("--subset", default="FD001", help="One of FD001, FD002, FD003, FD004")
    parser.add_argument("--window-size", type=int, default=20, help="Paper-aligned window size")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride")
    parser.add_argument("--rul-clip", type=int, default=125, help="Piece-wise linear RUL cap")
    parser.add_argument("--save-dir", default=None, help="Optional directory for processed outputs")
    args = parser.parse_args()

    train_split = load_cmapss_split(args.root, args.subset, "train", rul_clip=args.rul_clip)
    test_split = load_cmapss_split(args.root, args.subset, "test", rul_clip=args.rul_clip)
    normalizer = fit_normalizer(train_split)
    train_scaled = normalizer.transform(train_split)
    test_scaled = normalizer.transform(test_split)

    train_windows = build_windows(train_scaled, args.window_size, stride=args.stride, last_only=False)
    test_windows = build_windows(test_scaled, args.window_size, stride=args.stride, last_only=True)

    summary = {
        "root": str(_resolve_root(args.root)),
        "subset": args.subset.upper(),
        "feature_names": FEATURE_NAMES,
        "pseudo_label_sensor_drop": PSEUDO_LABEL_SENSOR_DROP,
        "condition_normalization": bool(args.subset.upper() in CONDITION_NORMALIZED_SUBSETS),
        "train_split": summarize_split(train_split),
        "test_split": summarize_split(test_split),
        "train_windows_shape": train_windows.shape,
        "test_windows_shape": test_windows.shape,
        "train_targets_shape": tuple(int(dim) for dim in train_windows.targets.shape),
        "test_targets_shape": tuple(int(dim) for dim in test_windows.targets.shape),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.save_dir:
        save_dir = Path(args.save_dir).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_dir / f"{args.subset.upper()}_train_windows.npz",
            windows=train_windows.windows,
            targets=train_windows.targets,
            unit_ids=train_windows.unit_ids,
            end_cycles=train_windows.end_cycles,
            feature_names=np.asarray(FEATURE_NAMES),
        )
        np.savez_compressed(
            save_dir / f"{args.subset.upper()}_test_windows.npz",
            windows=test_windows.windows,
            targets=test_windows.targets,
            unit_ids=test_windows.unit_ids,
            end_cycles=test_windows.end_cycles,
            feature_names=np.asarray(FEATURE_NAMES),
        )
        with (save_dir / f"{args.subset.upper()}_normalizer.json").open("w", encoding="utf-8") as handle:
            json.dump(normalizer.to_dict(), handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
