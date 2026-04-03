from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cd_mambatt.cross_domain import CrossDomainTask, resolve_cross_domain_task
from cd_mambatt.data import CMAPSSSplit, CMAPSSWindowDataset, WindowedData, build_windows, fit_normalizer, load_cmapss_split
from cd_mambatt.metrics import mae, nasa_score, rmse
from cd_mambatt.models.fomln import FOMLNRegressor


FOMLN_SENSOR_IDS_1_BASED = (2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)
FOMLN_SENSOR_INDICES = tuple(sensor_id - 1 for sensor_id in FOMLN_SENSOR_IDS_1_BASED)


@dataclass(frozen=True)
class WindowPool:
    windows: torch.Tensor
    targets: torch.Tensor
    unit_ids: np.ndarray
    end_cycles: np.ndarray

    @classmethod
    def from_windowed(cls, data: WindowedData) -> "WindowPool":
        return cls(
            windows=torch.from_numpy(data.windows.astype(np.float32)),
            targets=torch.from_numpy(data.targets.astype(np.float32)),
            unit_ids=data.unit_ids.astype(np.int32),
            end_cycles=data.end_cycles.astype(np.int32),
        )

    @property
    def size(self) -> int:
        return int(self.targets.shape[0])

    def sample(self, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        effective_batch = max(1, min(int(batch_size), self.size))
        indices = torch.randint(0, self.size, (effective_batch,), generator=generator)
        return self.windows[indices], self.targets[indices]

def parse_seeds(seed_text: str | None, num_runs: int, base_seed: int) -> list[int]:
    if seed_text:
        return [int(token.strip()) for token in seed_text.split(",") if token.strip()]
    return [base_seed + offset for offset in range(num_runs)]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_fomln_task(args: argparse.Namespace) -> CrossDomainTask:
    if args.same_domain_subset:
        subset = str(args.same_domain_subset).upper()
        return CrossDomainTask(
            name=f"{subset}_SAME_DOMAIN",
            source_subset=subset,
            target_subset=subset,
            difficulty="same_domain",
            description="few-shot same-domain FOMLN reproduction task",
        )
    return resolve_cross_domain_task(args.task, source_subset=args.source_subset, target_subset=args.target_subset)


def build_loader(dataset: CMAPSSWindowDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def select_sensor_subset(split: CMAPSSSplit, sensor_indices: tuple[int, ...]) -> CMAPSSSplit:
    return CMAPSSSplit(
        subset=split.subset,
        split=split.split,
        unit_ids=split.unit_ids,
        cycles=split.cycles,
        operating_settings=split.operating_settings,
        sensors=split.sensors[:, sensor_indices].astype(np.float32),
        rul=split.rul,
        condition_ids=split.condition_ids,
    )


def subsample_windowed(data: WindowedData, max_windows: int | None, seed: int) -> WindowedData:
    if max_windows is None or data.windows.shape[0] <= max_windows:
        return data
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(data.windows.shape[0], size=max_windows, replace=False))
    return WindowedData(
        windows=data.windows[indices].astype(np.float32),
        targets=data.targets[indices].astype(np.float32),
        unit_ids=data.unit_ids[indices].astype(np.int32),
        end_cycles=data.end_cycles[indices].astype(np.int32),
    )


def build_fixed_meta_task_indices(num_samples: int, task_size: int, seed: int) -> list[np.ndarray]:
    if task_size <= 0:
        raise ValueError("task_size must be > 0")
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(num_samples)
    tasks: list[np.ndarray] = []
    for start in range(0, num_samples, task_size):
        indices = shuffled[start : start + task_size]
        if indices.size == 0:
            continue
        tasks.append(indices.astype(np.int64))
    return tasks


def build_fixed_meta_task_indices_by_units(unit_ids: np.ndarray, units_per_task: int, seed: int) -> list[np.ndarray]:
    if units_per_task <= 0:
        raise ValueError("units_per_task must be > 0")
    unique_units = np.unique(unit_ids)
    if unique_units.size == 0:
        raise ValueError("No unit ids available to build unit-level meta-tasks")
    rng = np.random.default_rng(seed)
    shuffled_units = rng.permutation(unique_units)
    tasks: list[np.ndarray] = []
    for start in range(0, shuffled_units.shape[0], units_per_task):
        task_units = shuffled_units[start : start + units_per_task]
        if task_units.size == 0:
            continue
        task_indices = np.flatnonzero(np.isin(unit_ids, task_units))
        if task_indices.size > 0:
            tasks.append(task_indices.astype(np.int64))
    return tasks


def slice_windowed(data: WindowedData, indices: np.ndarray) -> WindowedData:
    idx = np.asarray(indices, dtype=np.int64)
    return WindowedData(
        windows=data.windows[idx].astype(np.float32),
        targets=data.targets[idx].astype(np.float32),
        unit_ids=data.unit_ids[idx].astype(np.int32),
        end_cycles=data.end_cycles[idx].astype(np.int32),
    )


def weighted_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    raw_targets: torch.Tensor,
    *,
    smoothing_lambda: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    rounded_targets = torch.round(raw_targets).to(torch.int64)
    unique_values, counts = torch.unique(rounded_targets, return_counts=True)
    max_count = counts.max().to(dtype=predictions.dtype)
    frequencies = torch.zeros_like(targets, dtype=predictions.dtype)
    for value, count in zip(unique_values, counts):
        frequencies = torch.where(
            rounded_targets == value,
            torch.full_like(frequencies, float(count.item())),
            frequencies,
        )
    weights = torch.pow(max_count / (frequencies + smoothing_lambda), beta)
    return torch.mean(weights * torch.square(predictions - targets))


def predict(model: nn.Module, loader: DataLoader, device: torch.device, target_scale: float) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    tgts: list[np.ndarray] = []
    with torch.no_grad():
        for windows, targets in loader:
            windows = windows.to(device, non_blocking=True)
            outputs = (model(windows) * target_scale).detach().cpu().numpy()
            preds.append(outputs.astype(np.float32))
            tgts.append(targets.numpy().astype(np.float32))
    return np.concatenate(preds), np.concatenate(tgts)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, target_scale: float) -> dict[str, float]:
    predictions, targets = predict(model, loader, device, target_scale)
    return {
        "rmse": rmse(predictions, targets),
        "mae": mae(predictions, targets),
        "score": nasa_score(predictions, targets),
    }


def evaluate_pool(model: nn.Module, pool: WindowPool, device: torch.device, batch_size: int, target_scale: float) -> dict[str, float]:
    dataset = CMAPSSWindowDataset(
        WindowedData(
            windows=pool.windows.numpy(),
            targets=pool.targets.numpy(),
            unit_ids=pool.unit_ids,
            end_cycles=pool.end_cycles,
        )
    )
    loader = build_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return evaluate(model, loader, device, target_scale)


def clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def interpolate_towards_state(model: nn.Module, target_state: dict[str, torch.Tensor], outer_lr: float) -> None:
    current_state = model.state_dict()
    parameter_names = set(dict(model.named_parameters()).keys())
    updated_state: dict[str, torch.Tensor] = {}
    for name, current_tensor in current_state.items():
        target_tensor = target_state[name].to(device=current_tensor.device, dtype=current_tensor.dtype)
        if torch.is_floating_point(current_tensor):
            if name in parameter_names:
                updated_state[name] = current_tensor * (1.0 - outer_lr) + target_tensor * outer_lr
            else:
                updated_state[name] = target_tensor
        else:
            updated_state[name] = target_tensor
    model.load_state_dict(updated_state)


def build_inner_optimizer(
    parameters,
    *,
    optimizer_name: str,
    lr: float,
) -> torch.optim.Optimizer:
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")


def sample_target_support_split(
    target_windows: WindowedData,
    *,
    target_shots: int,
    target_val_samples: int,
    shot_level: str,
    seed: int,
) -> dict[str, object]:
    if target_shots <= 0:
        raise ValueError("target_shots must be > 0")
    if target_val_samples < 0:
        raise ValueError("target_val_samples must be >= 0")
    rng = np.random.default_rng(seed)
    if shot_level == "windows":
        total_needed = target_shots + target_val_samples
        num_windows = int(target_windows.windows.shape[0])
        if total_needed >= num_windows:
            raise ValueError(
                f"Need fewer than {num_windows} target train windows, but requested {total_needed} support+val windows",
            )
        shuffled = rng.permutation(num_windows)
        support_indices = np.sort(shuffled[:target_shots]).astype(np.int64)
        val_indices = np.sort(shuffled[target_shots : target_shots + target_val_samples]).astype(np.int64)
        remaining_indices = np.sort(shuffled[target_shots + target_val_samples :]).astype(np.int64)
        return {
            "seed": int(seed),
            "shot_level": shot_level,
            "target_shots": int(target_shots),
            "target_val_samples": int(target_val_samples),
            "support_indices": support_indices.tolist(),
            "support_unit_ids": target_windows.unit_ids[support_indices].tolist(),
            "support_end_cycles": target_windows.end_cycles[support_indices].tolist(),
            "val_indices": val_indices.tolist(),
            "val_unit_ids": target_windows.unit_ids[val_indices].tolist(),
            "val_end_cycles": target_windows.end_cycles[val_indices].tolist(),
            "remaining_count": int(remaining_indices.shape[0]),
        }

    if shot_level != "units":
        raise ValueError("shot_level must be 'windows' or 'units'")

    unique_units = np.unique(target_windows.unit_ids)
    total_needed = target_shots + target_val_samples
    if total_needed >= unique_units.size:
        raise ValueError(
            f"Need fewer than {unique_units.size} target train units, but requested {total_needed} support+val units",
        )
    shuffled_units = rng.permutation(unique_units)
    support_units = np.sort(shuffled_units[:target_shots]).astype(np.int32)
    val_units = np.sort(shuffled_units[target_shots : target_shots + target_val_samples]).astype(np.int32)
    remaining_units = np.sort(shuffled_units[target_shots + target_val_samples :]).astype(np.int32)
    support_indices = np.flatnonzero(np.isin(target_windows.unit_ids, support_units)).astype(np.int64)
    val_indices = np.flatnonzero(np.isin(target_windows.unit_ids, val_units)).astype(np.int64)
    return {
        "seed": int(seed),
        "shot_level": shot_level,
        "target_shots": int(target_shots),
        "target_val_samples": int(target_val_samples),
        "support_indices": support_indices.tolist(),
        "support_unit_ids": support_units.tolist(),
        "support_end_cycles": target_windows.end_cycles[support_indices].tolist(),
        "val_indices": val_indices.tolist(),
        "val_unit_ids": val_units.tolist(),
        "val_end_cycles": target_windows.end_cycles[val_indices].tolist(),
        "remaining_count": int(remaining_units.shape[0]),
    }


def build_model(args: argparse.Namespace) -> FOMLNRegressor:
    return FOMLNRegressor(
        input_dim=len(FOMLN_SENSOR_INDICES),
        window_size=args.window_size,
        conv_channels=args.conv_channels,
        conv_kernel_size=args.encoder_conv_kernel_size,
        conv_stride=args.encoder_conv_stride,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_blocks=args.num_conformer_blocks,
        d_k=args.attn_key_dim,
        d_v=args.attn_value_dim,
        ffm_expansion=args.ffm_expansion,
        attn_dropout=args.attn_dropout,
        ffm_dropout=args.ffm_dropout,
        conformer_conv_kernel_size=args.conformer_conv_kernel_size,
        conformer_conv_dropout=args.conformer_conv_dropout,
        head_dropout=args.head_dropout,
        use_final_norm=args.use_final_norm,
    )


def sample_batch_tensors(
    windows: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size is None or batch_size <= 0 or int(batch_size) >= int(targets.shape[0]):
        return windows, targets
    indices = torch.randint(0, int(targets.shape[0]), (int(batch_size),))
    return windows[indices], targets[indices]


def run_inner_adaptation(
    model: nn.Module,
    task_windows: torch.Tensor,
    task_targets: torch.Tensor,
    *,
    device: torch.device,
    inner_steps: int,
    inner_batch_size: int | None,
    lr: float,
    optimizer_name: str,
    grad_clip_norm: float | None,
    smoothing_lambda: float,
    beta: float,
    target_scale: float,
) -> tuple[nn.Module, float]:
    fast_model = copy.deepcopy(model).to(device)
    fast_model.train(True)
    optimizer = build_inner_optimizer(fast_model.parameters(), optimizer_name=optimizer_name, lr=lr)
    last_loss = 0.0
    for _ in range(inner_steps):
        batch_windows, batch_targets = sample_batch_tensors(task_windows, task_targets, batch_size=inner_batch_size)
        batch_windows = batch_windows.to(device, non_blocking=True)
        raw_targets = batch_targets.to(device, non_blocking=True)
        scaled_targets = raw_targets / target_scale
        predictions = fast_model(batch_windows)
        loss = weighted_mse_loss(
            predictions,
            scaled_targets,
            raw_targets,
            smoothing_lambda=smoothing_lambda,
            beta=beta,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(fast_model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    return fast_model, last_loss


def meta_train(
    model: nn.Module,
    source_pool: WindowPool,
    *,
    device: torch.device,
    seed: int,
    outer_loops: int,
    meta_batch_tasks: int,
    inner_steps: int,
    source_task_size: int,
    inner_batch_size: int | None,
    inner_lr: float,
    outer_lr: float,
    inner_optimizer: str,
    grad_clip_norm: float | None,
    smoothing_lambda: float,
    beta: float,
    target_scale: float,
    source_task_level: str,
) -> list[dict[str, float | int]]:
    if source_task_level == "windows":
        meta_task_indices = build_fixed_meta_task_indices(source_pool.size, source_task_size, seed)
    elif source_task_level == "units":
        meta_task_indices = build_fixed_meta_task_indices_by_units(source_pool.unit_ids, source_task_size, seed)
    else:
        raise ValueError("source_task_level must be 'windows' or 'units'")
    task_rng = np.random.default_rng(seed + 1)
    history: list[dict[str, float | int]] = []
    for outer_step in range(1, outer_loops + 1):
        inner_losses: list[float] = []
        replace = len(meta_task_indices) < meta_batch_tasks
        selected_task_ids = task_rng.choice(len(meta_task_indices), size=meta_batch_tasks, replace=replace)
        for task_id in np.asarray(selected_task_ids).tolist():
            task_indices = meta_task_indices[int(task_id)]
            task_windows = source_pool.windows[task_indices]
            task_targets = source_pool.targets[task_indices]
            fast_model, last_loss = run_inner_adaptation(
                model,
                task_windows,
                task_targets,
                device=device,
                inner_steps=inner_steps,
                inner_batch_size=inner_batch_size,
                lr=inner_lr,
                optimizer_name=inner_optimizer,
                grad_clip_norm=grad_clip_norm,
                smoothing_lambda=smoothing_lambda,
                beta=beta,
                target_scale=target_scale,
            )
            interpolate_towards_state(model, clone_state_dict(fast_model), outer_lr=outer_lr)
            inner_losses.append(last_loss)
            del fast_model
        record = {
            "outer_step": outer_step,
            "mean_inner_weighted_mse": float(np.mean(inner_losses)),
            "meta_batch_tasks": meta_batch_tasks,
            "source_meta_task_count": len(meta_task_indices),
            "source_task_level": source_task_level,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
    return history


def adapt_to_target(
    meta_model: nn.Module,
    support_pool: WindowPool,
    *,
    val_loader: DataLoader | None,
    device: torch.device,
    seed: int,
    adapt_steps: int,
    adapt_batch_size: int | None,
    adapt_lr: float,
    adapt_optimizer: str,
    grad_clip_norm: float | None,
    smoothing_lambda: float,
    beta: float,
    target_scale: float,
) -> tuple[nn.Module, list[dict[str, float | int]], int, float | None]:
    model = copy.deepcopy(meta_model).to(device)
    model.train(True)
    optimizer = build_inner_optimizer(model.parameters(), optimizer_name=adapt_optimizer, lr=adapt_lr)

    best_state = clone_state_dict(model)
    best_step = 0
    best_val_rmse = None if val_loader is None else float("inf")
    best_support_rmse = float("inf")
    history: list[dict[str, float | int]] = []

    for adapt_step in range(1, adapt_steps + 1):
        model.train(True)
        batch_windows, batch_targets = sample_batch_tensors(
            support_pool.windows,
            support_pool.targets,
            batch_size=adapt_batch_size,
        )
        batch_windows = batch_windows.to(device, non_blocking=True)
        support_raw_targets = batch_targets.to(device, non_blocking=True)
        support_scaled_targets = support_raw_targets / target_scale
        predictions = model(batch_windows)
        loss = weighted_mse_loss(
            predictions,
            support_scaled_targets,
            support_raw_targets,
            smoothing_lambda=smoothing_lambda,
            beta=beta,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        support_metrics = evaluate_pool(
            model,
            support_pool,
            device,
            batch_size=max(1, support_pool.size),
            target_scale=target_scale,
        )
        record: dict[str, float | int] = {
            "adapt_step": adapt_step,
            "support_weighted_mse": float(loss.detach().cpu()),
            "support_rmse": support_metrics["rmse"],
        }
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, target_scale)
            record["val_rmse"] = val_metrics["rmse"]
            record["val_score"] = val_metrics["score"]
            if val_metrics["rmse"] < float(best_val_rmse):
                best_val_rmse = float(val_metrics["rmse"])
                best_state = clone_state_dict(model)
                best_step = adapt_step
        elif support_metrics["rmse"] < best_support_rmse:
            best_support_rmse = support_metrics["rmse"]
            best_state = clone_state_dict(model)
            best_step = adapt_step
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

    model.load_state_dict(best_state)
    return model, history, best_step, best_val_rmse


def prepare_domain_windows(
    root: str | Path,
    subset: str,
    *,
    window_size: int,
    stride: int,
    rul_clip: int,
    last_only_test: bool,
    max_train_windows: int | None,
    max_test_windows: int | None,
    seed: int,
) -> tuple[WindowedData, WindowedData]:
    train_raw = load_cmapss_split(root, subset, "train", rul_clip=rul_clip)
    test_raw = load_cmapss_split(root, subset, "test", rul_clip=rul_clip)
    normalizer = fit_normalizer(train_raw)
    train_scaled = select_sensor_subset(normalizer.transform(train_raw), FOMLN_SENSOR_INDICES)
    test_scaled = select_sensor_subset(normalizer.transform(test_raw), FOMLN_SENSOR_INDICES)
    train_windows = build_windows(train_scaled, window_size, stride=stride, last_only=False)
    test_windows = build_windows(test_scaled, window_size, stride=stride, last_only=last_only_test)
    train_windows = subsample_windowed(train_windows, max_train_windows, seed=seed)
    test_windows = subsample_windowed(test_windows, max_test_windows, seed=seed + 1)
    return train_windows, test_windows


def run_one_seed(
    args: argparse.Namespace,
    *,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> dict[str, object]:
    set_seed(seed)
    task = resolve_fomln_task(args)

    source_train_windows, _ = prepare_domain_windows(
        args.root,
        task.source_subset,
        window_size=args.window_size,
        stride=args.stride,
        rul_clip=args.rul_clip,
        last_only_test=True,
        max_train_windows=args.max_source_train_windows,
        max_test_windows=None,
        seed=seed,
    )
    target_train_windows, target_test_windows = prepare_domain_windows(
        args.root,
        task.target_subset,
        window_size=args.window_size,
        stride=args.stride,
        rul_clip=args.rul_clip,
        last_only_test=True,
        max_train_windows=args.max_target_train_windows,
        max_test_windows=args.max_target_test_windows,
        seed=seed + 17,
    )

    source_pool = WindowPool.from_windowed(source_train_windows)
    if args.source_task_level == "windows":
        source_meta_task_count = len(build_fixed_meta_task_indices(source_pool.size, args.source_task_size, seed))
    elif args.source_task_level == "units":
        source_meta_task_count = len(build_fixed_meta_task_indices_by_units(source_pool.unit_ids, args.source_task_size, seed))
    else:
        raise ValueError("source_task_level must be 'windows' or 'units'")
    target_split_payload = sample_target_support_split(
        target_train_windows,
        target_shots=args.target_shots,
        target_val_samples=args.target_val_samples,
        shot_level=args.shot_level,
        seed=seed,
    )
    support_indices = np.asarray(target_split_payload["support_indices"], dtype=np.int64)
    val_indices = np.asarray(target_split_payload["val_indices"], dtype=np.int64)
    support_pool = WindowPool.from_windowed(slice_windowed(target_train_windows, support_indices))
    val_loader = None
    if val_indices.size > 0:
        val_dataset = CMAPSSWindowDataset(slice_windowed(target_train_windows, val_indices))
        val_loader = build_loader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    target_test_loader = build_loader(
        CMAPSSWindowDataset(target_test_windows),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(args).to(device)
    meta_history = meta_train(
        model,
        source_pool,
        device=device,
        seed=seed,
        outer_loops=args.outer_loops,
        meta_batch_tasks=args.meta_batch_tasks,
        inner_steps=args.inner_steps,
        source_task_size=args.source_task_size,
        inner_batch_size=None if args.inner_batch_size <= 0 else int(args.inner_batch_size),
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        inner_optimizer=args.inner_optimizer,
        grad_clip_norm=None if args.grad_clip_norm <= 0 else float(args.grad_clip_norm),
        smoothing_lambda=args.loss_smoothing_lambda,
        beta=args.loss_beta,
        target_scale=float(args.target_scale),
        source_task_level=args.source_task_level,
    )

    direct_metrics = evaluate(model, target_test_loader, device, float(args.target_scale))
    adapted_model, adapt_history, best_adapt_step, best_val_rmse = adapt_to_target(
        model,
        support_pool,
        val_loader=val_loader,
        device=device,
        seed=seed,
        adapt_steps=args.adapt_steps,
        adapt_batch_size=None if args.adapt_batch_size <= 0 else int(args.adapt_batch_size),
        adapt_lr=float(args.inner_lr if args.adapt_lr is None else args.adapt_lr),
        adapt_optimizer=str(args.inner_optimizer if args.adapt_optimizer is None else args.adapt_optimizer),
        grad_clip_norm=None if args.grad_clip_norm <= 0 else float(args.grad_clip_norm),
        smoothing_lambda=args.loss_smoothing_lambda,
        beta=args.loss_beta,
        target_scale=float(args.target_scale),
    )
    adapted_metrics = evaluate(adapted_model, target_test_loader, device, float(args.target_scale))

    run_dir = output_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "target_support_split.json").open("w", encoding="utf-8") as handle:
        json.dump(target_split_payload, handle, indent=2, ensure_ascii=False)
    with (run_dir / "meta_history.json").open("w", encoding="utf-8") as handle:
        json.dump(meta_history, handle, indent=2, ensure_ascii=False)
    with (run_dir / "adapt_history.json").open("w", encoding="utf-8") as handle:
        json.dump(adapt_history, handle, indent=2, ensure_ascii=False)
    torch.save(
        {
            "model_state_dict": adapted_model.state_dict(),
            "args": vars(args),
            "seed": seed,
            "task": task.name,
        },
        run_dir / "adapted_model.pt",
    )

    result = {
        "implementation_stage": "paper_alignment_in_progress",
        "task": task.name,
        "source_subset": task.source_subset,
        "target_subset": task.target_subset,
        "seed": seed,
        "selected_sensors_1_based": list(FOMLN_SENSOR_IDS_1_BASED),
        "window_size": args.window_size,
        "source_train_windows": int(source_pool.size),
        "source_task_level": str(args.source_task_level),
        "source_meta_task_size": int(args.source_task_size),
        "source_meta_task_count": int(source_meta_task_count),
        "shot_level": str(args.shot_level),
        "target_train_windows": int(target_train_windows.windows.shape[0]),
        "target_support_samples": int(support_pool.size),
        "target_val_samples": int(val_indices.size),
        "target_test_windows": int(target_test_windows.windows.shape[0]),
        "direct_target_rmse": direct_metrics["rmse"],
        "direct_target_score": direct_metrics["score"],
        "adapted_target_rmse": adapted_metrics["rmse"],
        "adapted_target_score": adapted_metrics["score"],
        "adapted_target_mae": adapted_metrics["mae"],
        "best_adapt_step": int(best_adapt_step),
        "best_val_rmse": best_val_rmse,
        "run_dir": str(run_dir),
    }
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)
    with (run_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="FOMLN reproduction runner (paper-alignment in progress).")
    parser.add_argument("--root", default="/home/shelterpl/data/CMAPSS", help="CMAPSS root directory")
    parser.add_argument("--task", default="FD001_TO_FD003", help="Cross-domain task name")
    parser.add_argument("--source-subset", default=None, help="Custom source subset when --task is omitted")
    parser.add_argument("--target-subset", default=None, help="Custom target subset when --task is omitted")
    parser.add_argument("--same-domain-subset", default=None, help="Optional same-domain subset (e.g. FD001) for aligning the paper's few-shot same-domain section")
    parser.add_argument("--window-size", type=int, default=30, help="Paper-reported time window length")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride")
    parser.add_argument("--rul-clip", type=int, default=125, help="Piece-wise linear RUL cap")
    parser.add_argument("--target-shots", type=int, default=15, help="Target-domain support samples for adaptation")
    parser.add_argument("--target-val-samples", type=int, default=0, help="Optional extra target train samples reserved for model selection")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of seeds to run")
    parser.add_argument("--base-seed", type=int, default=42, help="Starting seed when --seeds is omitted")
    parser.add_argument("--seeds", default=None, help="Optional comma-separated explicit seed list")
    parser.add_argument("--outer-loops", type=int, default=50, help="Paper-reported outer-loop count")
    parser.add_argument("--meta-batch-tasks", type=int, default=4, help="Number of sampled source tasks per outer loop")
    parser.add_argument("--source-task-size", type=int, default=15, help="Source meta-task sample count; paper training is described under the same 15-shot regime")
    parser.add_argument("--source-task-level", choices=("windows", "units"), default="windows", help="How source meta-tasks are formed during reproduction development")
    parser.add_argument("--inner-steps", type=int, default=10, help="Paper-reported inner-loop count")
    parser.add_argument("--adapt-steps", type=int, default=10, help="Target adaptation steps")
    parser.add_argument("--inner-batch-size", type=int, default=256, help="Mini-batch size used inside each source-task inner loop; keeps weighted loss batch-based and reduces peak GPU memory")
    parser.add_argument("--adapt-batch-size", type=int, default=256, help="Mini-batch size used during target support adaptation; reduces peak GPU memory")
    parser.add_argument("--inner-lr", type=float, default=1e-3, help="Paper-reported inner-loop learning rate")
    parser.add_argument("--outer-lr", type=float, default=0.1, help="Paper-reported outer-loop interpolation rate")
    parser.add_argument("--inner-optimizer", choices=("adam", "sgd"), default="adam", help="Paper OCR indicates the inner-loop optimizer is e.g. Adam; use Adam by default for both source-task adaptation and target support adaptation")
    parser.add_argument("--adapt-lr", type=float, default=None, help="Optional target adaptation learning rate override; defaults to --inner-lr")
    parser.add_argument("--adapt-optimizer", choices=("adam", "sgd"), default=None, help="Optional target adaptation optimizer override; defaults to --inner-optimizer")
    parser.add_argument("--eval-batch-size", type=int, default=256, help="Evaluation batch size")
    parser.add_argument("--target-scale", type=float, default=1.0, help="Optional target divisor used during training; predictions are rescaled for evaluation")
    parser.add_argument("--shot-level", choices=("windows", "units"), default="windows", help="How the target 15-shot support set is sampled during reproduction development")
    parser.add_argument("--loss-smoothing-lambda", type=float, default=1.0, help="Paper-reported weighted loss smoothing λ")
    parser.add_argument("--loss-beta", type=float, default=1.0, help="Paper-reported weighted loss exponent β")
    parser.add_argument("--grad-clip-norm", type=float, default=0.0, help="Optional gradient clipping norm")
    parser.add_argument("--d-model", type=int, default=512, help="Working paper-alignment default inferred from Table 1: 8 heads x 64-dim attention")
    parser.add_argument("--num-heads", type=int, default=8, help="Paper-reported attention head count")
    parser.add_argument("--attn-key-dim", type=int, default=64, help="Paper-reported attention key/query dimension per head")
    parser.add_argument("--attn-value-dim", type=int, default=64, help="Paper-reported attention value dimension per head")
    parser.add_argument("--num-conformer-blocks", type=int, default=1, help="Paper-reported number of Conformer blocks")
    parser.add_argument("--conv-channels", type=int, default=10, help="Paper-reported number of encoder convolution kernels")
    parser.add_argument("--encoder-conv-kernel-size", type=int, default=10, help="Paper-reported encoder convolution kernel size")
    parser.add_argument("--encoder-conv-stride", type=int, default=1, help="Paper-reported encoder convolution stride")
    parser.add_argument("--ffm-expansion", type=int, default=4, help="FFM hidden expansion ratio; the paper does not explicitly report this value")
    parser.add_argument("--attn-dropout", type=float, default=0.2, help="Paper-reported attention dropout")
    parser.add_argument("--ffm-dropout", type=float, default=0.2, help="Paper-reported FFM dropout")
    parser.add_argument("--conformer-conv-kernel-size", type=int, default=31, help="Paper-reported depthwise convolution kernel size")
    parser.add_argument("--conformer-conv-dropout", type=float, default=0.2, help="Paper-reported convolution-module dropout")
    parser.add_argument("--head-dropout", type=float, default=0.0, help="Optional dropout before the regression head")
    parser.add_argument("--use-final-norm", action="store_true", help="Optional final LayerNorm after each Conformer block; disabled by default because the paper text does not state it explicitly")
    parser.add_argument("--max-source-train-windows", type=int, default=None, help="Optional cap for source train windows (smoke tests)")
    parser.add_argument("--max-target-train-windows", type=int, default=None, help="Optional cap for target train windows (smoke tests)")
    parser.add_argument("--max-target-test-windows", type=int, default=None, help="Optional cap for target test windows (smoke tests)")
    parser.add_argument("--device", default="cuda", help="cuda recommended; experiments in this project should run on GPU")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--output-dir", default="/home/shelterpl/cd_mambatt/runs/fomln_baseline", help="Run output directory")
    args = parser.parse_args()

    device = infer_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("train_fomln_baseline.py is configured to require CUDA for all experiments in this project.")
    output_dir = Path(args.output_dir).expanduser().resolve() / (args.task.upper() if args.task else "custom_task")
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_seeds(args.seeds, args.num_runs, args.base_seed)
    start_time = time.time()
    run_results: list[dict[str, object]] = []
    for seed in seeds:
        run_results.append(run_one_seed(args, seed=seed, device=device, output_dir=output_dir))
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    summary = {
        "implementation_stage": "paper_alignment_in_progress",
        "task": run_results[0]["task"],
        "source_subset": run_results[0]["source_subset"],
        "target_subset": run_results[0]["target_subset"],
        "seeds": seeds,
        "num_runs": len(run_results),
        "window_size": args.window_size,
        "target_shots": args.target_shots,
        "source_task_size": args.source_task_size,
        "source_task_level": args.source_task_level,
        "shot_level": args.shot_level,
        "target_val_samples": args.target_val_samples,
        "mean_direct_target_rmse": float(np.mean([float(item["direct_target_rmse"]) for item in run_results])),
        "std_direct_target_rmse": float(np.std([float(item["direct_target_rmse"]) for item in run_results])),
        "mean_adapted_target_rmse": float(np.mean([float(item["adapted_target_rmse"]) for item in run_results])),
        "std_adapted_target_rmse": float(np.std([float(item["adapted_target_rmse"]) for item in run_results])),
        "best_seed": min(run_results, key=lambda item: float(item["adapted_target_rmse"]))["seed"],
        "best_adapted_target_rmse": min(float(item["adapted_target_rmse"]) for item in run_results),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "device": str(device),
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"runs": run_results, "summary": summary}, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
