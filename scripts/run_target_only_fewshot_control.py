from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cd_mambatt.data import load_cmapss_split
from cd_mambatt.cross_domain import resolve_cross_domain_task
from train_cd_mambatt_v3 import build_lr_scheduler, uses_domain_conditioning
from train_cross_domain_baseline import build_target_finetune_data
from train_supervised import build_model, evaluate, infer_device, parse_seeds, set_seed


def load_record(run_root: Path, seed: int) -> dict[str, object]:
    return json.loads((run_root / f"seed_{seed}" / "result.json").read_text(encoding="utf-8"))


def args_from_record(record: dict[str, object]) -> Namespace:
    cd_stage = record.get("cd_stage", {})
    task = resolve_cross_domain_task(str(record["task"]))
    target_subset = str(record.get("target_subset", task.target_subset))
    return Namespace(
        root="/home/shelterpl/data/CMAPSS",
        task=str(record["task"]),
        target_subset=target_subset,
        window_size=20,
        stride=1,
        rul_clip=125,
        target_scale=1.0,
        grad_clip_norm=0.0,
        target_shots=5,
        target_val_units=10,
        few_shot_seed=int(record["few_shot_seed"]),
        target_partition_path=str(record["target_partition_path"]),
        batch_size=64,
        target_epochs=int(cd_stage.get("target_epochs", 20)),
        target_lr=float(cd_stage.get("target_lr", 5e-4)),
        target_lr_scheduler=str(cd_stage.get("target_lr_scheduler", "none")),
        target_lr_min=float(cd_stage.get("target_lr_min", 1e-5)),
        target_weight_decay=float(cd_stage.get("target_weight_decay", 0.0)),
        d_model=None,
        d_state=16,
        d_conv=8,
        expand=2,
        num_mamba_layers=1,
        num_transformer_layers=3,
        num_heads=7,
        dropout=0.5,
        dim_feedforward=84,
        transformer_impl="custom",
        transformer_norm_mode=str(cd_stage.get("transformer_norm_mode", "pre")),
        transformer_inner_dropout=0.0,
        mamba_block_mode=str(cd_stage.get("mamba_block_mode", "dd_spd")),
        spd_gate_init_bias=-2.0,
        spd_scan_mode="mixed",
        spd_gate_mode="token",
        spd_gate_scheme="shared",
        spd_predictor_mode=str(cd_stage.get("spd_predictor_mode", "shared_head")),
        domain_conditioned_gate=bool(cd_stage.get("domain_conditioned_gate", False)),
        frontend_adapter_mode=str(cd_stage.get("frontend_adapter_mode", "none")),
        transformer_domain_adapter_mode=str(cd_stage.get("transformer_domain_adapter_mode", "none")),
        target_val_all_windows=True,
        device="auto",
        num_workers=0,
        max_target_train_batches=None,
    )


def build_target_data(args: Namespace):
    target_train_full = load_cmapss_split(args.root, args.target_subset, "train", rul_clip=args.rul_clip)
    target_test_raw = load_cmapss_split(args.root, args.target_subset, "test", rul_clip=args.rul_clip)
    with open(args.target_partition_path, "r", encoding="utf-8") as handle:
        target_partition = json.load(handle)
    target_loaders, target_meta = build_target_finetune_data(args, target_train_full, target_test_raw, target_partition)
    return target_loaders, target_meta


def run_target_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
    target_scale: float,
    grad_clip_norm: float | None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_items = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    use_domain_label = uses_domain_conditioning(model)

    with context:
        for batch_idx, (windows, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            windows = windows.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True) / target_scale
            domain_label = None
            if use_domain_label:
                domain_label = torch.ones((windows.shape[0],), dtype=torch.long, device=device)
            predictions = model(windows, domain_label=domain_label)
            loss = criterion(predictions, targets)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

            batch_size = int(windows.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_items += batch_size

    if total_items == 0:
        raise ValueError("No batches were processed")
    return total_loss / total_items


def train_one_seed(
    args: Namespace,
    seed: int,
    record: dict[str, object],
    device: torch.device,
    run_dir: Path,
    *,
    init_from_source_checkpoint: bool,
) -> dict[str, object]:
    set_seed(seed)
    target_loaders, target_meta = build_target_data(args)
    model = build_model(args, int(target_meta["train_windows_shape"][2])).to(device)
    source_checkpoint_path = None
    if init_from_source_checkpoint:
        source_checkpoint_path = str(record["source_stage"]["checkpoint"])
        checkpoint = torch.load(source_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.target_lr, weight_decay=args.target_weight_decay)
    scheduler = build_lr_scheduler(
        optimizer,
        scheduler_name=str(args.target_lr_scheduler),
        total_epochs=int(args.target_epochs),
        min_lr=float(args.target_lr_min),
    )
    criterion = nn.MSELoss()
    target_scale = float(args.target_scale)
    grad_clip_norm = None if float(args.grad_clip_norm) <= 0 else float(args.grad_clip_norm)

    best_val_rmse = float("inf")
    best_epoch = -1
    history: list[dict[str, object]] = []
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pt"
    eval_domain = 1 if uses_domain_conditioning(model) else None

    for epoch in range(1, int(args.target_epochs) + 1):
        train_mse = run_target_epoch(
            model,
            target_loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            max_batches=args.max_target_train_batches,
            target_scale=target_scale,
            grad_clip_norm=grad_clip_norm,
        )
        val_metrics = evaluate(
            model,
            target_loaders["val"],
            device,
            target_scale,
            domain_label_value=eval_domain,
        )
        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_record = {
            "seed": seed,
            "epoch": epoch,
            "train_mse": train_mse,
            "train_rmse": float(np.sqrt(train_mse) * target_scale),
            "val_rmse": float(val_metrics["rmse"]),
            "val_mae": float(val_metrics["mae"]),
            "val_score": float(val_metrics["score"]),
            "lr": current_lr,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False))

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = float(val_metrics["rmse"])
            best_epoch = int(epoch)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_val_rmse": best_val_rmse,
                    "history": history,
                    "record_task": str(record["task"]),
                    "target_partition_path": str(args.target_partition_path),
                },
                ckpt_path,
            )

        if scheduler is not None:
            scheduler.step()

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(
        model,
        target_loaders["test"],
        device,
        target_scale,
        domain_label_value=eval_domain,
    )
    matched_transfer = record.get("cd_stage", record.get("target_finetune", {}))
    matched_transfer_rmse = matched_transfer.get("test_rmse")
    if matched_transfer_rmse is None:
        raise KeyError("Baseline record is missing both cd_stage.test_rmse and target_finetune.test_rmse")
    result = {
        "seed": seed,
        "task": str(record["task"]),
        "target_subset": str(args.target_subset),
        "target_partition_path": str(args.target_partition_path),
        "few_shot_seed": int(args.few_shot_seed),
        "best_epoch": int(best_epoch),
        "best_val_rmse": float(best_val_rmse),
        "test_rmse": float(test_metrics["rmse"]),
        "test_mae": float(test_metrics["mae"]),
        "test_score": float(test_metrics["score"]),
        "checkpoint": str(ckpt_path),
        "train_windows_shape": target_meta["train_windows_shape"],
        "val_windows_shape": target_meta["val_windows_shape"],
        "test_windows_shape": target_meta["test_windows_shape"],
        "labeled_unit_count": int(target_meta["labeled_unit_count"]),
        "val_unit_count": int(target_meta["val_unit_count"]),
        "unlabeled_unit_count": int(target_meta["unlabeled_unit_count"]),
        "matched_baseline_target_direct_rmse": float(record["target_direct"]["rmse"]),
        "matched_baseline_cd_test_rmse": float(matched_transfer_rmse),
        "matched_baseline_kind": "cd_stage" if "cd_stage" in record else "target_finetune",
        "control_variant": "source_initialized_finetune" if init_from_source_checkpoint else "target_only_scratch",
        "init_from_source_checkpoint": bool(init_from_source_checkpoint),
        "source_checkpoint": source_checkpoint_path,
        "train_from_scratch": not bool(init_from_source_checkpoint),
        "history": history,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def summarize_results(run_root: Path, experiment_name: str, run_results: list[dict[str, object]], start_time: float) -> dict[str, object]:
    test_rmses = [float(item["test_rmse"]) for item in run_results]
    baseline_cd_rmses = [float(item["matched_baseline_cd_test_rmse"]) for item in run_results]
    baseline_direct_rmses = [float(item["matched_baseline_target_direct_rmse"]) for item in run_results]
    best_run = min(run_results, key=lambda item: float(item["best_val_rmse"]))
    return {
        "experiment_name": experiment_name,
        "run_root": str(run_root),
        "task": str(run_results[0]["task"]),
        "target_subset": str(run_results[0]["target_subset"]),
        "seeds": [int(item["seed"]) for item in run_results],
        "num_runs": len(run_results),
        "init_from_source_checkpoint": bool(run_results[0]["init_from_source_checkpoint"]),
        "control_variant": "source_initialized_finetune" if run_results[0]["init_from_source_checkpoint"] else "target_only_scratch",
        "mean_control_test_rmse": float(np.mean(test_rmses)),
        "std_control_test_rmse": float(np.std(test_rmses)),
        "mean_target_only_test_rmse": float(np.mean(test_rmses)),
        "std_target_only_test_rmse": float(np.std(test_rmses)),
        "best_target_only_seed": int(best_run["seed"]),
        "best_target_only_val_rmse": float(best_run["best_val_rmse"]),
        "best_target_only_test_rmse": float(best_run["test_rmse"]),
        "mean_matched_baseline_cd_rmse": float(np.mean(baseline_cd_rmses)),
        "mean_matched_baseline_direct_rmse": float(np.mean(baseline_direct_rmses)),
        "delta_target_only_minus_matched_cd": float(np.mean(test_rmses) - np.mean(baseline_cd_rmses)),
        "delta_target_only_minus_matched_direct": float(np.mean(test_rmses) - np.mean(baseline_direct_rmses)),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a target-only few-shot control from scratch on the exact target partitions used by an existing cross-domain run.")
    parser.add_argument("--baseline-run-root", required=True, help="Existing cross-domain run root containing per-seed result.json files")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--seeds", required=True, help="Comma-separated seeds to replay with the matched target partitions")
    parser.add_argument("--output-root", default="/home/shelterpl/cd_mambatt/runs/target_only_fewshot_controls")
    parser.add_argument("--target-epochs", type=int, default=None)
    parser.add_argument("--target-lr", type=float, default=None)
    parser.add_argument("--target-lr-scheduler", choices=("none", "cosine"), default=None)
    parser.add_argument("--target-lr-min", type=float, default=None)
    parser.add_argument("--target-weight-decay", type=float, default=None)
    parser.add_argument("--transformer-norm-mode", choices=("pre", "post"), default=None)
    parser.add_argument("--mamba-block-mode", choices=("bare", "prenorm_residual", "dd_spd"), default=None)
    parser.add_argument("--init-from-source-checkpoint", action="store_true")
    parser.add_argument("--device", default="auto")
    args_cli = parser.parse_args()

    baseline_run_root = Path(args_cli.baseline_run_root).expanduser().resolve()
    seeds = parse_seeds(args_cli.seeds, num_runs=0, base_seed=0)
    device = infer_device(args_cli.device)
    if device.type != "cuda":
        raise RuntimeError("This control expects CUDA.")

    run_results: list[dict[str, object]] = []
    start_time = time.time()

    for seed in seeds:
        record = load_record(baseline_run_root, seed)
        args = args_from_record(record)
        if args_cli.target_epochs is not None:
            args.target_epochs = int(args_cli.target_epochs)
        if args_cli.target_lr is not None:
            args.target_lr = float(args_cli.target_lr)
        if args_cli.target_lr_scheduler is not None:
            args.target_lr_scheduler = str(args_cli.target_lr_scheduler)
        if args_cli.target_lr_min is not None:
            args.target_lr_min = float(args_cli.target_lr_min)
        if args_cli.target_weight_decay is not None:
            args.target_weight_decay = float(args_cli.target_weight_decay)
        if args_cli.transformer_norm_mode is not None:
            args.transformer_norm_mode = str(args_cli.transformer_norm_mode)
        if args_cli.mamba_block_mode is not None:
            args.mamba_block_mode = str(args_cli.mamba_block_mode)

        run_dir = Path(args_cli.output_root).expanduser().resolve() / args_cli.experiment_name / str(record["task"]) / f"seed_{seed}"
        result = train_one_seed(
            args,
            seed,
            record,
            device,
            run_dir,
            init_from_source_checkpoint=bool(args_cli.init_from_source_checkpoint),
        )
        print(json.dumps(result, ensure_ascii=False))
        run_results.append(result)

    summary = summarize_results(baseline_run_root, args_cli.experiment_name, run_results, start_time)
    summary_path = Path(args_cli.output_root).expanduser().resolve() / args_cli.experiment_name / str(run_results[0]["task"]) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"runs": run_results, "summary": summary}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
