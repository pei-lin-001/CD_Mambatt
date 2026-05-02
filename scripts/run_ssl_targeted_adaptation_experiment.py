from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cd_mambatt.cross_domain import resolve_cross_domain_task
from cd_mambatt.data import WindowedData, build_windows, fit_normalizer, load_cmapss_split, select_units
from cd_mambatt.self_supervised import (
    IndexedWindowPairDataset,
    IndexedWindowTripletDataset,
    WindowPseudoLabelDataset,
    assign_window_sensor_pseudo_labels,
    build_consecutive_pair_indices,
    build_temporal_triplet_indices,
    fit_sensor_pseudo_label_statistics,
    ntuplet_loss,
)
from train_cd_mambatt_v1 import fit_source_stage
from train_cd_mambatt_v3 import build_target_monotonic_loader, fit_cd_pseudo_stage, uses_domain_conditioning
from train_cross_domain_baseline import build_source_stage_data
from train_self_supervised import (
    apply_ssl_preset,
    build_ssl_loader,
    build_ssl_pretrainer,
    endless_loader,
    load_encoder_weights,
)
from train_supervised import build_model, evaluate, infer_device, set_seed


def load_record(run_root: Path, seed: int) -> dict[str, object]:
    return json.loads((run_root / f"seed_{seed}" / "result.json").read_text(encoding="utf-8"))


def args_from_record(record: dict[str, object]) -> Namespace:
    cd_stage = record["cd_stage"]
    inv_alignment_mode = str(cd_stage.get("inv_alignment_mode", "mmd") or "mmd")
    lambda_inv_mmd = float(
        cd_stage.get(
            "lambda_inv_mmd",
            cd_stage.get("lambda_domain_adv", 0.1 if inv_alignment_mode == "mmd" else 0.0),
        )
    )
    return Namespace(
        root="/home/shelterpl/data/CMAPSS",
        task=str(record["task"]),
        source_subset=str(record["source_subset"]),
        target_subset=str(record["target_subset"]),
        window_size=20,
        stride=1,
        rul_clip=125,
        target_scale=1.0,
        grad_clip_norm=0.0,
        source_train_ratio=0.8,
        source_split_seed=int(record["source_split_seed"]),
        source_split_path=str(record["source_split_path"]),
        resample_source_split_per_seed=False,
        source_normalizer_fit_scope="train_only",
        target_shots=5,
        target_val_units=10,
        few_shot_seed=int(record["few_shot_seed"]),
        target_partition_path=str(record["target_partition_path"]),
        resample_few_shot_per_seed=False,
        batch_size=64,
        source_epochs=50,
        target_epochs=20,
        lr=1e-3,
        target_lr=float(cd_stage.get("target_lr", 5e-4)),
        target_lr_scheduler=str(cd_stage.get("target_lr_scheduler", "none")),
        target_lr_min=float(cd_stage.get("target_lr_min", 1e-5)),
        weight_decay=0.0,
        target_weight_decay=0.0,
        source_loss_weight=1.0,
        target_loss_weight=1.0,
        lambda_mmd=float(cd_stage.get("lambda_mmd", 0.1)),
        mmd_sigmas="1,2,4,8,16",
        lambda_source_stage=float(cd_stage.get("lambda_source_stage", 1.0)),
        lambda_pseudo=float(cd_stage.get("lambda_pseudo", 0.5)),
        lambda_monotonic=float(cd_stage.get("lambda_monotonic", 0.05)),
        inv_alignment_mode=inv_alignment_mode,
        lambda_inv_mmd=lambda_inv_mmd,
        lambda_spec_domain=float(cd_stage.get("lambda_spec_domain", 0.0)),
        lambda_transformer_domain_adapter_l2=float(cd_stage.get("lambda_transformer_domain_adapter_l2", 0.0)),
        adaptation_freeze_mode=str(cd_stage.get("adaptation_freeze_mode", "none") or "none"),
        adaptation_freeze_epochs=int(cd_stage.get("adaptation_freeze_epochs", 0) or 0),
        domain_feature_tap=str(cd_stage.get("domain_feature_tap", "inv_mean")),
        stage_feature_mode=str(cd_stage.get("stage_feature_mode", "combined")),
        monotonic_margin=0.0,
        monotonic_pair_gap=1,
        monotonic_pair_stride=1,
        num_pseudo_stages=int(cd_stage.get("pseudo_num_stages", 3)),
        pseudo_start_quantile=float(cd_stage.get("pseudo_start_quantile", 0.5)),
        pseudo_end_quantile=float(cd_stage.get("pseudo_end_quantile", 0.9)),
        disable_pseudo_feature_normalization=False,
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
        transformer_norm_mode="pre",
        transformer_inner_dropout=0.0,
        mamba_block_mode="dd_spd",
        spd_gate_init_bias=-2.0,
        spd_scan_mode="mixed",
        spd_gate_mode="token",
        spd_gate_scheme="shared",
        spd_predictor_mode=str(cd_stage.get("spd_predictor_mode", "shared_head")),
        domain_conditioned_gate=bool(cd_stage.get("domain_conditioned_gate", False)),
        frontend_adapter_mode=str(cd_stage.get("frontend_adapter_mode", "none")),
        transformer_domain_adapter_mode=str(cd_stage.get("transformer_domain_adapter_mode", "none")),
        source_val_all_windows=True,
        target_val_all_windows=True,
        device="auto",
        num_workers=0,
        max_source_train_batches=None,
        max_target_train_batches=None,
        output_dir="",
        # SSL args
        ssl_preset="paper_full",
        ssl_lambda_temporal=0.25,
        ssl_lambda_ntuplet=0.25,
        ssl_lambda_pseudo=0.5,
        pretrain_epochs=20,
        pretrain_batch_size=64,
        pretrain_lr=1e-3,
        max_pretrain_batches=None,
        temporal_hidden_dim=128,
        pseudo_hidden_dim=64,
        ntuplet_mode="flatten",
        ntuplet_normalize=False,
        pseudo_value_mode="last",
    )


def build_stage_data(args: Namespace):
    source_train_full = load_cmapss_split(args.root, args.source_subset, "train", rul_clip=args.rul_clip)
    source_test_raw = load_cmapss_split(args.root, args.source_subset, "test", rul_clip=args.rul_clip)
    target_train_full = load_cmapss_split(args.root, args.target_subset, "train", rul_clip=args.rul_clip)
    target_test_raw = load_cmapss_split(args.root, args.target_subset, "test", rul_clip=args.rul_clip)
    with open(args.source_split_path, "r", encoding="utf-8") as handle:
        source_split = json.load(handle)
    with open(args.target_partition_path, "r", encoding="utf-8") as handle:
        target_partition = json.load(handle)
    source_loaders, source_meta = build_source_stage_data(args, source_train_full, source_test_raw, source_split)
    from train_cd_mambatt_v1 import build_target_cd_data

    target_loaders, target_meta = build_target_cd_data(args, target_train_full, target_test_raw, target_partition)
    target_monotonic_loader = None
    monotonic_meta = None
    if args.lambda_monotonic > 0:
        target_monotonic_loader, monotonic_meta = build_target_monotonic_loader(args, target_train_full, target_partition)
    return (
        source_train_full,
        source_test_raw,
        target_train_full,
        target_test_raw,
        source_loaders,
        source_meta,
        target_loaders,
        target_meta,
        target_monotonic_loader,
        monotonic_meta,
    )


def remap_encoder_state_dict_for_spd(encoder_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    for key, value in encoder_state_dict.items():
        new_key = key
        if ".mamba." in new_key:
            new_key = new_key.replace(".mamba.", ".", 1)
        if ".x_proj.weight" in new_key:
            new_key = new_key.replace(".x_proj.weight", ".x_proj_inv.weight")
        if ".dt_proj.weight" in new_key:
            new_key = new_key.replace(".dt_proj.weight", ".dt_proj_inv.weight")
        if ".dt_proj.bias" in new_key:
            new_key = new_key.replace(".dt_proj.bias", ".dt_proj_inv.bias")
        remapped[new_key] = value
    return remapped


def merge_windowed_data(first: WindowedData, second: WindowedData, *, second_unit_offset: int) -> WindowedData:
    return WindowedData(
        windows=np.concatenate([first.windows, second.windows], axis=0).astype(np.float32),
        targets=np.concatenate([first.targets, second.targets], axis=0).astype(np.float32),
        unit_ids=np.concatenate([first.unit_ids, second.unit_ids + int(second_unit_offset)], axis=0).astype(np.int32),
        end_cycles=np.concatenate([first.end_cycles, second.end_cycles], axis=0).astype(np.int32),
    )


def build_cross_domain_ssl_windows(args: Namespace, source_train_full, target_train_full) -> tuple[WindowedData, dict[str, object]]:
    with open(args.source_split_path, "r", encoding="utf-8") as handle:
        source_split = json.load(handle)
    source_train_raw = select_units(source_train_full, source_split["train_units"])
    source_norm_base = source_train_raw if args.source_normalizer_fit_scope == "train_only" else source_train_full
    source_normalizer = fit_normalizer(source_norm_base)
    target_normalizer = fit_normalizer(target_train_full)

    source_scaled_all = source_normalizer.transform(source_train_full)
    target_scaled_all = target_normalizer.transform(target_train_full)

    source_windows = build_windows(source_scaled_all, args.window_size, stride=args.stride, last_only=False)
    target_windows = build_windows(target_scaled_all, args.window_size, stride=args.stride, last_only=False)
    merged = merge_windowed_data(source_windows, target_windows, second_unit_offset=int(source_windows.unit_ids.max()) + 1000)
    sensor_rows = np.concatenate([source_scaled_all.sensors, target_scaled_all.sensors], axis=0).astype(np.float32)
    return merged, {
        "num_source_windows": int(source_windows.windows.shape[0]),
        "num_target_windows": int(target_windows.windows.shape[0]),
        "merged_windows_shape": tuple(int(dim) for dim in merged.windows.shape),
        "sensor_rows_shape": tuple(int(dim) for dim in sensor_rows.shape),
        "sensor_rows": sensor_rows,
    }


def run_ssl_pretrain(args: Namespace, device: torch.device, run_dir: Path, merged_windows: WindowedData, sensor_rows: np.ndarray) -> dict[str, object]:
    pseudo_statistics = fit_sensor_pseudo_label_statistics(sensor_rows)
    triplet_indices, triplet_labels = build_temporal_triplet_indices(merged_windows)
    pair_indices = build_consecutive_pair_indices(merged_windows)
    pseudo_labels = assign_window_sensor_pseudo_labels(
        merged_windows.windows,
        pseudo_statistics,
        value_mode=args.pseudo_value_mode,
    )

    triplet_loader = build_ssl_loader(
        dataset=IndexedWindowTripletDataset(
            merged_windows.windows,
            triplet_indices,
            triplet_labels,
        ),
        batch_size=args.pretrain_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    pair_loader = build_ssl_loader(
        dataset=IndexedWindowPairDataset(
            merged_windows.windows,
            pair_indices,
        ),
        batch_size=args.pretrain_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    pseudo_loader = None
    if args.ssl_lambda_pseudo > 0:
        pseudo_loader = build_ssl_loader(
            dataset=WindowPseudoLabelDataset(
                merged_windows.windows,
                pseudo_labels,
            ),
            batch_size=args.pretrain_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

    model = build_ssl_pretrainer(args, int(merged_windows.windows.shape[2]), pseudo_statistics.sensor_indices).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    triplet_iter = endless_loader(triplet_loader)
    pair_iter = endless_loader(pair_loader)
    pseudo_iter = None if pseudo_loader is None else endless_loader(pseudo_loader)
    num_steps = max(len(triplet_loader), len(pair_loader), 0 if pseudo_loader is None else len(pseudo_loader))
    if args.max_pretrain_batches is not None:
        num_steps = min(num_steps, int(args.max_pretrain_batches))

    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = -1
    best_ckpt_path = run_dir / "ssl_pretrain_best.pt"

    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        total_loss = 0.0
        total_temporal = 0.0
        total_ntuplet = 0.0
        total_pseudo = 0.0
        for _ in range(num_steps):
            triplets, temporal_labels = next(triplet_iter)
            query_windows, positive_windows = next(pair_iter)

            triplets = triplets.to(device, non_blocking=True)
            temporal_labels = temporal_labels.to(device, non_blocking=True)
            query_windows = query_windows.to(device, non_blocking=True)
            positive_windows = positive_windows.to(device, non_blocking=True)

            temporal_logits = model.forward_temporal_logits(triplets)
            temporal_loss = bce(temporal_logits, temporal_labels)

            query_embeddings = model.forward_ntuplet_embeddings(query_windows)
            positive_embeddings = model.forward_ntuplet_embeddings(positive_windows)
            nt_loss = ntuplet_loss(query_embeddings, positive_embeddings)

            if pseudo_iter is not None:
                pseudo_windows, pseudo_targets = next(pseudo_iter)
                pseudo_windows = pseudo_windows.to(device, non_blocking=True)
                pseudo_targets = pseudo_targets.to(device, non_blocking=True)
                pseudo_logits = model.forward_pseudo_logits(pseudo_windows)
                pseudo_loss = ce(pseudo_logits.view(-1, 3), pseudo_targets.view(-1))
            else:
                pseudo_loss = query_embeddings.new_zeros(())

            loss = (
                args.ssl_lambda_temporal * temporal_loss
                + args.ssl_lambda_ntuplet * nt_loss
                + args.ssl_lambda_pseudo * pseudo_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_temporal += float(temporal_loss.detach().cpu())
            total_ntuplet += float(nt_loss.detach().cpu())
            total_pseudo += float(pseudo_loss.detach().cpu())

        record = {
            "epoch": epoch,
            "ssl_loss": total_loss / num_steps,
            "temporal_loss": total_temporal / num_steps,
            "ntuplet_loss": total_ntuplet / num_steps,
            "pseudo_loss": total_pseudo / num_steps,
        }
        history.append(record)
        print(json.dumps({"stage": "ssl_cross_domain_pretrain", **record}, ensure_ascii=False))
        if record["ssl_loss"] < best_loss:
            best_loss = float(record["ssl_loss"])
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.export_encoder_state_dict(),
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_loss": best_loss,
                },
                best_ckpt_path,
            )

    checkpoint = torch.load(best_ckpt_path, map_location=device)
    return {
        "checkpoint": str(best_ckpt_path),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_loss": float(checkpoint["best_loss"]),
        "history": checkpoint["history"],
        "encoder_state_dict": checkpoint["encoder_state_dict"],
        "triplet_count": int(triplet_indices.shape[0]),
        "pair_count": int(pair_indices.shape[0]),
        "pseudo_window_count": int(pseudo_labels.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-domain SSL pretrain -> source supervised -> targeted adaptation.")
    parser.add_argument("--baseline-run-root", default="/home/shelterpl/cd_mambatt/runs/cd_mambatt_v3_spd_fullmatch/FD001_TO_FD003")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--output-root", default="/home/shelterpl/cd_mambatt/runs/ssl_targeted_experiments")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ssl-preset", choices=("paper_full", "paper_temp_ntuplet", "temporal_only", "custom"), default="paper_full")
    parser.add_argument("--pretrained-ssl-checkpoint", default=None, help="Optional existing ssl_pretrain_best.pt path; when set, skip SSL pretraining and reuse this encoder checkpoint")
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--pretrain-batch-size", type=int, default=64)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--max-pretrain-batches", type=int, default=None)
    parser.add_argument("--domain-feature-tap", default=None)
    parser.add_argument("--target-lr", type=float, default=None)
    parser.add_argument("--target-lr-scheduler", choices=("none", "cosine"), default=None)
    parser.add_argument("--target-lr-min", type=float, default=None)
    args_cli = parser.parse_args()

    set_seed(args_cli.seed)
    device = infer_device(args_cli.device)
    if device.type != "cuda":
        raise RuntimeError("This experiment expects CUDA.")

    record = load_record(Path(args_cli.baseline_run_root).expanduser().resolve(), args_cli.seed)
    args = args_from_record(record)
    args.ssl_preset = str(args_cli.ssl_preset)
    args.pretrain_epochs = int(args_cli.pretrain_epochs)
    args.pretrain_batch_size = int(args_cli.pretrain_batch_size)
    args.pretrain_lr = float(args_cli.pretrain_lr)
    args.max_pretrain_batches = args_cli.max_pretrain_batches
    if args_cli.domain_feature_tap is not None:
        args.domain_feature_tap = str(args_cli.domain_feature_tap)
    if args_cli.target_lr is not None:
        args.target_lr = float(args_cli.target_lr)
    if args_cli.target_lr_scheduler is not None:
        args.target_lr_scheduler = str(args_cli.target_lr_scheduler)
    if args_cli.target_lr_min is not None:
        args.target_lr_min = float(args_cli.target_lr_min)
    ssl_args = Namespace(ssl_preset=str(args.ssl_preset), lambda_temporal=0.0, lambda_ntuplet=0.0, lambda_pseudo=0.0)
    apply_ssl_preset(ssl_args)
    args.ssl_lambda_temporal = float(ssl_args.lambda_temporal)
    args.ssl_lambda_ntuplet = float(ssl_args.lambda_ntuplet)
    args.ssl_lambda_pseudo = float(ssl_args.lambda_pseudo)

    run_dir = Path(args_cli.output_root).expanduser().resolve() / args_cli.experiment_name / record["task"] / f"seed_{args_cli.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (
        source_train_full,
        source_test_raw,
        target_train_full,
        _target_test_raw,
        source_loaders,
        source_meta,
        target_loaders,
        target_meta,
        target_monotonic_loader,
        monotonic_meta,
    ) = build_stage_data(args)

    if args_cli.pretrained_ssl_checkpoint is not None:
        ssl_ckpt_path = Path(args_cli.pretrained_ssl_checkpoint).expanduser().resolve()
        checkpoint = torch.load(ssl_ckpt_path, map_location=device)
        ssl_stage = {
            "checkpoint": str(ssl_ckpt_path),
            "best_epoch": int(checkpoint.get("best_epoch", -1)),
            "best_loss": float(checkpoint.get("best_loss", checkpoint.get("best_ssl_loss", 0.0))),
            "history": checkpoint.get("history", []),
            "encoder_state_dict": checkpoint["encoder_state_dict"],
            "triplet_count": 0,
            "pair_count": 0,
            "pseudo_window_count": 0,
        }
        ssl_meta = {
            "num_source_windows": 0,
            "num_target_windows": 0,
            "merged_windows_shape": None,
        }
    else:
        merged_windows, ssl_meta = build_cross_domain_ssl_windows(args, source_train_full, target_train_full)
        ssl_stage = run_ssl_pretrain(args, device, run_dir, merged_windows, ssl_meta["sensor_rows"])

    model = build_model(args, int(source_meta["train_windows_shape"][2])).to(device)
    remapped_encoder_state_dict = remap_encoder_state_dict_for_spd(ssl_stage["encoder_state_dict"])
    loaded_encoder_keys = load_encoder_weights(model, remapped_encoder_state_dict)
    source_stage = fit_source_stage(
        args,
        args_cli.seed,
        model,
        source_loaders,
        device,
        run_dir,
        float(args.target_scale),
        None if float(args.grad_clip_norm) <= 0 else float(args.grad_clip_norm),
    )
    source_test_metrics = evaluate(
        model,
        source_loaders["test"],
        device,
        float(args.target_scale),
        domain_label_value=0 if uses_domain_conditioning(model) else None,
    )
    target_direct_metrics = evaluate(
        model,
        target_loaders["test"],
        device,
        float(args.target_scale),
        domain_label_value=1 if uses_domain_conditioning(model) else None,
    )

    cd_stage = fit_cd_pseudo_stage(
        args,
        args_cli.seed,
        model,
        source_loaders["train"],
        target_loaders,
        target_monotonic_loader,
        device,
        run_dir,
        float(args.target_scale),
        None if float(args.grad_clip_norm) <= 0 else float(args.grad_clip_norm),
    )
    test_metrics = evaluate(
        model,
        target_loaders["test"],
        device,
        float(args.target_scale),
        domain_label_value=1 if uses_domain_conditioning(model) else None,
    )
    result = {
        "experiment_name": args_cli.experiment_name,
        "seed": int(args_cli.seed),
        "baseline_run_root": str(Path(args_cli.baseline_run_root).expanduser().resolve()),
        "task": str(record["task"]),
        "source_subset": str(record["source_subset"]),
        "target_subset": str(record["target_subset"]),
        "source_split_path": str(record["source_split_path"]),
        "target_partition_path": str(record["target_partition_path"]),
        "ssl_preset": str(args.ssl_preset),
        "ssl_weights": {
            "lambda_temporal": float(args.ssl_lambda_temporal),
            "lambda_ntuplet": float(args.ssl_lambda_ntuplet),
            "lambda_pseudo": float(args.ssl_lambda_pseudo),
        },
        "pretrain_epochs": int(args.pretrain_epochs),
        "pretrain_batch_size": int(args.pretrain_batch_size),
        "pretrain_lr": float(args.pretrain_lr),
        "target_lr": float(args.target_lr),
        "target_lr_scheduler": str(args.target_lr_scheduler),
        "target_lr_min": float(args.target_lr_min),
        "domain_feature_tap": str(args.domain_feature_tap),
        "ssl_stage": {
            "checkpoint": str(ssl_stage["checkpoint"]),
            "best_epoch": int(ssl_stage["best_epoch"]),
            "best_loss": float(ssl_stage["best_loss"]),
            "triplet_count": int(ssl_stage["triplet_count"]),
            "pair_count": int(ssl_stage["pair_count"]),
            "pseudo_window_count": int(ssl_stage["pseudo_window_count"]),
            "loaded_encoder_key_count": int(len(loaded_encoder_keys)),
        },
        "ssl_data": {
            "num_source_windows": int(ssl_meta["num_source_windows"]),
            "num_target_windows": int(ssl_meta["num_target_windows"]),
            "merged_windows_shape": ssl_meta["merged_windows_shape"],
        },
        "source_stage": source_stage,
        "source_test": source_test_metrics,
        "target_direct": target_direct_metrics,
        "cd_stage": cd_stage,
        "test_metrics": test_metrics,
        "source_meta": source_meta,
        "target_meta": target_meta,
        "monotonic_meta": monotonic_meta,
        "baseline_target_direct_rmse": (
            float(record["target_direct"]["rmse"]) if record.get("target_direct", {}).get("rmse") is not None else None
        ),
        "baseline_cd_test_rmse": (
            float(record["cd_stage"]["test_rmse"]) if record.get("cd_stage", {}).get("test_rmse") is not None else None
        ),
    }
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
