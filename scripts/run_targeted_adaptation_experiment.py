from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cd_mambatt.cross_domain import resolve_cross_domain_task
from cd_mambatt.data import load_cmapss_split
from train_cd_mambatt_v1 import build_target_cd_data
from train_cd_mambatt_v3 import build_target_monotonic_loader, fit_cd_pseudo_stage
from train_cross_domain_baseline import build_source_stage_data
from train_supervised import build_model, evaluate, infer_device, set_seed


def load_record(run_root: Path, seed: int) -> dict[str, object]:
    return json.loads((run_root / f"seed_{seed}" / "result.json").read_text(encoding="utf-8"))


def build_manual_record(args_cli: argparse.Namespace) -> dict[str, object]:
    if not args_cli.source_checkpoint:
        raise ValueError("Manual mode requires --source-checkpoint")
    if not args_cli.source_split_path:
        raise ValueError("Manual mode requires --source-split-path")
    if not args_cli.target_partition_path:
        raise ValueError("Manual mode requires --target-partition-path")

    task = resolve_cross_domain_task(
        args_cli.task,
        source_subset=args_cli.source_subset,
        target_subset=args_cli.target_subset,
    )
    few_shot_seed = int(args_cli.few_shot_seed if args_cli.few_shot_seed is not None else args_cli.seed)
    return {
        "seed": int(args_cli.seed),
        "task": task.name,
        "source_subset": task.source_subset,
        "target_subset": task.target_subset,
        "source_split_path": str(Path(args_cli.source_split_path).expanduser().resolve()),
        "source_split_seed": int(args_cli.source_split_seed),
        "target_partition_path": str(Path(args_cli.target_partition_path).expanduser().resolve()),
        "few_shot_seed": few_shot_seed,
        "source_stage": {
            "checkpoint": str(Path(args_cli.source_checkpoint).expanduser().resolve()),
        },
        "target_direct": {
            "rmse": float(args_cli.baseline_target_direct_rmse) if args_cli.baseline_target_direct_rmse is not None else None,
        },
        "cd_stage": {
            "test_rmse": float(args_cli.baseline_cd_test_rmse) if args_cli.baseline_cd_test_rmse is not None else None,
        },
    }


def args_from_record(record: dict[str, object]) -> Namespace:
    cd_stage = record["cd_stage"]
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
        seeds=str(record["seed"]),
        num_runs=1,
        base_seed=int(record["seed"]),
        batch_size=64,
        source_epochs=50,
        target_epochs=20,
        lr=1e-3,
        target_lr=5e-4,
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
        lambda_contrastive=float(cd_stage.get("lambda_contrastive", 0.0)),
        lambda_monotonic=float(cd_stage.get("lambda_monotonic", 0.05)),
        inv_alignment_mode=str(cd_stage.get("inv_alignment_mode", "mmd")),
        lambda_domain_adv=float(cd_stage.get("lambda_domain_adv", 0.0)),
        lambda_inv_mmd=float(cd_stage.get("lambda_inv_mmd", 0.1)),
        lambda_spec_domain=float(cd_stage.get("lambda_spec_domain", 0.0)),
        lambda_conditional_inv_mmd=float(cd_stage.get("lambda_conditional_inv_mmd", 0.0)),
        lambda_conditional_proto=float(cd_stage.get("lambda_conditional_proto", 0.0)),
        lambda_conditional_proto_ce=float(cd_stage.get("lambda_conditional_proto_ce", 0.0)),
        conditional_target_scope=str(cd_stage.get("conditional_target_scope", "labeled_accepted")),
        lambda_inv_spec_orth=float(cd_stage.get("lambda_inv_spec_orth", 0.0)),
        lambda_spec_residual=float(cd_stage.get("lambda_spec_residual", 0.0)),
        lambda_inv_aux=float(cd_stage.get("lambda_inv_aux", 0.0)),
        lambda_spec_reconstruction=float(cd_stage.get("lambda_spec_reconstruction", 0.0)),
        semantic_warmup_epochs=int(cd_stage.get("semantic_warmup_epochs", 0)),
        semantic_warmup_lr=float(cd_stage.get("semantic_warmup_lr", 5e-4)),
        adaptation_freeze_mode=str(cd_stage.get("adaptation_freeze_mode", "none") or "none"),
        adaptation_freeze_epochs=int(cd_stage.get("adaptation_freeze_epochs", 0) or 0),
        grl_lambda=float(cd_stage.get("grl_lambda", 1.0)),
        grl_warmup_epochs=int(cd_stage.get("grl_warmup_epochs", 5)),
        domain_feature_tap=str(cd_stage.get("domain_feature_tap", "inv_mean")),
        domain_adv_hidden_dim=int(cd_stage.get("domain_adv_hidden_dim", 16)),
        domain_adv_dropout=0.0,
        stage_feature_mode=str(cd_stage.get("stage_feature_mode", "combined")),
        contrastive_temperature=0.1,
        disable_contrastive_feature_normalization=False,
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
        source_val_all_windows=True,
        target_val_all_windows=True,
        device="auto",
        num_workers=0,
        max_source_train_batches=None,
        max_target_train_batches=None,
        output_dir="",
    )


def build_data(args: Namespace):
    source_train_full = load_cmapss_split(args.root, args.source_subset, "train", rul_clip=args.rul_clip)
    source_test_raw = load_cmapss_split(args.root, args.source_subset, "test", rul_clip=args.rul_clip)
    target_train_full = load_cmapss_split(args.root, args.target_subset, "train", rul_clip=args.rul_clip)
    target_test_raw = load_cmapss_split(args.root, args.target_subset, "test", rul_clip=args.rul_clip)
    with open(args.source_split_path, "r", encoding="utf-8") as handle:
        source_split = json.load(handle)
    with open(args.target_partition_path, "r", encoding="utf-8") as handle:
        target_partition = json.load(handle)
    source_loaders, source_meta = build_source_stage_data(args, source_train_full, source_test_raw, source_split)
    target_loaders, target_meta = build_target_cd_data(args, target_train_full, target_test_raw, target_partition)
    target_monotonic_loader = None
    monotonic_meta = None
    if args.lambda_monotonic > 0:
        target_monotonic_loader, monotonic_meta = build_target_monotonic_loader(args, target_train_full, target_partition)
    return source_loaders, source_meta, target_loaders, target_meta, target_monotonic_loader, monotonic_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Run targeted adaptation experiments from an existing source checkpoint.")
    parser.add_argument("--baseline-run-root", default="/home/shelterpl/cd_mambatt/runs/cd_mambatt_v3_spd_fullmatch/FD001_TO_FD003")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--output-root", default="/home/shelterpl/cd_mambatt/runs/targeted_mechanism_experiments")
    parser.add_argument("--task", default=None)
    parser.add_argument("--source-subset", default=None)
    parser.add_argument("--target-subset", default=None)
    parser.add_argument("--source-checkpoint", default=None)
    parser.add_argument("--source-split-path", default=None)
    parser.add_argument("--source-split-seed", type=int, default=42)
    parser.add_argument("--target-partition-path", default=None)
    parser.add_argument("--few-shot-seed", type=int, default=None)
    parser.add_argument("--baseline-target-direct-rmse", type=float, default=None)
    parser.add_argument("--baseline-cd-test-rmse", type=float, default=None)
    parser.add_argument("--adaptation-freeze-mode", default=None)
    parser.add_argument("--adaptation-freeze-epochs", type=int, default=None)
    parser.add_argument("--inv-alignment-mode", default=None)
    parser.add_argument("--lambda-inv-mmd", type=float, default=None)
    parser.add_argument("--lambda-spec-domain", type=float, default=None)
    parser.add_argument("--lambda-conditional-inv-mmd", type=float, default=None)
    parser.add_argument("--lambda-conditional-proto", type=float, default=None)
    parser.add_argument("--lambda-conditional-proto-ce", type=float, default=None)
    parser.add_argument("--lambda-contrastive", type=float, default=None)
    parser.add_argument("--lambda-pseudo", type=float, default=None)
    parser.add_argument("--lambda-monotonic", type=float, default=None)
    parser.add_argument("--conditional-target-scope", default=None)
    parser.add_argument("--stage-feature-mode", default=None)
    parser.add_argument("--domain-feature-tap", default=None)
    parser.add_argument("--pseudo-start-quantile", type=float, default=None)
    parser.add_argument("--pseudo-end-quantile", type=float, default=None)
    parser.add_argument("--target-epochs", type=int, default=None)
    parser.add_argument("--target-lr", type=float, default=None)
    parser.add_argument("--target-lr-scheduler", choices=("none", "cosine"), default=None)
    parser.add_argument("--target-lr-min", type=float, default=None)
    parser.add_argument("--adaptation-seed", type=int, default=None)
    parser.add_argument("--max-target-train-batches", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args_cli = parser.parse_args()

    set_seed(args_cli.seed)
    device = infer_device(args_cli.device)
    if device.type != "cuda":
        raise RuntimeError("This experiment expects CUDA.")

    baseline_run_root: Path | None = None
    if args_cli.source_checkpoint is not None:
        record = build_manual_record(args_cli)
    else:
        baseline_run_root = Path(args_cli.baseline_run_root).expanduser().resolve()
        record = load_record(baseline_run_root, args_cli.seed)
    args = args_from_record(record)
    if args_cli.adaptation_freeze_mode is not None:
        args.adaptation_freeze_mode = str(args_cli.adaptation_freeze_mode)
    if args_cli.adaptation_freeze_epochs is not None:
        args.adaptation_freeze_epochs = int(args_cli.adaptation_freeze_epochs)
    if args_cli.inv_alignment_mode is not None:
        args.inv_alignment_mode = str(args_cli.inv_alignment_mode)
    if args_cli.lambda_inv_mmd is not None:
        args.lambda_inv_mmd = float(args_cli.lambda_inv_mmd)
    if args_cli.lambda_spec_domain is not None:
        args.lambda_spec_domain = float(args_cli.lambda_spec_domain)
    if args_cli.lambda_conditional_inv_mmd is not None:
        args.lambda_conditional_inv_mmd = float(args_cli.lambda_conditional_inv_mmd)
    if args_cli.lambda_conditional_proto is not None:
        args.lambda_conditional_proto = float(args_cli.lambda_conditional_proto)
    if args_cli.lambda_conditional_proto_ce is not None:
        args.lambda_conditional_proto_ce = float(args_cli.lambda_conditional_proto_ce)
    if args_cli.lambda_contrastive is not None:
        args.lambda_contrastive = float(args_cli.lambda_contrastive)
    if args_cli.lambda_pseudo is not None:
        args.lambda_pseudo = float(args_cli.lambda_pseudo)
    if args_cli.lambda_monotonic is not None:
        args.lambda_monotonic = float(args_cli.lambda_monotonic)
    if args_cli.conditional_target_scope is not None:
        args.conditional_target_scope = str(args_cli.conditional_target_scope)
    if args_cli.stage_feature_mode is not None:
        args.stage_feature_mode = str(args_cli.stage_feature_mode)
    if args_cli.domain_feature_tap is not None:
        args.domain_feature_tap = str(args_cli.domain_feature_tap)
    if args_cli.pseudo_start_quantile is not None:
        args.pseudo_start_quantile = float(args_cli.pseudo_start_quantile)
    if args_cli.pseudo_end_quantile is not None:
        args.pseudo_end_quantile = float(args_cli.pseudo_end_quantile)
    if args_cli.target_epochs is not None:
        args.target_epochs = int(args_cli.target_epochs)
    if args_cli.target_lr is not None:
        args.target_lr = float(args_cli.target_lr)
    if args_cli.target_lr_scheduler is not None:
        args.target_lr_scheduler = str(args_cli.target_lr_scheduler)
    if args_cli.target_lr_min is not None:
        args.target_lr_min = float(args_cli.target_lr_min)
    if args_cli.max_target_train_batches is not None:
        args.max_target_train_batches = int(args_cli.max_target_train_batches)

    source_loaders, source_meta, target_loaders, target_meta, target_monotonic_loader, monotonic_meta = build_data(args)

    model = build_model(args, int(source_meta["train_windows_shape"][2])).to(device)
    source_checkpoint = torch.load(record["source_stage"]["checkpoint"], map_location=device)
    model.load_state_dict(source_checkpoint["model_state_dict"])
    adaptation_seed = int(args_cli.adaptation_seed) if args_cli.adaptation_seed is not None else int(args_cli.seed)
    set_seed(adaptation_seed)

    run_dir = Path(args_cli.output_root).expanduser().resolve() / args_cli.experiment_name / record["task"] / f"seed_{args_cli.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
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
    test_metrics = evaluate(model, target_loaders["test"], device, float(args.target_scale))
    result = {
        "experiment_name": args_cli.experiment_name,
        "baseline_run_root": str(baseline_run_root) if baseline_run_root is not None else None,
        "seed": int(args_cli.seed),
        "source_checkpoint": str(record["source_stage"]["checkpoint"]),
        "adaptation_freeze_mode": str(args.adaptation_freeze_mode),
        "adaptation_freeze_epochs": int(args.adaptation_freeze_epochs),
        "inv_alignment_mode": str(args.inv_alignment_mode),
        "lambda_inv_mmd": float(args.lambda_inv_mmd),
        "lambda_spec_domain": float(args.lambda_spec_domain),
            "lambda_conditional_inv_mmd": float(args.lambda_conditional_inv_mmd),
            "lambda_conditional_proto": float(args.lambda_conditional_proto),
            "lambda_conditional_proto_ce": float(args.lambda_conditional_proto_ce),
            "lambda_contrastive": float(args.lambda_contrastive),
            "lambda_pseudo": float(args.lambda_pseudo),
            "lambda_monotonic": float(args.lambda_monotonic),
            "conditional_target_scope": str(args.conditional_target_scope),
        "stage_feature_mode": str(args.stage_feature_mode),
        "domain_feature_tap": str(args.domain_feature_tap),
        "pseudo_start_quantile": float(args.pseudo_start_quantile),
        "pseudo_end_quantile": float(args.pseudo_end_quantile),
        "target_epochs": int(args.target_epochs),
        "target_lr": float(args.target_lr),
        "target_lr_scheduler": str(args.target_lr_scheduler),
        "target_lr_min": float(args.target_lr_min),
        "adaptation_seed": adaptation_seed,
        "max_target_train_batches": args.max_target_train_batches,
        "source_meta": source_meta,
        "target_meta": target_meta,
        "monotonic_meta": monotonic_meta,
        "cd_stage": cd_stage,
        "test_metrics": test_metrics,
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
