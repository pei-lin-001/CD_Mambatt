from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from _pathfix import ensure_repo_root

ensure_repo_root()

from cd_mambatt.data import load_cmapss_split
from train_cd_mambatt_v1 import build_target_cd_data
from train_cd_mambatt_v3 import build_target_monotonic_loader, fit_cd_pseudo_stage
from train_cross_domain_baseline import (
    build_source_stage_data,
    load_or_create_source_split,
    load_or_create_target_partition,
)
from train_supervised import build_model, evaluate, infer_device, set_seed

ROOT = "/home/shelterpl/data/CMAPSS"
TASK_DIR = Path("/home/shelterpl/cd_mambatt/runs/cd_mambatt_v3_spd_fullmatch/FD001_TO_FD003")
SOURCE_SPLIT_PATH = TASK_DIR / "splits/source_engine_split.json"
TARGET_PARTITION_PATH = TASK_DIR / "splits/target_few_shot_seed42.json"
SOURCE_CKPT_PATH = TASK_DIR / "seed_42/source/best.pt"
OUT_ROOT = Path("/home/shelterpl/cd_mambatt/runs/dual_state_quick_ab/FD001_TO_FD003")
SEED = 42


def make_args(spd_scan_mode: str, spd_gate_mode: str, spd_gate_scheme: str) -> argparse.Namespace:
    return argparse.Namespace(
        root=ROOT,
        subset="FD001",
        task="FD001_TO_FD003",
        source_subset="FD001",
        target_subset="FD003",
        window_size=20,
        stride=1,
        rul_clip=125,
        target_scale=1.0,
        grad_clip_norm=0.0,
        source_train_ratio=0.8,
        source_split_seed=42,
        source_split_path=str(SOURCE_SPLIT_PATH),
        resample_source_split_per_seed=False,
        source_normalizer_fit_scope="train_only",
        target_shots=5,
        target_val_units=10,
        target_partition_path=str(TARGET_PARTITION_PATH),
        resample_few_shot_per_seed=False,
        few_shot_seed=SEED,
        seeds=str(SEED),
        num_runs=1,
        base_seed=SEED,
        batch_size=64,
        source_epochs=50,
        target_epochs=6,
        lr=1e-3,
        target_lr=5e-4,
        weight_decay=0.0,
        target_weight_decay=0.0,
        source_loss_weight=1.0,
        target_loss_weight=1.0,
        lambda_mmd=0.1,
        mmd_sigmas="1,2,4,8,16",
        lambda_source_stage=1.0,
        lambda_pseudo=0.5,
        lambda_contrastive=0.0,
        lambda_monotonic=0.05,
        inv_alignment_mode="mmd",
        lambda_domain_adv=0.0,
        lambda_inv_mmd=0.1,
        lambda_conditional_inv_mmd=0.0,
        lambda_inv_spec_orth=0.0,
        lambda_spec_residual=0.0,
        lambda_inv_aux=0.0,
        lambda_spec_reconstruction=0.0,
        semantic_warmup_epochs=0,
        semantic_warmup_lr=5e-4,
        adaptation_freeze_mode="none",
        adaptation_freeze_epochs=0,
        grl_lambda=1.0,
        grl_warmup_epochs=5,
        domain_feature_tap="inv_mean",
        domain_adv_hidden_dim=16,
        domain_adv_dropout=0.0,
        stage_feature_mode="combined",
        contrastive_temperature=0.1,
        disable_contrastive_feature_normalization=False,
        monotonic_margin=0.0,
        monotonic_pair_gap=1,
        monotonic_pair_stride=1,
        num_pseudo_stages=3,
        pseudo_start_quantile=0.5,
        pseudo_end_quantile=0.9,
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
        spd_scan_mode=spd_scan_mode,
        spd_gate_mode=spd_gate_mode,
        spd_gate_scheme=spd_gate_scheme,
        spd_predictor_mode="shared_head",
        source_val_all_windows=True,
        target_val_all_windows=True,
        device="auto",
        num_workers=0,
        max_source_train_batches=None,
        max_target_train_batches=None,
        output_dir=str(OUT_ROOT),
    )


def load_source_weights(model: torch.nn.Module, checkpoint_path: Path, device: torch.device, *, gate_scheme: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"]
    if gate_scheme == "dt_bc":
        translated = dict(state)
        for prefix in ("mamba_blocks.0.",):
            weight_key = prefix + "gate_proj.weight"
            bias_key = prefix + "gate_proj.bias"
            if weight_key in state and bias_key in state:
                translated[prefix + "gate_proj_dt.weight"] = state[weight_key].clone()
                translated[prefix + "gate_proj_dt.bias"] = state[bias_key].clone()
                translated[prefix + "gate_proj_bc.weight"] = state[weight_key].clone()
                translated[prefix + "gate_proj_bc.bias"] = state[bias_key].clone()
        missing, unexpected = model.load_state_dict(translated, strict=False)
        allowed_missing = {"mamba_blocks.0.gate_proj_dt.weight", "mamba_blocks.0.gate_proj_dt.bias", "mamba_blocks.0.gate_proj_bc.weight", "mamba_blocks.0.gate_proj_bc.bias"}
        unexpected = [key for key in unexpected if not key.endswith("gate_proj.weight") and not key.endswith("gate_proj.bias")]
        missing = [key for key in missing if key not in allowed_missing]
        if missing or unexpected:
            raise RuntimeError(f"Unexpected load_state_dict mismatch: missing={missing}, unexpected={unexpected}")
    else:
        model.load_state_dict(state, strict=True)


def run_mode(
    spd_scan_mode: str,
    spd_gate_mode: str,
    spd_gate_scheme: str,
    device: torch.device,
) -> dict[str, float | int | str]:
    set_seed(SEED)
    args = make_args(spd_scan_mode, spd_gate_mode, spd_gate_scheme)
    mode_name = f"{spd_scan_mode}__{spd_gate_mode}__{spd_gate_scheme}"
    run_dir = OUT_ROOT / mode_name / f"seed_{SEED}"
    run_dir.mkdir(parents=True, exist_ok=True)

    source_train_full = load_cmapss_split(ROOT, "FD001", "train", rul_clip=125)
    source_test_raw = load_cmapss_split(ROOT, "FD001", "test", rul_clip=125)
    target_train_full = load_cmapss_split(ROOT, "FD003", "train", rul_clip=125)
    target_test_raw = load_cmapss_split(ROOT, "FD003", "test", rul_clip=125)

    source_split = load_or_create_source_split(args, source_train_full, OUT_ROOT, SEED)
    source_loaders, source_meta = build_source_stage_data(args, source_train_full, source_test_raw, source_split)
    target_partition = load_or_create_target_partition(args, target_train_full, OUT_ROOT, SEED)
    target_loaders, _ = build_target_cd_data(args, target_train_full, target_test_raw, target_partition)
    target_monotonic_loader = None
    if args.lambda_monotonic > 0:
        target_monotonic_loader, _ = build_target_monotonic_loader(args, target_train_full, target_partition)

    model = build_model(args, int(source_meta["train_windows_shape"][2])).to(device)
    load_source_weights(model, SOURCE_CKPT_PATH, device, gate_scheme=spd_gate_scheme)

    stage_result = fit_cd_pseudo_stage(
        args,
        SEED,
        model,
        source_loaders["train"],
        target_loaders,
        target_monotonic_loader,
        device,
        run_dir,
        float(args.target_scale),
        None,
    )
    val_metrics = evaluate(model, target_loaders["val"], device, float(args.target_scale))
    test_metrics = evaluate(model, target_loaders["test"], device, float(args.target_scale))
    best_record = stage_result["best_record"] or {}
    result = {
        "mode": mode_name,
        "spd_scan_mode": spd_scan_mode,
        "spd_gate_mode": spd_gate_mode,
        "spd_gate_scheme": spd_gate_scheme,
        "best_epoch": int(stage_result["best_epoch"]),
        "best_val_rmse": float(stage_result["best_val_rmse"]),
        "final_val_rmse": float(val_metrics["rmse"]),
        "final_test_rmse": float(test_metrics["rmse"]),
        "final_test_score": float(test_metrics["score"]),
        "best_record_gate_mean": float(best_record.get("gate_mean", 0.0)),
        "best_record_gate_dt_mean": float(best_record.get("gate_dt_mean", 0.0)),
        "best_record_gate_bc_mean": float(best_record.get("gate_bc_mean", 0.0)),
        "best_record_pseudo_acceptance_ratio": float(best_record.get("pseudo_acceptance_ratio", 0.0)),
        "checkpoint": str(stage_result["checkpoint"]),
    }
    with (run_dir / "quick_result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def main() -> None:
    device = infer_device("auto")
    if device.type != "cuda":
        raise RuntimeError(f"Expected CUDA device, got {device}")
    results = [
        run_mode("mixed", "token", "shared", device),
        run_mode("mixed", "token", "dt_bc", device),
        run_mode("dual_state", "token", "shared", device),
        run_mode("dual_state", "token", "dt_bc", device),
    ]
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
