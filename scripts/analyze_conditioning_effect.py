from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch

from cd_mambatt.data import load_cmapss_split
from train_cd_mambatt_v1 import build_target_cd_data
from train_cross_domain_baseline import build_source_stage_data
from train_supervised import build_model, evaluate


def gate_mean(model, loader, device: torch.device, label_value: int, max_batches: int | None = None) -> float:
    vals: list[float] = []
    with torch.no_grad():
        for idx, (windows, _) in enumerate(loader):
            if max_batches is not None and idx >= max_batches:
                break
            windows = windows.to(device)
            lab = torch.full((windows.shape[0],), label_value, dtype=torch.long, device=device)
            out = model.forward_features_with_aux(windows, domain_label=lab)
            vals.append(float(out["gate_mean"].detach().cpu()))
    return float(sum(vals) / max(len(vals), 1))


def pred_delta(model, loader, device: torch.device, max_batches: int = 4) -> float:
    diffs: list[float] = []
    with torch.no_grad():
        for idx, (windows, _) in enumerate(loader):
            if idx >= max_batches:
                break
            windows = windows.to(device)
            z0 = torch.zeros(windows.shape[0], dtype=torch.long, device=device)
            z1 = torch.ones(windows.shape[0], dtype=torch.long, device=device)
            p0 = model(windows, domain_label=z0)
            p1 = model(windows, domain_label=z1)
            diffs.append(float((p1 - p0).abs().mean().detach().cpu()))
    return float(sum(diffs) / max(len(diffs), 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze label-conditioning effect for a completed run")
    parser.add_argument("--run-root", required=True, help="Run root directory that contains task subdirectories")
    parser.add_argument("--task", default=None, help="Optional task subdirectory name; defaults to the first one found")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--root", default="/home/shelterpl/data/CMAPSS")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    task_dir = run_root / args.task if args.task else next(path for path in sorted(run_root.iterdir()) if path.is_dir())
    summary_payload = json.loads((task_dir / "summary.json").read_text())
    summary = summary_payload["summary"]
    source_subset = summary["source_subset"]
    target_subset = summary["target_subset"]
    first_seed = int(summary["seeds"][0])
    probe_ckpt = torch.load(task_dir / f"seed_{first_seed}" / "cd_stage" / "best.pt", map_location="cpu")
    probe_state = probe_ckpt["model_state_dict"]
    inferred_domain_conditioned_gate = any("domain_gate_shift" in key for key in probe_state)
    inferred_frontend_adapter_mode = "none"
    inferred_transformer_domain_adapter_mode = "none"
    if any("frontend_target_adapter" in key for key in probe_state):
        inferred_frontend_adapter_mode = "target_residual"
    elif any("frontend_target_scale" in key or "frontend_target_bias" in key for key in probe_state):
        inferred_frontend_adapter_mode = "target_affine"
    if any("target_ln1_gamma" in key or "target_ln2_gamma" in key for key in probe_state):
        inferred_transformer_domain_adapter_mode = "target_film"
    elif any("target_ln1_beta" in key or "target_ln2_beta" in key for key in probe_state):
        inferred_transformer_domain_adapter_mode = "target_shift"

    model_args = Namespace(
        root=args.root,
        window_size=20,
        stride=1,
        rul_clip=125,
        target_scale=1.0,
        source_normalizer_fit_scope="train_only",
        batch_size=64,
        num_workers=0,
        source_val_all_windows=True,
        target_val_all_windows=True,
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
        spd_predictor_mode=str(summary.get("spd_predictor_mode", "shared_head")),
        domain_conditioned_gate=bool(summary.get("domain_conditioned_gate", inferred_domain_conditioned_gate)),
        frontend_adapter_mode=str(summary.get("frontend_adapter_mode", inferred_frontend_adapter_mode)),
        transformer_domain_adapter_mode=str(summary.get("transformer_domain_adapter_mode", inferred_transformer_domain_adapter_mode)),
    )

    device = torch.device(args.device)
    src_train = load_cmapss_split(args.root, source_subset, "train", rul_clip=125)
    src_test = load_cmapss_split(args.root, source_subset, "test", rul_clip=125)
    tgt_train = load_cmapss_split(args.root, target_subset, "train", rul_clip=125)
    tgt_test = load_cmapss_split(args.root, target_subset, "test", rul_clip=125)
    source_split = json.loads((task_dir / "splits" / "source_engine_split.json").read_text())
    source_loaders, _ = build_source_stage_data(model_args, src_train, src_test, source_split)

    rows: list[dict[str, float | int]] = []
    for seed in summary["seeds"]:
        target_partition = json.loads((task_dir / "splits" / f"target_few_shot_seed{seed}.json").read_text())
        target_loaders, _ = build_target_cd_data(model_args, tgt_train, tgt_test, target_partition)
        model = build_model(model_args, 21).to(device)
        ckpt = torch.load(task_dir / f"seed_{seed}" / "cd_stage" / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        row = {
            "seed": int(seed),
            "src_rmse_label0": float(evaluate(model, source_loaders["test"], device, 1.0, domain_label_value=0)["rmse"]),
            "src_rmse_label1": float(evaluate(model, source_loaders["test"], device, 1.0, domain_label_value=1)["rmse"]),
            "tgt_rmse_label0": float(evaluate(model, target_loaders["test"], device, 1.0, domain_label_value=0)["rmse"]),
            "tgt_rmse_label1": float(evaluate(model, target_loaders["test"], device, 1.0, domain_label_value=1)["rmse"]),
            "src_gate_label0": gate_mean(model, source_loaders["test"], device, 0),
            "src_gate_label1": gate_mean(model, source_loaders["test"], device, 1),
            "tgt_gate_label0": gate_mean(model, target_loaders["test"], device, 0),
            "tgt_gate_label1": gate_mean(model, target_loaders["test"], device, 1),
            "src_pred_abs_delta_1_vs_0": pred_delta(model, source_loaders["test"], device),
            "tgt_pred_abs_delta_1_vs_0": pred_delta(model, target_loaders["test"], device),
        }
        for key, value in ckpt["model_state_dict"].items():
            if "domain_gate_shift" in key and value.numel() == 1:
                row[key] = float(value.item())
            if "frontend_target_scale" in key and value.numel() > 0:
                row[f"{key}_norm"] = float(value.norm().item())
            if "frontend_target_bias" in key and value.numel() > 0:
                row[f"{key}_norm"] = float(value.norm().item())
            if "frontend_target_adapter.weight" in key:
                row[f"{key}_norm"] = float(value.norm().item())
            if "frontend_target_adapter.bias" in key:
                row[f"{key}_norm"] = float(value.norm().item())
            if "target_ln1_gamma" in key or "target_ln2_gamma" in key or "target_ln1_beta" in key or "target_ln2_beta" in key:
                row[f"{key}_norm"] = float(value.norm().item())
        rows.append(row)

    payload = {
        "run_root": str(run_root),
        "task": task_dir.name,
        "frontend_adapter_mode": str(model_args.frontend_adapter_mode),
        "transformer_domain_adapter_mode": str(model_args.transformer_domain_adapter_mode),
        "domain_conditioned_gate": bool(model_args.domain_conditioned_gate),
        "rows": rows,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
