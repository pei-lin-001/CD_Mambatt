from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F
from _pathfix import ensure_repo_root

ensure_repo_root()

from cd_mambatt.data import CMAPSSWindowDataset, build_windows, fit_normalizer, load_cmapss_split
from cd_mambatt.losses.mmd import gaussian_mmd_loss
from train_supervised import build_loader, build_model, infer_device

ROOT = "/home/shelterpl/data/CMAPSS"
RUN_ROOT = Path("/home/shelterpl/cd_mambatt/runs/dual_state_quick_ab/FD001_TO_FD003")
WINDOW_SIZE = 20
MAX_BATCHES = 4
MAX_SAMPLES = 200

MODES = [
    ("mixed__token__shared", "mixed", "token", "shared"),
    ("mixed__token__dt_bc", "mixed", "token", "dt_bc"),
    ("dual_state__token__shared", "dual_state", "token", "shared"),
    ("dual_state__token__dt_bc", "dual_state", "token", "dt_bc"),
]


def make_args(spd_scan_mode: str, spd_gate_mode: str, spd_gate_scheme: str) -> Namespace:
    return Namespace(
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
    )


def build_test_loaders():
    src_train = load_cmapss_split(ROOT, "FD001", "train", rul_clip=125)
    tgt_train = load_cmapss_split(ROOT, "FD003", "train", rul_clip=125)
    src_test = load_cmapss_split(ROOT, "FD001", "test", rul_clip=125)
    tgt_test = load_cmapss_split(ROOT, "FD003", "test", rul_clip=125)

    src_norm = fit_normalizer(src_train)
    tgt_norm = fit_normalizer(tgt_train)
    src_w = build_windows(src_norm.transform(src_test), WINDOW_SIZE, stride=1)
    tgt_w = build_windows(tgt_norm.transform(tgt_test), WINDOW_SIZE, stride=1)
    return (
        build_loader(CMAPSSWindowDataset(src_w), 256, False, 0),
        build_loader(CMAPSSWindowDataset(tgt_w), 256, False, 0),
    )


def collect_hidden_steps(model, loader, device, mode_name: str):
    dd_block = model.mamba_blocks[0]
    all_steps = {
        "mixed_combined": [[] for _ in range(WINDOW_SIZE)],
        "dual_inv": [[] for _ in range(WINDOW_SIZE)],
        "dual_spec": [[] for _ in range(WINDOW_SIZE)],
    }
    gate_means = []
    gate_stds = []

    with torch.no_grad():
        for batch_idx, (windows, _) in enumerate(loader):
            if batch_idx >= MAX_BATCHES:
                break
            hidden = model.input_proj(windows.to(device))
            batch, seqlen, _ = hidden.shape

            xz = torch.einsum("od,bld->bol", dd_block.in_proj.weight, hidden)
            if dd_block.in_proj.bias is not None:
                xz = xz + dd_block.in_proj.bias.to(dtype=hidden.dtype).view(1, -1, 1)
            x, _z = xz.chunk(2, dim=1)
            x_conv = dd_block.act(dd_block.conv1d(x)[..., :seqlen])
            selectivity = dd_block._project_selectivity(x_conv, batch=batch, seqlen=seqlen)

            gate = selectivity["gate"]
            gate_means.append(float(gate.mean().detach().cpu()))
            gate_stds.append(float(gate.std(dim=1).mean().detach().cpu()))

            A = -torch.exp(dd_block.A_log.float())
            delta_bias = None if dd_block.dt_proj_inv.bias is None else dd_block.dt_proj_inv.bias.float()

            def run_state_trajectory(dt_key: str, b_key: str):
                dt = selectivity[dt_key].float()
                if delta_bias is not None:
                    dt = dt + delta_bias.unsqueeze(0).unsqueeze(-1)
                dt = F.softplus(dt)
                B_val = selectivity[b_key].float()
                state = torch.zeros(batch, A.shape[0], A.shape[1], device=device)
                per_step = []
                for step in range(seqlen):
                    deltaA = torch.exp(dt[:, :, step].unsqueeze(-1) * A.unsqueeze(0))
                    deltaB_u = dt[:, :, step].unsqueeze(-1) * B_val[:, :, step].unsqueeze(1) * x_conv[:, :, step].unsqueeze(-1)
                    state = deltaA * state + deltaB_u
                    per_step.append(state.reshape(batch, -1).detach().cpu())
                return per_step

            if mode_name.startswith("mixed"):
                mixed_steps = run_state_trajectory("dt_combined", "B_combined")
                for step in range(WINDOW_SIZE):
                    all_steps["mixed_combined"][step].append(mixed_steps[step])
            else:
                inv_steps = run_state_trajectory("dt_inv", "B_inv")
                spec_steps = run_state_trajectory("dt_spec", "B_spec")
                for step in range(WINDOW_SIZE):
                    all_steps["dual_inv"][step].append(inv_steps[step])
                    all_steps["dual_spec"][step].append(spec_steps[step])

    stacked = {}
    for key, per_step_lists in all_steps.items():
        if any(per_step_lists):
            stacked[key] = [torch.cat(chunks, dim=0) if chunks else None for chunks in per_step_lists]
    return stacked, {
        "gate_mean": sum(gate_means) / len(gate_means),
        "gate_time_std": sum(gate_stds) / len(gate_stds),
    }


def compute_step_mmd(src_steps, tgt_steps, device):
    values = []
    for step in range(WINDOW_SIZE):
        src = src_steps[step]
        tgt = tgt_steps[step]
        if src is None or tgt is None:
            values.append(None)
            continue
        n = min(MAX_SAMPLES, src.shape[0], tgt.shape[0])
        values.append(
            float(
                gaussian_mmd_loss(
                    src[:n].to(device),
                    tgt[:n].to(device),
                    sigmas=(0.1, 0.5, 1.0, 2.0, 5.0),
                ).item()
            )
        )
    return values


def summarize_curve(curve):
    first = curve[0] if curve[0] and curve[0] > 1e-8 else 1e-8
    return {
        "step1": curve[0],
        "step5": curve[4],
        "step10": curve[9],
        "step20": curve[19],
        "drift_ratio_20_over_1": curve[19] / first,
    }


def main():
    device = infer_device("auto")
    if device.type != "cuda":
        raise RuntimeError(f"Expected CUDA, got {device}")
    src_loader, tgt_loader = build_test_loaders()
    results = []
    for mode_name, scan_mode, gate_mode, gate_scheme in MODES:
        args = make_args(scan_mode, gate_mode, gate_scheme)
        ckpt_path = RUN_ROOT / mode_name / "seed_42" / "cd_stage" / "best.pt"
        model = build_model(args, 21).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        src_steps, src_gate = collect_hidden_steps(model, src_loader, device, mode_name)
        tgt_steps, tgt_gate = collect_hidden_steps(model, tgt_loader, device, mode_name)

        if mode_name.startswith("mixed"):
            curve = compute_step_mmd(src_steps["mixed_combined"], tgt_steps["mixed_combined"], device)
            result = {
                "mode": mode_name,
                "path": "mixed_combined",
                "gate_source_mean": src_gate["gate_mean"],
                "gate_target_mean": tgt_gate["gate_mean"],
                "gate_source_time_std": src_gate["gate_time_std"],
                "gate_target_time_std": tgt_gate["gate_time_std"],
                "mmd_curve": curve,
                "summary": summarize_curve(curve),
            }
            results.append(result)
        else:
            for path in ("dual_inv", "dual_spec"):
                curve = compute_step_mmd(src_steps[path], tgt_steps[path], device)
                result = {
                    "mode": mode_name,
                    "path": path,
                    "gate_source_mean": src_gate["gate_mean"],
                    "gate_target_mean": tgt_gate["gate_mean"],
                    "gate_source_time_std": src_gate["gate_time_std"],
                    "gate_target_time_std": tgt_gate["gate_time_std"],
                    "mmd_curve": curve,
                    "summary": summarize_curve(curve),
                }
                results.append(result)

    out_path = RUN_ROOT / "drift_probe_summary.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
