from __future__ import annotations

import argparse
import copy
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cd_mambatt.losses.mmd import gaussian_mmd_loss
from cd_mambatt.metrics import rmse
from train_cd_mambatt_v1 import build_target_cd_data
from train_cd_mambatt_v3 import (
    collect_source_stage_statistics,
    resolve_pseudo_quantile,
    run_cd_pseudo_epoch,
)
from train_cross_domain_baseline import build_source_stage_data
from train_supervised import build_model, evaluate, infer_device, set_seed


def load_run_record(run_root: Path, seed: int) -> dict[str, object]:
    result_path = run_root / f"seed_{seed}" / "result.json"
    return json.loads(result_path.read_text(encoding="utf-8"))


def make_default_args(record: dict[str, object]) -> Namespace:
    cd_stage = record["cd_stage"]
    output_root = str(Path(record["source_split_path"]).parents[2])
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
        lambda_conditional_inv_mmd=float(cd_stage.get("lambda_conditional_inv_mmd", 0.0)),
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
        output_dir=output_root,
    )


def build_loaders_from_record(args: Namespace, record: dict[str, object]):
    from cd_mambatt.data import load_cmapss_split

    source_train_full = load_cmapss_split(args.root, args.source_subset, "train", rul_clip=args.rul_clip)
    source_test_raw = load_cmapss_split(args.root, args.source_subset, "test", rul_clip=args.rul_clip)
    target_train_full = load_cmapss_split(args.root, args.target_subset, "train", rul_clip=args.rul_clip)
    target_test_raw = load_cmapss_split(args.root, args.target_subset, "test", rul_clip=args.rul_clip)

    with open(record["source_split_path"], "r", encoding="utf-8") as handle:
        source_split = json.load(handle)
    with open(record["target_partition_path"], "r", encoding="utf-8") as handle:
        target_partition = json.load(handle)

    source_loaders, source_meta = build_source_stage_data(args, source_train_full, source_test_raw, source_split)
    target_loaders, target_meta = build_target_cd_data(args, target_train_full, target_test_raw, target_partition)
    return source_loaders, source_meta, target_loaders, target_meta


def load_model(args: Namespace, checkpoint_path: str | Path, device: torch.device) -> nn.Module:
    model = build_model(args, 21).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def get_dd_block(model: nn.Module):
    if not hasattr(model, "mamba_blocks") or len(model.mamba_blocks) != 1:
        raise ValueError("This diagnostic currently expects exactly one Mamba block.")
    block = model.mamba_blocks[0]
    if not hasattr(block, "_project_selectivity"):
        raise TypeError("Expected a DDMambaBlock-compatible module.")
    return block


def compute_block_tensors(model: nn.Module, windows: torch.Tensor) -> dict[str, torch.Tensor]:
    block = get_dd_block(model)
    hidden = model.input_proj(windows)
    batch, seqlen, _ = hidden.shape
    xz = rearrange(
        block.in_proj.weight @ rearrange(hidden, "b l d -> d (b l)"),
        "d (b l) -> b d l",
        l=seqlen,
    )
    if block.in_proj.bias is not None:
        xz = xz + rearrange(block.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
    x_raw, z_raw = xz.chunk(2, dim=1)
    x_conv = block.act(block.conv1d(x_raw)[..., :seqlen])
    selectivity = block._project_selectivity(x_conv, batch=batch, seqlen=seqlen)
    inv_core = block._run_selective_scan_core(
        x_conv,
        selectivity["dt_inv"],
        selectivity["B_inv"],
        selectivity["C_inv"],
    )
    combined_core = block._run_selective_scan_core(
        x_conv,
        selectivity["dt_combined"],
        selectivity["B_combined"],
        selectivity["C_combined"],
    )
    D_x = x_conv * rearrange(block.D.float(), "d -> d 1")
    aux = model.forward_features_with_aux(windows)
    return {
        "x_conv": x_conv.detach(),
        "z_raw": z_raw.detach(),
        "gate": selectivity["gate"].detach(),
        "dt_inv": selectivity["dt_inv"].detach(),
        "dt_spec": selectivity["dt_spec"].detach(),
        "B_inv": selectivity["B_inv"].detach(),
        "B_spec": selectivity["B_spec"].detach(),
        "C_inv": selectivity["C_inv"].detach(),
        "C_spec": selectivity["C_spec"].detach(),
        "D_x": D_x.detach(),
        "inv_core": inv_core.detach(),
        "combined_core": combined_core.detach(),
        "pre_transformer_last": aux["pre_transformer_features"].detach(),
        "features": aux["features"].detach(),
        "invariant_features": aux["invariant_features"].detach(),
        "specific_features": aux["specific_features"].detach(),
    }


def flatten_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return rearrange(tensor, "b d l -> b (d l)")
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 1:
        return tensor.unsqueeze(-1)
    raise ValueError(f"Unsupported tensor rank: {tensor.ndim}")


def collect_views(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    max_batches: int,
) -> dict[str, torch.Tensor]:
    store: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for batch_idx, (windows, _) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            windows = windows.to(device, non_blocking=True)
            tensors = compute_block_tensors(model, windows)
            for name, tensor in tensors.items():
                store.setdefault(name, []).append(flatten_view(tensor).cpu())
    return {name: torch.cat(chunks, dim=0) for name, chunks in store.items()}


def linear_domain_probe(src: torch.Tensor, tgt: torch.Tensor) -> dict[str, float]:
    X = np.concatenate([src.numpy(), tgt.numpy()], axis=0)
    y = np.concatenate([np.zeros(len(src), dtype=np.int64), np.ones(len(tgt), dtype=np.int64)])
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    n = min(len(src), len(tgt), 256)
    mmd = float(
        gaussian_mmd_loss(
            src[:n].cuda(),
            tgt[:n].cuda(),
            sigmas=(0.1, 0.5, 1.0, 2.0, 5.0),
        ).item()
    )
    return {
        "domain_accuracy_mean": float(scores.mean()),
        "domain_accuracy_std": float(scores.std()),
        "mmd": mmd,
    }


def compute_state_trajectory(block, x_conv: torch.Tensor, dt: torch.Tensor, B: torch.Tensor) -> list[torch.Tensor]:
    A = -torch.exp(block.A_log.float())
    delta_bias = None if block.dt_proj_inv.bias is None else block.dt_proj_inv.bias.float()
    dt_eff = dt.float()
    if delta_bias is not None:
        dt_eff = dt_eff + delta_bias.unsqueeze(0).unsqueeze(-1)
    dt_eff = F.softplus(dt_eff)
    state = torch.zeros(x_conv.shape[0], A.shape[0], A.shape[1], device=x_conv.device)
    per_step: list[torch.Tensor] = []
    for step in range(x_conv.shape[-1]):
        deltaA = torch.exp(dt_eff[:, :, step].unsqueeze(-1) * A.unsqueeze(0))
        deltaB_u = (
            dt_eff[:, :, step].unsqueeze(-1)
            * B[:, :, step].unsqueeze(1)
            * x_conv[:, :, step].unsqueeze(-1)
        )
        state = deltaA * state + deltaB_u
        per_step.append(state.reshape(state.shape[0], -1).detach().cpu())
    return per_step


def collect_stepwise_drift(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    max_batches: int,
) -> dict[str, list[torch.Tensor]]:
    block = get_dd_block(model)
    out = {
        "x_conv": [[] for _ in range(20)],
        "D_x": [[] for _ in range(20)],
        "inv_state": [[] for _ in range(20)],
        "mixed_state": [[] for _ in range(20)],
    }
    with torch.no_grad():
        for batch_idx, (windows, _) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            windows = windows.to(device, non_blocking=True)
            tensors = compute_block_tensors(model, windows)
            x_conv = tensors["x_conv"]
            D_x = tensors["D_x"]
            batch, _, seqlen = x_conv.shape
            selectivity = block._project_selectivity(x_conv, batch=batch, seqlen=seqlen)
            inv_states = compute_state_trajectory(block, x_conv, selectivity["dt_inv"], selectivity["B_inv"])
            mixed_states = compute_state_trajectory(block, x_conv, selectivity["dt_combined"], selectivity["B_combined"])
            for step in range(seqlen):
                out["x_conv"][step].append(x_conv[:, :, step].detach().cpu())
                out["D_x"][step].append(D_x[:, :, step].detach().cpu())
                out["inv_state"][step].append(inv_states[step])
                out["mixed_state"][step].append(mixed_states[step])
    return {name: [torch.cat(parts, dim=0) for parts in steps] for name, steps in out.items()}


def stepwise_curve(src_steps: list[torch.Tensor], tgt_steps: list[torch.Tensor], device: torch.device) -> dict[str, object]:
    values: list[float] = []
    for src, tgt in zip(src_steps, tgt_steps):
        n = min(len(src), len(tgt), 256)
        values.append(
            float(
                gaussian_mmd_loss(
                    src[:n].to(device),
                    tgt[:n].to(device),
                    sigmas=(0.1, 0.5, 1.0, 2.0, 5.0),
                ).item()
            )
        )
    first = values[0] if values[0] > 1e-8 else 1e-8
    return {
        "curve": values,
        "step1": values[0],
        "step5": values[4],
        "step10": values[9],
        "step20": values[19],
        "drift_ratio_20_over_1": float(values[19] / first),
    }


def parameter_groups(model: nn.Module) -> dict[str, list[tuple[str, nn.Parameter]]]:
    groups: dict[str, list[tuple[str, nn.Parameter]]] = {
        "head": [],
        "transformer": [],
        "mamba_frontend": [],
        "mamba_inv_selectivity": [],
        "mamba_spec_gate": [],
        "mamba_dynamics": [],
        "mamba_out_proj": [],
        "other": [],
    }
    for name, param in model.named_parameters():
        if name.startswith("head."):
            key = "head"
        elif name.startswith("transformer_blocks.") or name.startswith("transformer_encoder."):
            key = "transformer"
        elif ".x_proj_spec." in name or ".dt_proj_spec." in name or ".gate_proj" in name:
            key = "mamba_spec_gate"
        elif ".x_proj_inv." in name or ".dt_proj_inv." in name:
            key = "mamba_inv_selectivity"
        elif ".A_log" in name or ".D" in name:
            key = "mamba_dynamics"
        elif ".out_proj." in name:
            key = "mamba_out_proj"
        elif ".in_proj." in name or ".conv1d." in name or name.startswith("input_proj."):
            key = "mamba_frontend"
        else:
            key = "other"
        groups[key].append((name, param))
    return groups


def gradient_norm_report(model: nn.Module, windows: torch.Tensor, targets: torch.Tensor) -> dict[str, dict[str, float]]:
    model.zero_grad(set_to_none=True)
    predictions = model(windows)
    loss = nn.MSELoss()(predictions, targets)
    loss.backward()
    report: dict[str, dict[str, float]] = {"_meta": {"loss": float(loss.detach().cpu())}}
    for group_name, params in parameter_groups(model).items():
        sq_norm = 0.0
        elem_count = 0
        with_grad = 0
        for _, param in params:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            sq_norm += float(grad.pow(2).sum().cpu())
            elem_count += int(grad.numel())
            with_grad += 1
        report[group_name] = {
            "l2_grad_norm": float(np.sqrt(sq_norm)) if sq_norm > 0 else 0.0,
            "param_tensors_with_grad": float(with_grad),
            "grad_element_count": float(elem_count),
        }
    model.zero_grad(set_to_none=True)
    return report


def forward_variant(model: nn.Module, windows: torch.Tensor, variant: str) -> torch.Tensor:
    block = get_dd_block(model)
    hidden = model.input_proj(windows)
    batch, seqlen, _ = hidden.shape
    xz = rearrange(
        block.in_proj.weight @ rearrange(hidden, "b l d -> d (b l)"),
        "d (b l) -> b d l",
        l=seqlen,
    )
    if block.in_proj.bias is not None:
        xz = xz + rearrange(block.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
    x, z = xz.chunk(2, dim=1)
    x = block.act(block.conv1d(x)[..., :seqlen])
    selectivity = block._project_selectivity(x, batch=batch, seqlen=seqlen)

    if variant == "full":
        hidden_out = block._run_selective_scan(
            x,
            z,
            selectivity["dt_combined"],
            selectivity["B_combined"],
            selectivity["C_combined"],
        )
    elif variant == "inv_only":
        hidden_out = block._run_selective_scan(
            x,
            z,
            selectivity["dt_inv"],
            selectivity["B_inv"],
            selectivity["C_inv"],
        )
    elif variant == "dt_only":
        hidden_out = block._run_selective_scan(
            x,
            z,
            selectivity["dt_combined"],
            selectivity["B_inv"],
            selectivity["C_inv"],
        )
    elif variant == "bc_only":
        hidden_out = block._run_selective_scan(
            x,
            z,
            selectivity["dt_inv"],
            selectivity["B_combined"],
            selectivity["C_combined"],
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    features = model._decode_sequence(hidden_out)
    return model.predict_from_features(features)


def evaluate_variant(model: nn.Module, loader, device: torch.device, variant: str) -> float:
    preds: list[np.ndarray] = []
    tgts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for windows, targets in loader:
            windows = windows.to(device, non_blocking=True)
            outputs = forward_variant(model, windows, variant).detach().cpu().numpy().astype(np.float32)
            preds.append(outputs)
            tgts.append(targets.numpy().astype(np.float32))
    return float(rmse(np.concatenate(preds), np.concatenate(tgts)))


def collect_gate_and_spec_usage(model: nn.Module, loader, device: torch.device, *, max_batches: int) -> dict[str, float]:
    gate_vals = []
    spec_ratios = []
    with torch.no_grad():
        for batch_idx, (windows, _) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            windows = windows.to(device, non_blocking=True)
            outputs = model.forward_features_with_aux(windows)
            gate = outputs["gate_sequence"].mean(dim=1).squeeze(-1)
            spec = outputs["specific_features"]
            feat = outputs["features"]
            spec_ratio = spec.norm(dim=1) / (feat.norm(dim=1) + 1e-8)
            gate_vals.append(gate.detach().cpu())
            spec_ratios.append(spec_ratio.detach().cpu())
    gate_all = torch.cat(gate_vals)
    ratio_all = torch.cat(spec_ratios)
    return {
        "gate_mean": float(gate_all.mean()),
        "gate_std": float(gate_all.std()),
        "spec_feature_ratio_mean": float(ratio_all.mean()),
        "spec_feature_ratio_std": float(ratio_all.std()),
    }


def select_trainable_params(model: nn.Module, mode: str) -> list[nn.Parameter]:
    named_params = list(model.named_parameters())
    if mode == "full":
        for _, param in named_params:
            param.requires_grad_(True)
        return [param for _, param in named_params]
    for name, param in named_params:
        allowed = False
        if mode == "head_only":
            allowed = name.startswith("head.")
        elif mode == "transformer_head":
            allowed = name.startswith("head.") or name.startswith("transformer_blocks.") or name.startswith("transformer_encoder.")
        elif mode == "spec_gate_head":
            allowed = name.startswith("head.") or ".x_proj_spec." in name or ".dt_proj_spec." in name or ".gate_proj" in name
        elif mode == "spec_gate_transformer_head":
            allowed = (
                name.startswith("head.")
                or name.startswith("transformer_blocks.")
                or name.startswith("transformer_encoder.")
                or ".x_proj_spec." in name
                or ".dt_proj_spec." in name
                or ".gate_proj" in name
            )
        elif mode == "mamba_only":
            allowed = not (name.startswith("head.") or name.startswith("transformer_blocks.") or name.startswith("transformer_encoder."))
        else:
            raise ValueError(f"Unknown adaptation mode: {mode}")
        param.requires_grad_(allowed)
    return [param for _, param in named_params if param.requires_grad]


def quick_adaptation_eval(
    base_args: Namespace,
    source_checkpoint: str | Path,
    source_loaders,
    target_loaders,
    device: torch.device,
    *,
    mode: str,
    domain_feature_tap: str,
    epochs: int,
    max_batches: int | None,
) -> dict[str, float]:
    args = copy.deepcopy(base_args)
    args.target_epochs = int(epochs)
    args.max_target_train_batches = max_batches
    args.domain_feature_tap = domain_feature_tap
    model = load_model(args, source_checkpoint, device)
    trainable = select_trainable_params(model, mode)
    stage_head = nn.Linear(model.head.in_features, args.num_pseudo_stages).to(device)
    optimizer = torch.optim.Adam(
        list(trainable) + list(stage_head.parameters()),
        lr=float(args.target_lr),
        weight_decay=float(args.target_weight_decay),
    )

    best_val_rmse = float("inf")
    best_test_rmse = float("inf")
    best_epoch = -1
    best_gate_mean = 0.0

    for epoch in range(1, epochs + 1):
        stage_quantile = resolve_pseudo_quantile(
            epoch,
            epochs,
            start_quantile=args.pseudo_start_quantile,
            end_quantile=args.pseudo_end_quantile,
        )
        stage_stats = collect_source_stage_statistics(
            model,
            source_loaders["train"],
            device,
            rul_clip=float(args.rul_clip),
            num_stages=args.num_pseudo_stages,
            quantile=stage_quantile,
            normalize_features=not args.disable_pseudo_feature_normalization,
            stage_feature_mode=str(args.stage_feature_mode),
        )
        train_stats = run_cd_pseudo_epoch(
            model,
            stage_head,
            None,
            source_loaders["train"],
            target_loaders["labeled"],
            target_loaders["unlabeled"],
            None,
            stage_stats,
            optimizer,
            device,
            source_loss_weight=args.source_loss_weight,
            target_loss_weight=args.target_loss_weight,
            lambda_mmd=args.lambda_mmd,
            lambda_source_stage=args.lambda_source_stage,
            lambda_pseudo=args.lambda_pseudo,
            lambda_contrastive=args.lambda_contrastive,
            lambda_monotonic=0.0,
            lambda_domain_adv=0.0,
            lambda_inv_mmd=args.lambda_inv_mmd if args.inv_alignment_mode == "mmd" else 0.0,
            lambda_conditional_inv_mmd=args.lambda_conditional_inv_mmd,
            lambda_conditional_proto=getattr(args, "lambda_conditional_proto", 0.0),
            lambda_conditional_proto_ce=getattr(args, "lambda_conditional_proto_ce", 0.0),
            lambda_inv_spec_orth=args.lambda_inv_spec_orth,
            lambda_spec_residual=args.lambda_spec_residual,
            lambda_inv_aux=args.lambda_inv_aux,
            lambda_spec_reconstruction=args.lambda_spec_reconstruction,
            rul_clip=float(args.rul_clip),
            num_pseudo_stages=args.num_pseudo_stages,
            target_scale=float(args.target_scale),
            grad_clip_norm=None,
            max_batches=max_batches,
            mmd_sigmas=tuple(float(token) for token in args.mmd_sigmas.split(",")),
            contrastive_temperature=float(args.contrastive_temperature),
            contrastive_normalize_features=not args.disable_contrastive_feature_normalization,
            monotonic_margin=float(args.monotonic_margin),
            grl_lambda=float(args.grl_lambda),
            inv_alignment_mode=str(args.inv_alignment_mode),
            domain_feature_tap=str(args.domain_feature_tap),
            stage_feature_mode=str(args.stage_feature_mode),
        )
        val_rmse = evaluate(model, target_loaders["val"], device, float(args.target_scale))["rmse"]
        if val_rmse < best_val_rmse:
            best_val_rmse = float(val_rmse)
            best_test_rmse = float(evaluate(model, target_loaders["test"], device, float(args.target_scale))["rmse"])
            best_epoch = epoch
            best_gate_mean = float(train_stats.get("gate_mean", 0.0))
    return {
        "mode": mode,
        "domain_feature_tap": domain_feature_tap,
        "best_epoch": int(best_epoch),
        "best_val_rmse": float(best_val_rmse),
        "best_test_rmse": float(best_test_rmse),
        "best_gate_mean": float(best_gate_mean),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mechanism-first Mamba diagnosis for cross-domain SPD runs.")
    parser.add_argument(
        "--run-root",
        default="/home/shelterpl/cd_mambatt/runs/cd_mambatt_v3_spd_fullmatch/FD001_TO_FD003",
        help="Run directory containing seed_xx/result.json files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-batches", type=int, default=6)
    parser.add_argument("--drift-batches", type=int, default=4)
    parser.add_argument("--quick-epochs", type=int, default=6)
    parser.add_argument("--quick-max-batches", type=int, default=40)
    parser.add_argument(
        "--output",
        default="/home/shelterpl/cd_mambatt/docs/generated/mamba_mechanism_diagnosis_seed42.json",
    )
    args_cli = parser.parse_args()

    set_seed(42)
    device = infer_device(args_cli.device)
    if device.type != "cuda":
        raise RuntimeError("This diagnosis expects CUDA.")

    run_root = Path(args_cli.run_root).expanduser().resolve()
    record = load_run_record(run_root, args_cli.seed)
    args = make_default_args(record)
    source_loaders, _, target_loaders, _ = build_loaders_from_record(args, record)

    source_checkpoint = record["source_stage"]["checkpoint"]
    cd_checkpoint = record["cd_stage"]["checkpoint"]
    source_model = load_model(args, source_checkpoint, device)
    adapted_model = load_model(args, cd_checkpoint, device)

    # Q1 + Q3: where domain information is strongest and whether dt or B/C look more domain-specific.
    src_views = collect_views(adapted_model, source_loaders["test"], device, max_batches=args_cli.feature_batches)
    tgt_views = collect_views(adapted_model, target_loaders["test"], device, max_batches=args_cli.feature_batches)
    view_probe = {
        name: linear_domain_probe(src_views[name], tgt_views[name])
        for name in [
            "x_conv",
            "z_raw",
            "gate",
            "dt_inv",
            "dt_spec",
            "B_inv",
            "B_spec",
            "C_inv",
            "C_spec",
            "D_x",
            "inv_core",
            "combined_core",
            "pre_transformer_last",
            "features",
            "invariant_features",
            "specific_features",
        ]
    }

    # Q4: drift amplification source
    src_steps = collect_stepwise_drift(adapted_model, source_loaders["test"], device, max_batches=args_cli.drift_batches)
    tgt_steps = collect_stepwise_drift(adapted_model, target_loaders["test"], device, max_batches=args_cli.drift_batches)
    drift = {name: stepwise_curve(src_steps[name], tgt_steps[name], device) for name in src_steps}

    # Q2: where target few-shot supervision sends gradient at adaptation start
    first_target_batch = next(iter(target_loaders["labeled"]))
    grad_windows = first_target_batch[0].to(device, non_blocking=True)
    grad_targets = first_target_batch[1].to(device, non_blocking=True) / float(args.target_scale)
    gradient_init = gradient_norm_report(source_model, grad_windows, grad_targets)
    gradient_adapted = gradient_norm_report(adapted_model, grad_windows, grad_targets)

    # Q5: whether spec is actually used and whether dt or bc carry useful calibration
    usage = {
        "source_test": collect_gate_and_spec_usage(adapted_model, source_loaders["test"], device, max_batches=args_cli.feature_batches),
        "target_test": collect_gate_and_spec_usage(adapted_model, target_loaders["test"], device, max_batches=args_cli.feature_batches),
    }
    component_ablation = {
        "source_test": {
            variant: evaluate_variant(adapted_model, source_loaders["test"], device, variant)
            for variant in ["full", "inv_only", "dt_only", "bc_only"]
        },
        "target_test": {
            variant: evaluate_variant(adapted_model, target_loaders["test"], device, variant)
            for variant in ["full", "inv_only", "dt_only", "bc_only"]
        },
    }

    # Q2 + Q6: quick adaptation mode and tap comparisons from the same source checkpoint
    quick_modes = {}
    for mode in ["head_only", "transformer_head", "spec_gate_head", "spec_gate_transformer_head", "full"]:
        quick_modes[mode] = quick_adaptation_eval(
            args,
            source_checkpoint,
            source_loaders,
            target_loaders,
            device,
            mode=mode,
            domain_feature_tap="inv_mean",
            epochs=args_cli.quick_epochs,
            max_batches=args_cli.quick_max_batches,
        )
    quick_taps = {}
    for tap in ["inv_mean", "pre_transformer_last", "concat_inv_pre"]:
        quick_taps[tap] = quick_adaptation_eval(
            args,
            source_checkpoint,
            source_loaders,
            target_loaders,
            device,
            mode="full",
            domain_feature_tap=tap,
            epochs=args_cli.quick_epochs,
            max_batches=args_cli.quick_max_batches,
        )

    result = {
        "run_root": str(run_root),
        "seed": int(args_cli.seed),
        "source_checkpoint": str(source_checkpoint),
        "cd_checkpoint": str(cd_checkpoint),
        "question_1_domain_entry_views": view_probe,
        "question_2_gradient_routing": {
            "source_init_target_supervised": gradient_init,
            "adapted_checkpoint_target_supervised": gradient_adapted,
            "quick_adaptation_modes": quick_modes,
        },
        "question_3_dt_vs_bc": {
            "view_probe_subset": {k: view_probe[k] for k in ["dt_inv", "dt_spec", "B_inv", "B_spec", "C_inv", "C_spec"]},
            "component_ablation": component_ablation,
        },
        "question_4_drift_amplification": drift,
        "question_5_spec_branch_usage": {
            "usage": usage,
            "component_ablation": component_ablation,
        },
        "question_6_mamba_vs_transformer_and_tap": {
            "quick_adaptation_modes": quick_modes,
            "quick_domain_taps": quick_taps,
        },
    }

    output_path = Path(args_cli.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
