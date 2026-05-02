from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_targeted_adaptation_experiment import args_from_record
from train_cd_mambatt_v3 import uses_domain_conditioning
from train_cross_domain_baseline import build_source_stage_data, build_target_direct_test_loader
from train_supervised import build_model, evaluate, infer_device, set_seed
from cd_mambatt.data import load_cmapss_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple source checkpoints under one fixed source split and target direct test context."
    )
    parser.add_argument("--reference-result-json", required=True)
    parser.add_argument("--checkpoint-path", action="append", required=True)
    parser.add_argument("--source-split-path", default=None)
    parser.add_argument("--target-partition-path", default=None)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args_cli = parse_args()
    set_seed(int(args_cli.seed))
    device = infer_device(args_cli.device)

    reference_path = Path(args_cli.reference_result_json).expanduser().resolve()
    record = json.loads(reference_path.read_text(encoding="utf-8"))
    args = args_from_record(record)
    if args_cli.source_split_path is not None:
        args.source_split_path = str(Path(args_cli.source_split_path).expanduser().resolve())
    if args_cli.target_partition_path is not None:
        args.target_partition_path = str(Path(args_cli.target_partition_path).expanduser().resolve())

    source_train_full = load_cmapss_split(args.root, args.source_subset, "train", rul_clip=args.rul_clip)
    source_test_raw = load_cmapss_split(args.root, args.source_subset, "test", rul_clip=args.rul_clip)
    target_train_full = load_cmapss_split(args.root, args.target_subset, "train", rul_clip=args.rul_clip)
    target_test_raw = load_cmapss_split(args.root, args.target_subset, "test", rul_clip=args.rul_clip)

    with open(args.source_split_path, "r", encoding="utf-8") as handle:
        source_split = json.load(handle)

    source_loaders, source_meta = build_source_stage_data(args, source_train_full, source_test_raw, source_split)
    target_direct_loader, target_direct_meta = build_target_direct_test_loader(args, target_train_full, target_test_raw)
    input_dim = int(source_meta["train_windows_shape"][2])

    results: list[dict[str, object]] = []
    for checkpoint_text in args_cli.checkpoint_path:
        checkpoint_path = Path(checkpoint_text).expanduser().resolve()
        model = build_model(args, input_dim).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        source_metrics = evaluate(
            model,
            source_loaders["test"],
            device,
            float(args.target_scale),
            domain_label_value=0 if uses_domain_conditioning(model) else None,
        )
        target_direct_metrics = evaluate(
            model,
            target_direct_loader,
            device,
            float(args.target_scale),
            domain_label_value=1 if uses_domain_conditioning(model) else None,
        )
        result = {
            "checkpoint_path": str(checkpoint_path),
            "load_missing_keys": list(missing),
            "load_unexpected_keys": list(unexpected),
            "source_test": source_metrics,
            "target_direct": target_direct_metrics,
            "checkpoint_best_epoch": checkpoint.get("best_epoch"),
            "checkpoint_best_val_rmse": checkpoint.get("best_val_rmse"),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    payload = {
        "reference_result_json": str(reference_path),
        "seed": int(args_cli.seed),
        "device": str(device),
        "source_split_path": str(args.source_split_path),
        "target_partition_path": str(args.target_partition_path),
        "source_meta": source_meta,
        "target_direct_meta": target_direct_meta,
        "results": results,
    }
    output_path = Path(args_cli.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
