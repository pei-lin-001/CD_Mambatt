from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def build_configs() -> list[dict[str, object]]:
    return [
        {
            "name": "baseline_d21_m1_t3_ff84",
            "d_model": 21,
            "num_mamba_layers": 1,
            "num_transformer_layers": 3,
            "dim_feedforward": 84,
            "num_heads": 7,
        },
        {
            "name": "mid_d42_m2_t4_ff168",
            "d_model": 42,
            "num_mamba_layers": 2,
            "num_transformer_layers": 4,
            "dim_feedforward": 168,
            "num_heads": 7,
        },
        {
            "name": "large_d84_m2_t4_ff336",
            "d_model": 84,
            "num_mamba_layers": 2,
            "num_transformer_layers": 4,
            "dim_feedforward": 336,
            "num_heads": 7,
        },
    ]


def run_config(
    repo_root: Path,
    args: argparse.Namespace,
    config: dict[str, object],
) -> dict[str, object]:
    output_dir = Path(args.output_root).expanduser().resolve() / str(config["name"])
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "python",
        "train_supervised.py",
        "--subset",
        args.subset,
        "--seeds",
        args.seeds,
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--transformer-norm-mode",
        args.transformer_norm_mode,
        "--dim-feedforward",
        str(config["dim_feedforward"]),
        "--d-model",
        str(config["d_model"]),
        "--num-mamba-layers",
        str(config["num_mamba_layers"]),
        "--num-transformer-layers",
        str(config["num_transformer_layers"]),
        "--num-heads",
        str(config["num_heads"]),
        "--dropout",
        str(args.dropout),
        "--transformer-inner-dropout",
        str(args.transformer_inner_dropout),
        "--device",
        str(args.device),
        "--output-dir",
        str(output_dir),
    ]
    if args.val_all_windows:
        cmd.append("--val-all-windows")
    if args.resample_split_per_seed:
        cmd.append("--resample-split-per-seed")

    print(json.dumps({"config": config["name"], "command": cmd}, ensure_ascii=False))
    subprocess.run(cmd, cwd=repo_root, check=True)

    summary_path = output_dir / args.subset.upper() / "summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    summary = payload["summary"]
    return {
        "name": config["name"],
        "summary_path": str(summary_path),
        "d_model": int(config["d_model"]),
        "num_mamba_layers": int(config["num_mamba_layers"]),
        "num_transformer_layers": int(config["num_transformer_layers"]),
        "dim_feedforward": int(config["dim_feedforward"]),
        "mean_test_rmse": float(summary["mean_test_rmse"]),
        "std_test_rmse": float(summary["std_test_rmse"]),
        "best_test_rmse": float(summary["best_test_rmse"]),
        "best_val_rmse": float(summary["best_val_rmse"]),
        "best_seed": int(summary["best_seed"]),
        "elapsed_seconds": float(summary["elapsed_seconds"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled supervised capacity sweep.")
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--transformer-inner-dropout", type=float, default=0.0)
    parser.add_argument("--transformer-norm-mode", default="post", choices=("pre", "post"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conda-env", default=os.environ.get("CONDA_DEFAULT_ENV", "cd_mamba"))
    parser.add_argument("--val-all-windows", action="store_true", default=True)
    parser.add_argument("--resample-split-per-seed", action="store_true")
    parser.add_argument(
        "--output-root",
        default="/home/shelterpl/cd_mambatt/runs/supervised_capacity_sweep",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results = [run_config(repo_root, args, config) for config in build_configs()]
    results.sort(key=lambda item: item["mean_test_rmse"])

    summary = {
        "subset": args.subset.upper(),
        "seeds": args.seeds,
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "transformer_norm_mode": str(args.transformer_norm_mode),
        "val_all_windows": bool(args.val_all_windows),
        "resample_split_per_seed": bool(args.resample_split_per_seed),
        "results": results,
    }
    summary_path = Path(args.output_root).expanduser().resolve() / f"{args.subset.upper()}_capacity_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
