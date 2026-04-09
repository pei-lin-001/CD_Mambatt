from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/shelterpl/cd_mambatt")
LOG_PATH = ROOT / "runs/logs/overnight_live_watch_20260408.md"
MASTER_LOG = ROOT / "runs/logs/overnight_architecture_search_20260408.log"
SUMMARY_TXT = ROOT / "runs/logs/overnight_architecture_search_20260408.summary.txt"

RUNS = [
    "cd_mambatt_v3_frontaffine_20260408",
    "cd_mambatt_v3_frontresidual_20260408",
    "cd_mambatt_v3_frontresidual_dualstate_20260408",
]
TASK = "FD001_TO_FD003"


def sh(command: str) -> str:
    return subprocess.run(command, shell=True, check=False, capture_output=True, text=True).stdout.strip()


def append(text: str) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def process_snapshot() -> str:
    return sh(
        "ps -eo pid,ppid,etime,%cpu,%mem,cmd --sort=start_time "
        "| grep -E 'run_overnight_architecture_search|frontaffine|frontresidual|train_cd_mambatt_v3|conda run -n cd_mamba' "
        "| grep -v grep"
    )


def latest_seed_status(run_name: str) -> list[str]:
    task_dir = ROOT / "runs" / run_name / TASK
    rows: list[str] = []
    if not task_dir.exists():
        return rows
    for seed_dir in sorted(task_dir.glob("seed_*")):
        result_path = seed_dir / "result.json"
        if result_path.exists():
            payload = json.loads(result_path.read_text())
            cd = payload["cd_stage"]
            rows.append(
                f"- {run_name} / {seed_dir.name}: test_rmse={cd['test_rmse']:.4f}, "
                f"val_rmse={cd['best_val_rmse']:.4f}, epoch={cd['best_epoch']}, "
                f"gate={cd.get('best_gate_mean', 0.0):.4f}"
            )
        else:
            rows.append(f"- {run_name} / {seed_dir.name}: running")
    return rows


def aggregate_status(run_name: str) -> str | None:
    summary_path = ROOT / "runs" / run_name / TASK / "summary.json"
    if not summary_path.exists():
        return None
    payload = json.loads(summary_path.read_text())["summary"]
    return (
        f"- {run_name} DONE: mean_rmse={payload['mean_cd_test_rmse']:.4f}, "
        f"std={payload['std_cd_test_rmse']:.4f}, best_test={payload['best_cd_test_rmse']:.4f}"
    )


def conditioning_status(run_name: str) -> str | None:
    path = ROOT / "runs/logs" / f"{run_name}.conditioning.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    rows = payload.get("rows", [])
    if not rows:
        return f"- {run_name} conditioning: empty"
    mean_tgt_delta = sum(float(row.get("tgt_pred_abs_delta_1_vs_0", 0.0)) for row in rows) / len(rows)
    mean_tgt_gate_gap = sum(
        abs(float(row.get("tgt_gate_label1", 0.0)) - float(row.get("tgt_gate_label0", 0.0))) for row in rows
    ) / len(rows)
    return (
        f"- {run_name} conditioning: mean_target_pred_delta={mean_tgt_delta:.6f}, "
        f"mean_target_gate_gap={mean_tgt_gate_gap:.6f}"
    )


def main() -> None:
    LOG_PATH.write_text("# Overnight Live Watch\n\n", encoding="utf-8")
    append(f"start_time={time.strftime('%F %T')}")
    seen_aggregates: set[str] = set()
    seen_conditioning: set[str] = set()

    while True:
        append("\n## tick " + time.strftime("%F %T"))
        ps_text = process_snapshot()
        append("### process")
        append("```")
        append(ps_text or "(no matching process)")
        append("```")

        append("### seed_progress")
        for run_name in RUNS:
            for row in latest_seed_status(run_name):
                append(row)

        for run_name in RUNS:
            agg = aggregate_status(run_name)
            if agg is not None and run_name not in seen_aggregates:
                append("### aggregate_ready")
                append(agg)
                seen_aggregates.add(run_name)
            cond = conditioning_status(run_name)
            if cond is not None and run_name not in seen_conditioning:
                append("### conditioning_ready")
                append(cond)
                seen_conditioning.add(run_name)

        if all(run_name in seen_aggregates for run_name in RUNS):
            append("\nall_runs_finished=true")
            break

        time.sleep(180)


if __name__ == "__main__":
    main()
