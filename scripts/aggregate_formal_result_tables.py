#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path("/home/shelterpl/cd_mambatt")
DOCS_DIR = REPO_ROOT / "docs"
GENERATED_DIR = DOCS_DIR / "generated"


TASKS = [
    {
        "task": "FD001_TO_FD003",
        "pretty": "FD001 → FD003",
        "cd_5shot": REPO_ROOT / "runs/cd_mambatt_bestcfg_fd001_to_fd003_5seeds/FD001_TO_FD003/summary.json",
        "cd_15shot": REPO_ROOT / "runs/cd_mambatt_bestcfg_fd001_to_fd003_15shot_5seeds/FD001_TO_FD003/summary.json",
        "fomln_15shot": REPO_ROOT / "runs/fomln_formal_mb256_unit15_5seeds_fd001_to_fd003/FD001_TO_FD003/summary_aggregated.json",
    },
    {
        "task": "FD003_TO_FD001",
        "pretty": "FD003 → FD001",
        "cd_5shot": REPO_ROOT / "runs/cd_mambatt_bestcfg_fd003_to_fd001_5seeds/FD003_TO_FD001/summary.json",
        "cd_15shot": REPO_ROOT / "runs/cd_mambatt_bestcfg_fd003_to_fd001_15shot_5seeds/FD003_TO_FD001/summary.json",
        "fomln_15shot": REPO_ROOT / "runs/fomln_formal_mb256_unit15_5seeds_fd003_to_fd001/FD003_TO_FD001/summary_aggregated.json",
    },
    {
        "task": "FD002_TO_FD004",
        "pretty": "FD002 → FD004",
        "cd_5shot": REPO_ROOT / "runs/cd_mambatt_bestcfg_fd002_to_fd004_5seeds_fixcond/FD002_TO_FD004/summary.json",
        "cd_15shot": REPO_ROOT / "runs/cd_mambatt_bestcfg_fd002_to_fd004_15shot_5seeds/FD002_TO_FD004/summary.json",
        "fomln_15shot": REPO_ROOT / "runs/fomln_formal_mb256_unit15_5seeds_fd002_to_fd004/FD002_TO_FD004/summary_aggregated.json",
    },
    {
        "task": "FD001_TO_FD004",
        "pretty": "FD001 → FD004",
        "cd_5shot": REPO_ROOT / "runs/cd_mambatt_bestcfg_fd001_to_fd004_5seeds_fixcond/FD001_TO_FD004/summary.json",
        "cd_15shot": REPO_ROOT / "runs/cd_mambatt_bestcfg_fd001_to_fd004_15shot_5seeds/FD001_TO_FD004/summary.json",
        "fomln_15shot": REPO_ROOT / "runs/fomln_formal_mb256_unit15_5seeds_fd001_to_fd004/FD001_TO_FD004/summary_aggregated.json",
    },
]


def load_summary(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["summary"]


def fmt(value: float) -> str:
    return f"{value:.4f}"


def make_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    align = "|" + "|".join(["---"] * len(headers)) + "|"
    head = "|" + "|".join(headers) + "|"
    body = "\n".join("|" + "|".join(row) + "|" for row in rows)
    return "\n".join([head, align, body])


def main() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    fair_rows: list[dict[str, object]] = []
    efficiency_rows: list[dict[str, object]] = []

    for config in TASKS:
        cd5 = load_summary(config["cd_5shot"])
        cd15 = load_summary(config["cd_15shot"])
        fomln = load_summary(config["fomln_15shot"])

        fair_rows.append(
            {
                "task": config["pretty"],
                "cd_mambatt_15shot_mean_rmse": cd15["mean_cd_test_rmse"],
                "fomln_15shot_mean_rmse": fomln["mean_adapted_target_rmse"],
                "delta_cd_minus_fomln": cd15["mean_cd_test_rmse"] - fomln["mean_adapted_target_rmse"],
                "cd_mambatt_15shot_mean_direct_rmse": cd15["mean_direct_target_rmse"],
                "fomln_15shot_mean_direct_rmse": fomln["mean_direct_target_rmse"],
            }
        )

        efficiency_rows.append(
            {
                "task": config["pretty"],
                "cd_mambatt_5shot_mean_rmse": cd5["mean_cd_test_rmse"],
                "cd_mambatt_15shot_mean_rmse": cd15["mean_cd_test_rmse"],
                "fomln_15shot_mean_rmse": fomln["mean_adapted_target_rmse"],
                "cd_mambatt_5shot_mean_direct_rmse": cd5["mean_direct_target_rmse"],
                "cd_mambatt_15shot_mean_direct_rmse": cd15["mean_direct_target_rmse"],
                "cd_5shot_gain_over_direct": cd5["mean_direct_target_rmse"] - cd5["mean_cd_test_rmse"],
                "cd_15shot_gain_over_direct": cd15["mean_direct_target_rmse"] - cd15["mean_cd_test_rmse"],
                "cd_15shot_minus_fomln_15shot": cd15["mean_cd_test_rmse"] - fomln["mean_adapted_target_rmse"],
            }
        )

    fair_csv = GENERATED_DIR / "fair_comparison_cd_mambatt15_vs_fomln15.csv"
    with fair_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fair_rows)

    efficiency_csv = GENERATED_DIR / "data_efficiency_snapshot_cd5_cd15_fomln15.csv"
    with efficiency_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(efficiency_rows[0].keys()))
        writer.writeheader()
        writer.writerows(efficiency_rows)

    fair_table = make_markdown_table(
        ["Task", "CD-MambAtt 15-shot", "FOMLN 15-shot", "Δ (CD - FOMLN)"],
        [
            [
                row["task"],
                fmt(float(row["cd_mambatt_15shot_mean_rmse"])),
                fmt(float(row["fomln_15shot_mean_rmse"])),
                fmt(float(row["delta_cd_minus_fomln"])),
            ]
            for row in fair_rows
        ],
    )
    efficiency_table = make_markdown_table(
        [
            "Task",
            "CD 5-shot",
            "CD 15-shot",
            "FOMLN 15-shot",
            "CD 5-shot gain",
            "CD 15-shot gain",
        ],
        [
            [
                row["task"],
                fmt(float(row["cd_mambatt_5shot_mean_rmse"])),
                fmt(float(row["cd_mambatt_15shot_mean_rmse"])),
                fmt(float(row["fomln_15shot_mean_rmse"])),
                fmt(float(row["cd_5shot_gain_over_direct"])),
                fmt(float(row["cd_15shot_gain_over_direct"])),
            ]
            for row in efficiency_rows
        ],
    )

    summary_json = GENERATED_DIR / "formal_result_snapshot.json"
    summary_json.write_text(
        json.dumps(
            {
                "fair_comparison": fair_rows,
                "data_efficiency_snapshot": efficiency_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    markdown_path = GENERATED_DIR / "formal_result_tables.md"
    markdown_path.write_text(
        "\n".join(
            [
                "# Generated Formal Result Tables",
                "",
                "## Fair comparison: CD-MambAtt 15-shot vs FOMLN 15-shot",
                "",
                fair_table,
                "",
                "## Data-efficiency snapshot",
                "",
                efficiency_table,
                "",
                "Generated by `scripts/aggregate_formal_result_tables.py`.",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote {fair_csv}")
    print(f"Wrote {efficiency_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {markdown_path}")
    print()
    print(fair_table)
    print()
    print(efficiency_table)


if __name__ == "__main__":
    main()
