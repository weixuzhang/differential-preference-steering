#!/usr/bin/env python3
"""
Evaluate and plot routing ablation (soft vs hard).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

sys.path.append("/scratch/weixuz/decore")
from evaluate_predictions import evaluate_predictions


def _scan_preds(root: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in root.rglob("pred_LAMP_*__DPSWeighted*.json")
            if not p.name.endswith("_eval.json")
        ]
    )


def _metric_group(task: str) -> str:
    if task in {"LaMP-1", "LaMP-2"}:
        return "classification"
    if task in {"LaMP-3"}:
        return "regression"
    return "generation"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot routing ablation results")
    parser.add_argument(
        "--root_dir",
        default="/scratch/weixuz/decore/outputs/hparam/routing_ablation",
    )
    parser.add_argument(
        "--output_tsv",
        default=None,
    )
    parser.add_argument(
        "--plot_png",
        default=None,
    )
    args = parser.parse_args()

    root = Path(args.root_dir)
    pred_files = _scan_preds(root)
    if not pred_files:
        print(f"No prediction files found in {root}")
        return

    rows: List[Dict] = []
    for pred in pred_files:
        results = evaluate_predictions(str(pred))
        task = results.get("task")
        routing = pred.parent.name  # soft/hard
        row = {
            "task": task,
            "routing": routing,
            "samples": results.get("total_samples"),
            "accuracy": results.get("accuracy"),
            "f1": results.get("f1"),
            "mae": results.get("mae"),
            "rmse": results.get("rmse"),
            "rouge-1": results.get("rouge-1"),
            "rouge-L": results.get("rouge-L"),
            "meteor": results.get("meteor"),
            "pred_file": str(pred),
        }
        rows.append(row)

    output_tsv = (
        Path(args.output_tsv) if args.output_tsv else root / "summary.tsv"
    )
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "task",
        "routing",
        "samples",
        "accuracy",
        "f1",
        "mae",
        "rmse",
        "rouge-1",
        "rouge-L",
        "meteor",
        "pred_file",
    ]
    with output_tsv.open("w") as f:
        f.write("\t".join(headers) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(h, "")) for h in headers) + "\n")

    # Build plot: only classification + generation
    group_metrics = {
        "classification": ("accuracy", "Accuracy"),
        "generation": ("rouge-L", "ROUGE-L"),
    }

    grouped: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in rows:
        task = row["task"]
        routing = row["routing"]
        group = _metric_group(task)
        grouped.setdefault(group, {}).setdefault(task, {})[routing] = row

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (group, (metric, title)) in zip(axes, group_metrics.items()):
        tasks_all = sorted(grouped.get(group, {}).keys())
        tasks = []
        soft_vals = []
        hard_vals = []
        for task in tasks_all:
            soft_val = grouped[group][task].get("soft", {}).get(metric)
            hard_val = grouped[group][task].get("hard", {}).get(metric)
            if soft_val is None or hard_val is None:
                continue
            tasks.append(task)
            soft_vals.append(soft_val)
            hard_vals.append(hard_val)
        if not tasks:
            ax.axis("off")
            ax.set_title(f"{title} (no paired data)")
            continue
        x = range(len(tasks))

        width = 0.35
        ax.bar([i - width / 2 for i in x], soft_vals, width, label="soft", color="#A6CEE3")
        ax.bar([i + width / 2 for i in x], hard_vals, width, label="hard", color="#FDBF6F")
        ax.set_xticks(list(x), tasks, rotation=15)
        ax.set_title(title)
        ax.legend()
        ax.set_ylabel(metric)
    fig.tight_layout()
    plot_path = Path(args.plot_png) if args.plot_png else root / "routing_ablation.png"
    fig.savefig(plot_path, dpi=200)
    print(f"Saved summary to: {output_tsv}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
