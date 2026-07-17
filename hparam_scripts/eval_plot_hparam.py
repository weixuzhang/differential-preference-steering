#!/usr/bin/env python3
"""
Evaluate hparam sweep predictions and (optionally) plot a metric.

Example:
  python decore/hparam_scripts/eval_plot_hparam.py \
    --root_dir /scratch/weixuz/decore/outputs/hparam/heads_quick \
    --metric accuracy
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import sys

sys.path.append("/scratch/weixuz/decore")
from evaluate_predictions import evaluate_predictions


def _infer_param(name: str):
    if name.startswith(("h", "g")) and name[1:].isdigit():
        return int(name[1:])
    return name


def _sort_key(param):
    if isinstance(param, int):
        return (0, param)
    return (1, str(param))


def _pick_metric(rows: List[Dict], requested: str | None) -> str:
    if requested:
        return requested
    for row in rows:
        for key in ("accuracy", "f1", "mae", "rmse", "rouge-L", "meteor"):
            if row.get(key) is not None:
                return key
    return "accuracy"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and plot hparam sweeps")
    parser.add_argument("--root_dir", required=True, help="Root directory with pred_*.json")
    parser.add_argument("--metric", default=None, help="Metric to plot (default: auto)")
    parser.add_argument("--output_tsv", default=None, help="Output TSV path")
    parser.add_argument("--plot_png", default=None, help="Output plot path (PNG)")
    args = parser.parse_args()

    root = Path(args.root_dir)
    pred_files = [
        p
        for p in root.rglob("pred_LAMP_*.json")
        if not p.name.endswith("_eval.json")
    ]
    if not pred_files:
        print(f"No prediction files found in {root}")
        return

    rows: List[Dict] = []
    for pred in sorted(pred_files):
        param = _infer_param(pred.parent.name)
        results = evaluate_predictions(str(pred))
        row = {
            "param": param,
            "task": results.get("task"),
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

    rows.sort(key=lambda r: _sort_key(r["param"]))
    output_tsv = Path(args.output_tsv) if args.output_tsv else root / "summary.tsv"
    output_tsv.parent.mkdir(parents=True, exist_ok=True)

    with output_tsv.open("w") as f:
        headers = [
            "param",
            "task",
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
        f.write("\t".join(headers) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(h, "")) for h in headers) + "\n")

    metric = _pick_metric(rows, args.metric)
    print(f"Saved summary to: {output_tsv}")
    print(f"Plot metric: {metric}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable, skipping plot: {exc}")
        return

    params = [str(r["param"]) for r in rows]
    values = [r.get(metric) for r in rows]
    values = [0.0 if v is None else float(v) for v in values]

    plt.figure(figsize=(8, 4))
    plt.bar(params, values, color="#4C78A8")
    plt.title(f"{root.name} | {metric}")
    plt.xlabel("Setting")
    plt.ylabel(metric)
    plt.tight_layout()

    plot_path = Path(args.plot_png) if args.plot_png else root / f"plot_{metric}.png"
    plt.savefig(plot_path, dpi=200)
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
