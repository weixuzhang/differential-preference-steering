#!/usr/bin/env python3
"""Plot performance vs compute overhead tradeoff for decoding methods."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRIMARY_METRIC = {
    "LaMP-1": ("accuracy", True),
    "LaMP-2": ("accuracy", True),
    "LaMP-3": ("mae", False),
    "LaMP-4": ("rouge-L", True),
    "LaMP-5": ("rouge-L", True),
    "LaMP-7": ("rouge-L", True),
}

METHODS = [
    "Baseline",
    "ContextAwareDecoding",
    "DeCoReVanilla",
    "DPSWeightedSoft",
    "DoLa",
]

OVERHEAD = {
    "Baseline": 1.0,
    "ContextAwareDecoding": 2.1,  # extra prefill vs baseline
    "DeCoReVanilla": 2.0,
    "DPSWeightedSoft": 2.0,
    "DoLa": 1.2,
}

COLORS = {
    "Baseline": "#9ecae1",
    "ContextAwareDecoding": "#9e9ac8",
    "DeCoReVanilla": "#a1d99b",
    "DPSWeightedSoft": "#74c476",
    "DoLa": "#fdae6b",
}


def load_summary(path: Path) -> dict[tuple[str, str, str], dict]:
    data = json.loads(path.read_text())
    out = {}
    for key, val in data.items():
        task, model, method = key.split("|", 2)
        out[(task, model, method)] = val
    return out


def compute_ratio(summary: dict, model: str) -> dict[str, float]:
    tasks = list(PRIMARY_METRIC.keys())
    baseline = {}
    for task in tasks:
        metric, higher = PRIMARY_METRIC[task]
        entry = summary.get((task, model, "Baseline"))
        if entry is None:
            continue
        baseline[task] = entry.get(metric)

    ratios = {}
    for method in METHODS:
        vals = []
        for task in tasks:
            metric, higher = PRIMARY_METRIC[task]
            base = baseline.get(task)
            entry = summary.get((task, model, method))
            if base is None or entry is None:
                continue
            val = entry.get(metric)
            if val is None:
                continue
            if higher:
                ratio = val / base if base else 1.0
            else:
                ratio = base / val if val else 1.0
            vals.append(ratio)
        if vals:
            ratios[method] = sum(vals) / len(vals)
    return ratios


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot performance vs overhead")
    parser.add_argument(
        "--summary",
        default="/scratch/weixuz/outputs/evaluation_summary_combined.json",
    )
    parser.add_argument(
        "--model",
        default="LLaMA3-8b-Instruct",
    )
    parser.add_argument(
        "--out",
        default="/scratch/weixuz/dps/plots/efficiency/efficiency_tradeoff.png",
    )
    args = parser.parse_args()

    summary = load_summary(Path(args.summary))
    ratios = compute_ratio(summary, args.model)

    # Prepare data
    xs = []
    ys = []
    labels = []
    colors = []
    for method in METHODS:
        if method not in ratios:
            continue
        xs.append(OVERHEAD.get(method, 1.0))
        ys.append(ratios[method])
        labels.append(method)
        colors.append(COLORS.get(method, "#bdbdbd"))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, s=120, c=colors, edgecolors="#333333", linewidths=0.6)
    for x, y, label in zip(xs, ys, labels):
        plt.text(x + 0.02, y + 0.002, label, fontsize=9)
    plt.xlabel("Relative compute (decode-time multiplier)")
    plt.ylabel("Avg normalized primary metric (baseline=1.0)")
    plt.title("Performance vs Compute Overhead (LLaMA3, LaMP dev)")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
