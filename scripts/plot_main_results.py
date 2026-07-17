#!/usr/bin/env python3
"""Plot primary-metric comparisons for main experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODEL_ALIASES = {
    "Mistral-7b-Instruct": "Mistral-7B-Instruct-v0.3",
    "Qwen2-7b-Instruct": "Qwen2-7B-Instruct",
}

MODELS = [
    "LLaMA3-8b-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Qwen2-7B-Instruct",
]

METHODS = [
    "Baseline",
    "ContextAwareDecoding",
    "DeCoReVanilla",
    "DPSWeightedSoft",
    "DoLa",
]

PRIMARY_METRIC = {
    "LaMP-1": ("accuracy", "Accuracy"),
    "LaMP-2": ("accuracy", "Accuracy"),
    "LaMP-3": ("mae", "MAE (lower is better)"),
    "LaMP-4": ("rouge-L", "ROUGE-L"),
    "LaMP-5": ("rouge-L", "ROUGE-L"),
    "LaMP-7": ("rouge-L", "ROUGE-L"),
}

COLORS = {
    "Baseline": "#8c8c8c",
    "ContextAwareDecoding": "#4c78a8",
    "DeCoReVanilla": "#f58518",
    "DPSWeightedSoft": "#54a24b",
    "DoLa": "#b279a2",
}


def load_summary(path: Path) -> dict[tuple[str, str, str], dict]:
    raw = json.loads(path.read_text())
    out = {}
    for key, val in raw.items():
        task, model, method = key.split("|", 2)
        model = MODEL_ALIASES.get(model, model)
        out[(task, model, method)] = val
    return out


def plot_task(task: str, summary: dict, out_dir: Path) -> None:
    metric_key, metric_label = PRIMARY_METRIC[task]
    fig, axes = plt.subplots(1, len(MODELS), figsize=(14, 4), sharey=True)

    for ax, model in zip(axes, MODELS):
        values = []
        missing_idx = []
        for i, method in enumerate(METHODS):
            entry = summary.get((task, model, method))
            if entry is None or metric_key not in entry:
                values.append(np.nan)
                missing_idx.append(i)
                continue
            values.append(entry[metric_key])

        bars = ax.bar(range(len(METHODS)), values, color=[COLORS[m] for m in METHODS])
        for i in missing_idx:
            ax.text(i, 0, "NA", ha="center", va="bottom", fontsize=8, color="#aa0000", rotation=90)
            bars[i].set_hatch("//")
            bars[i].set_edgecolor("#aa0000")
            bars[i].set_facecolor("#ffffff")

        ax.set_title(model, fontsize=10)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(METHODS, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[0].set_ylabel(metric_label)
    fig.suptitle(f"{task} | Primary Metric", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / f"{task.replace('-', '').lower()}_primary_metric.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/evaluation_summary_combined.json"),
        help="Path to evaluation summary JSON.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("decore/plots/main_results"),
        help="Output directory for plots.",
    )
    args = parser.parse_args()

    summary = load_summary(args.summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for task in PRIMARY_METRIC:
        plot_task(task, summary, args.out_dir)


if __name__ == "__main__":
    main()
