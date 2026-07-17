#!/usr/bin/env python3
"""
Plot heads sweep: accuracy/F1 line plot + head-set overlap heatmap.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hparam_plot_utils import (
    average_head_weights,
    bootstrap_ci,
    extract_pred_target,
    jaccard,
    load_jsonl,
    normalize_task,
    topk_set,
)


def _parse_counts(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _find_pred_file(run_dir: Path) -> Path | None:
    preds = list(run_dir.glob("pred_LAMP_*__DPSWeightedSoft.json"))
    return preds[0] if preds else None


def _find_head_dir(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[-1] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot heads sweep results")
    parser.add_argument("--outputs_root", default="/scratch/weixuz/decore/outputs/hparam/heads_quick")
    parser.add_argument("--heads_root", default="/scratch/weixuz/preference_head/cluster_heads")
    parser.add_argument("--task", default="LaMP-1")
    parser.add_argument("--model_slug", default="llama3-8b-instruct")
    parser.add_argument("--head_counts", default="10,20,40,80,160")
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--out_dir", default="/scratch/weixuz/decore/outputs/hparam/figures/heads")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    heads_root = Path(args.heads_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    head_counts = _parse_counts(args.head_counts)

    acc_means = []
    acc_lo = []
    acc_hi = []
    f1_means = []
    f1_lo = []
    f1_hi = []

    for count in head_counts:
        run_dir = outputs_root / f"h{count}"
        pred_file = _find_pred_file(run_dir)
        if not pred_file:
            print(f"Missing prediction file for {run_dir}")
            acc_means.append(np.nan)
            acc_lo.append(np.nan)
            acc_hi.append(np.nan)
            f1_means.append(np.nan)
            f1_lo.append(np.nan)
            f1_hi.append(np.nan)
            continue

        samples = load_jsonl(pred_file)
        task = normalize_task(samples[0].get("task", args.task))
        preds, targets = zip(*(extract_pred_target(s) for s in samples))
        stats = bootstrap_ci(list(preds), list(targets), task, num_samples=args.bootstrap)
        acc_means.append(stats["acc"])
        acc_lo.append(stats["acc_lo"])
        acc_hi.append(stats["acc_hi"])
        f1_means.append(stats["f1"])
        f1_lo.append(stats["f1_lo"])
        f1_hi.append(stats["f1_hi"])

    # Head overlap heatmap
    head_sets = []
    for count in head_counts:
        pattern = f"lamp1_k*_{args.model_slug}_quick50_h{count}"
        head_dir = _find_head_dir(heads_root, pattern)
        if not head_dir:
            print(f"Missing head dir for {count} heads")
            head_sets.append(set())
            continue
        weights = average_head_weights(head_dir)
        head_sets.append(topk_set(weights, count))

    n = len(head_counts)
    jacc = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            jacc[i, j] = jaccard(head_sets[i], head_sets[j])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Line plot with error bars
    x = np.array(head_counts)
    axes[0].errorbar(
        x, acc_means, yerr=[np.array(acc_means) - np.array(acc_lo), np.array(acc_hi) - np.array(acc_means)],
        marker="o", label="accuracy", color="#4C78A8"
    )
    axes[0].errorbar(
        x, f1_means, yerr=[np.array(f1_means) - np.array(f1_lo), np.array(f1_hi) - np.array(f1_means)],
        marker="s", linestyle="--", label="f1", color="#F58518"
    )
    axes[0].set_title("Heads sweep")
    axes[0].set_xlabel("Number of heads")
    axes[0].set_ylabel("Metric")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Heatmap
    im = axes[1].imshow(jacc, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    axes[1].set_title("Head set overlap (Jaccard)")
    axes[1].set_xticks(range(n), [str(h) for h in head_counts])
    axes[1].set_yticks(range(n), [str(h) for h in head_counts])
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path = out_dir / "heads_sweep.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
