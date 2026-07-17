#!/usr/bin/env python3
"""
Plot ablation results: metrics + win/tie/loss + head weight heatmaps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hparam_plot_utils import (
    average_head_weights,
    bootstrap_ci,
    correct_map,
    extract_pred_target,
    load_jsonl,
    normalize_task,
    win_tie_loss,
)


def _find_latest_true_pred(root: Path) -> Path | None:
    candidates = [
        p for p in root.rglob("pred_LAMP_1_LLaMA3-8b-Instruct__DPSWeightedSoft.json")
        if "ablation_full/random_heads" not in str(p)
        if "ablation_full/random_mask" not in str(p)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_preds(path: Path):
    samples = load_jsonl(path)
    task = normalize_task(samples[0].get("task", "LaMP-1")) if samples else "LaMP-1"
    preds, targets = zip(*(extract_pred_target(s) for s in samples)) if samples else ([], [])
    return samples, task, list(preds), list(targets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ablation results")
    parser.add_argument("--true_pred", default=None)
    parser.add_argument("--random_heads_pred", default="/scratch/weixuz/dps/outputs/hparam/ablation_full/random_heads/pred_LAMP_1_LLaMA3-8b-Instruct__DPSWeightedSoft.json")
    parser.add_argument("--random_mask_pred", default="/scratch/weixuz/dps/outputs/hparam/ablation_full/random_mask/pred_LAMP_1_LLaMA3-8b-Instruct__DPSWeightedSoft.json")
    parser.add_argument("--true_heads_dir", default="/scratch/weixuz/dps/preference_head/cluster_heads/lamp1_k25_llama3-8b-instruct")
    parser.add_argument("--random_heads_dir", default="/scratch/weixuz/dps/preference_head/cluster_heads/lamp1_k25_llama3-8b-instruct_random")
    parser.add_argument("--random_mask_dir", default="/scratch/weixuz/dps/preference_head/cluster_heads/lamp1_k25_llama3-8b-instruct_random_mask")
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--out_dir", default="/scratch/weixuz/dps/outputs/hparam/figures/ablation")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.true_pred:
        true_pred = Path(args.true_pred)
    else:
        true_pred = _find_latest_true_pred(Path("/scratch/weixuz/dps/outputs"))
    if not true_pred or not true_pred.exists():
        raise FileNotFoundError("Could not find true-heads prediction file")

    true_samples, task, pred_t, tgt_t = _load_preds(true_pred)
    rand_samples, _, pred_r, tgt_r = _load_preds(Path(args.random_heads_pred))
    mask_samples, _, pred_m, tgt_m = _load_preds(Path(args.random_mask_pred))

    stats_true = bootstrap_ci(pred_t, tgt_t, task, num_samples=args.bootstrap)
    stats_rand = bootstrap_ci(pred_r, tgt_r, task, num_samples=args.bootstrap)
    stats_mask = bootstrap_ci(pred_m, tgt_m, task, num_samples=args.bootstrap)

    corr_true = correct_map(true_samples, task)
    corr_rand = correct_map(rand_samples, task)
    corr_mask = correct_map(mask_samples, task)

    win_r, tie_r, loss_r = win_tie_loss(corr_true, corr_rand)
    win_m, tie_m, loss_m = win_tie_loss(corr_true, corr_mask)

    # Metrics + W/T/L
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    labels = ["True", "Random weights", "Random mask"]
    acc_vals = [stats_true["acc"], stats_rand["acc"], stats_mask["acc"]]
    f1_vals = [stats_true["f1"], stats_rand["f1"], stats_mask["f1"]]
    acc_err = [
        [stats_true["acc"] - stats_true["acc_lo"], stats_rand["acc"] - stats_rand["acc_lo"], stats_mask["acc"] - stats_mask["acc_lo"]],
        [stats_true["acc_hi"] - stats_true["acc"], stats_rand["acc_hi"] - stats_rand["acc"], stats_mask["acc_hi"] - stats_mask["acc"]],
    ]
    f1_err = [
        [stats_true["f1"] - stats_true["f1_lo"], stats_rand["f1"] - stats_rand["f1_lo"], stats_mask["f1"] - stats_mask["f1_lo"]],
        [stats_true["f1_hi"] - stats_true["f1"], stats_rand["f1_hi"] - stats_rand["f1"], stats_mask["f1_hi"] - stats_mask["f1"]],
    ]
    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width / 2, acc_vals, width, yerr=acc_err, label="accuracy", color="#4C78A8")
    axes[0].bar(x + width / 2, f1_vals, width, yerr=f1_err, label="f1", color="#F58518")
    axes[0].set_title("Ablation metrics (LaMP-1)")
    axes[0].set_xticks(x, labels, rotation=15)
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    axes[1].bar(["Random weights"], [win_r], color="#4C78A8", label="win")
    axes[1].bar(["Random weights"], [tie_r], bottom=[win_r], color="#BAB0AC", label="tie")
    axes[1].bar(["Random weights"], [loss_r], bottom=[win_r + tie_r], color="#E45756", label="loss")
    axes[1].bar(["Random mask"], [win_m], color="#4C78A8")
    axes[1].bar(["Random mask"], [tie_m], bottom=[win_m], color="#BAB0AC")
    axes[1].bar(["Random mask"], [loss_m], bottom=[win_m + tie_m], color="#E45756")
    axes[1].set_title("Win/Tie/Loss vs True")
    axes[1].set_ylabel("count")
    axes[1].legend()

    fig.tight_layout()
    metrics_path = out_dir / "ablation_metrics.png"
    fig.savefig(metrics_path, dpi=200)
    print(f"Saved plot to {metrics_path}")

    # Head weight heatmaps
    true_w = average_head_weights(Path(args.true_heads_dir))
    rand_w = average_head_weights(Path(args.random_heads_dir))
    mask_w = average_head_weights(Path(args.random_mask_dir))

    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    im0 = axes2[0].imshow(true_w, cmap="YlGnBu")
    axes2[0].set_title("True heads")
    axes2[1].imshow(rand_w, cmap="YlGnBu")
    axes2[1].set_title("Random weights")
    axes2[2].imshow(mask_w, cmap="YlGnBu")
    axes2[2].set_title("Random mask")
    for ax in axes2:
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
    fig2.colorbar(im0, ax=axes2, location="right", shrink=0.85, pad=0.04)
    heat_path = out_dir / "ablation_head_heatmaps.png"
    fig2.savefig(heat_path, dpi=200)
    print(f"Saved plot to {heat_path}")


if __name__ == "__main__":
    main()
