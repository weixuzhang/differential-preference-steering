#!/usr/bin/env python3
"""
Plot gamma sweep: metrics + alpha distribution + win/tie/loss.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hparam_plot_utils import (
    bootstrap_ci,
    correct_map,
    extract_pred_target,
    load_jsonl,
    normalize_task,
    win_tie_loss,
)


def _load_preds(path: Path):
    samples = load_jsonl(path)
    task = normalize_task(samples[0].get("task", "LaMP-1")) if samples else "LaMP-1"
    preds, targets = zip(*(extract_pred_target(s) for s in samples)) if samples else ([], [])
    return samples, task, list(preds), list(targets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot gamma sweep results")
    parser.add_argument("--adaptive_pred", default="/scratch/weixuz/dps/outputs/hparam/gamma_full/adaptive/pred_LAMP_1_LLaMA3-8b-Instruct__DPSWeightedSoft.json")
    parser.add_argument("--fixed_pred", default="/scratch/weixuz/dps/outputs/hparam/gamma_full/fixed_alpha_0.5/pred_LAMP_1_LLaMA3-8b-Instruct__DPSWeightedSoft.json")
    parser.add_argument("--fixed_alpha", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--out_dir", default="/scratch/weixuz/dps/outputs/hparam/figures/gamma")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adaptive_samples, task, pred_a, tgt_a = _load_preds(Path(args.adaptive_pred))
    fixed_samples, _, pred_f, tgt_f = _load_preds(Path(args.fixed_pred))

    stats_a = bootstrap_ci(pred_a, tgt_a, task, num_samples=args.bootstrap)
    stats_f = bootstrap_ci(pred_f, tgt_f, task, num_samples=args.bootstrap)

    # Alpha distribution
    alphas = []
    for s in adaptive_samples:
        alpha_seq = s.get("alphas", [])
        if isinstance(alpha_seq, list):
            alphas.extend(alpha_seq)

    # Win/tie/loss
    corr_a = correct_map(adaptive_samples, task)
    corr_f = correct_map(fixed_samples, task)
    win, tie, loss = win_tie_loss(corr_a, corr_f)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Metrics bar
    labels = ["Adaptive", "Fixed"]
    acc_vals = [stats_a["acc"], stats_f["acc"]]
    f1_vals = [stats_a["f1"], stats_f["f1"]]
    acc_err = [
        [stats_a["acc"] - stats_a["acc_lo"], stats_f["acc"] - stats_f["acc_lo"]],
        [stats_a["acc_hi"] - stats_a["acc"], stats_f["acc_hi"] - stats_f["acc"]],
    ]
    f1_err = [
        [stats_a["f1"] - stats_a["f1_lo"], stats_f["f1"] - stats_f["f1_lo"]],
        [stats_a["f1_hi"] - stats_a["f1"], stats_f["f1_hi"] - stats_f["f1"]],
    ]
    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width / 2, acc_vals, width, yerr=acc_err, label="accuracy", color="#4C78A8")
    axes[0].bar(x + width / 2, f1_vals, width, yerr=f1_err, label="f1", color="#F58518")
    axes[0].set_title("Gamma sweep metrics")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    # Alpha histogram
    axes[1].hist(alphas, bins=30, color="#72B7B2", alpha=0.8)
    axes[1].axvline(args.fixed_alpha, color="#E45756", linestyle="--", label=f"fixed={args.fixed_alpha}")
    axes[1].set_title("Adaptive alpha distribution")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("count")
    axes[1].legend()

    # Win/tie/loss
    axes[2].bar(["Adaptive vs Fixed"], [win], color="#4C78A8", label="win")
    axes[2].bar(["Adaptive vs Fixed"], [tie], bottom=[win], color="#BAB0AC", label="tie")
    axes[2].bar(["Adaptive vs Fixed"], [loss], bottom=[win + tie], color="#E45756", label="loss")
    axes[2].set_title("Win/Tie/Loss (accuracy)")
    axes[2].set_ylabel("count")
    axes[2].legend()

    fig.tight_layout()
    out_path = out_dir / "gamma_sweep.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
