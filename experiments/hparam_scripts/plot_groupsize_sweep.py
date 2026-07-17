#!/usr/bin/env python3
"""
Plot group-size sweep: accuracy/F1 line plot + cluster size distribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hparam_plot_utils import bootstrap_ci, extract_pred_target, load_jsonl, normalize_task


def _parse_sizes(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _find_pred_file(run_dir: Path) -> Path | None:
    preds = list(run_dir.glob("pred_LAMP_*__DPSWeightedSoft.json"))
    return preds[0] if preds else None


def _cluster_dirs(cluster_root: Path) -> List[Path]:
    return sorted([p for p in cluster_root.glob("lamp1_k*") if (p / "clusters.json").exists()])


def _build_group_map(cluster_root: Path) -> Dict[int, Path]:
    mapping: Dict[int, Path] = {}
    for cdir in _cluster_dirs(cluster_root):
        with (cdir / "clusters.json").open("r") as f:
            data = json_load(f)
        k = int(data.get("k", 0))
        n = int(data.get("num_samples", 0))
        if k <= 0 or n <= 0:
            continue
        group_size = int(round(n / k))
        if group_size not in mapping:
            mapping[group_size] = cdir
    return mapping


def json_load(fh):
    import json

    return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot group-size sweep results")
    parser.add_argument("--outputs_root", default="outputs/hparam/groupsize_quick")
    parser.add_argument("--cluster_root", default="results/preference_head/cluster_runs")
    parser.add_argument("--group_sizes", default="10,50,100,200,400")
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--out_dir", default="outputs/hparam/figures/groupsize")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    cluster_root = Path(args.cluster_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    group_sizes = _parse_sizes(args.group_sizes)
    group_map = _build_group_map(cluster_root)

    acc_means = []
    acc_lo = []
    acc_hi = []
    f1_means = []
    f1_lo = []
    f1_hi = []
    cluster_sizes_list: List[List[int]] = []

    for group_size in group_sizes:
        run_dir = outputs_root / f"g{group_size}"
        pred_file = _find_pred_file(run_dir)
        if not pred_file:
            print(f"Missing prediction file for {run_dir}")
            acc_means.append(np.nan)
            acc_lo.append(np.nan)
            acc_hi.append(np.nan)
            f1_means.append(np.nan)
            f1_lo.append(np.nan)
            f1_hi.append(np.nan)
        else:
            samples = load_jsonl(pred_file)
            task = normalize_task(samples[0].get("task", "LaMP-1"))
            preds, targets = zip(*(extract_pred_target(s) for s in samples))
            stats = bootstrap_ci(list(preds), list(targets), task, num_samples=args.bootstrap)
            acc_means.append(stats["acc"])
            acc_lo.append(stats["acc_lo"])
            acc_hi.append(stats["acc_hi"])
            f1_means.append(stats["f1"])
            f1_lo.append(stats["f1_lo"])
            f1_hi.append(stats["f1_hi"])

        cluster_dir = group_map.get(group_size)
        if cluster_dir and (cluster_dir / "clusters.json").exists():
            with (cluster_dir / "clusters.json").open("r") as f:
                data = json_load(f)
            cluster_sizes = data.get("cluster_sizes", [])
            cluster_sizes_list.append(cluster_sizes)
        else:
            cluster_sizes_list.append([])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x = np.array(group_sizes)
    axes[0].errorbar(
        x, acc_means, yerr=[np.array(acc_means) - np.array(acc_lo), np.array(acc_hi) - np.array(acc_means)],
        marker="o", label="accuracy", color="#4C78A8"
    )
    axes[0].errorbar(
        x, f1_means, yerr=[np.array(f1_means) - np.array(f1_lo), np.array(f1_hi) - np.array(f1_means)],
        marker="s", linestyle="--", label="f1", color="#F58518"
    )
    axes[0].set_title("Group size sweep (LaMP-1)")
    axes[0].set_xlabel("Target group size")
    axes[0].set_ylabel("Metric")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Cluster size distribution
    axes[1].boxplot(cluster_sizes_list, labels=[str(g) for g in group_sizes], showfliers=False)
    axes[1].set_title("Cluster size distribution")
    axes[1].set_xlabel("Target group size")
    axes[1].set_ylabel("Cluster size")

    fig.tight_layout()
    out_path = out_dir / "groupsize_sweep.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
