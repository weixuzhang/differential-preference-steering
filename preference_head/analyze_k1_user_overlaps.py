#!/usr/bin/env python3
"""
Analyze per-user (k=1-style) overlaps using PCS scores.

This script treats each PCS sample index as a user and computes:
  - Per-user top-head sets
  - Pairwise Jaccard overlaps
  - Overlap histogram and heatmap
  - Head frequency across users
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

Head = Tuple[int, int]


def load_pcs_matrix(pcs_file: Path) -> Tuple[List[Head], np.ndarray]:
    with pcs_file.open("r") as f:
        scores_dict = json.load(f)

    heads = []
    values = []
    for key in sorted(scores_dict.keys(), key=lambda k: tuple(map(int, k.split("-")))):
        layer, head = map(int, key.split("-"))
        heads.append((layer, head))
        values.append(scores_dict[key])

    lengths = {len(v) for v in values}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent PCS lengths found: {sorted(lengths)}")

    matrix = np.asarray(values, dtype=float)  # shape [num_heads, num_samples]
    return heads, matrix


def build_user_head_sets(
    heads: List[Head],
    pcs_matrix: np.ndarray,
    top_percent: float,
    top_k: int | None,
) -> List[set[Head]]:
    num_heads, num_samples = pcs_matrix.shape
    if top_k is None:
        top_k = max(1, int(num_heads * top_percent))

    user_sets = []
    for sample_idx in range(num_samples):
        scores = pcs_matrix[:, sample_idx]
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        user_sets.append({heads[i] for i in top_idx})
    return user_sets


def jaccard_matrix(sets: List[set[Head]]) -> np.ndarray:
    n = len(sets)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            union = sets[i] | sets[j]
            inter = sets[i] & sets[j]
            value = float(len(inter) / len(union)) if union else 0.0
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def write_csv(matrix: np.ndarray, labels: List[str], output_file: Path) -> None:
    with output_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [f"{v:.4f}" for v in row])


def plot_heatmap(
    matrix: np.ndarray,
    labels: List[str],
    output_file: Path,
    cmap: str,
    vmax: float | None,
    mask_diagonal: bool,
) -> None:
    num_users = len(labels)
    tick_step = max(1, num_users // 25)
    fig_width = max(12, 0.28 * num_users + 4.5)
    fig_height = max(10, 0.28 * num_users + 3.0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_matrix = matrix.copy()
    if mask_diagonal:
        np.fill_diagonal(plot_matrix, np.nan)
    if vmax is None:
        vmax = float(np.nanmax(plot_matrix)) if np.isfinite(plot_matrix).any() else 1.0

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#f0f0f0")
    masked = np.ma.masked_invalid(plot_matrix)
    im = ax.imshow(masked, cmap=cmap_obj, vmin=0.0, vmax=vmax, aspect="auto")

    tick_positions = np.arange(0, num_users, tick_step)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels([labels[i] for i in tick_positions], rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([labels[i] for i in tick_positions], fontsize=9)

    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.14, top=0.90)
    cax = fig.add_axes([0.90, 0.16, 0.02, 0.64])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Jaccard", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title("Pairwise Jaccard Overlap of Top-K Preference Head Sets Across Users", fontsize=16, fontweight="bold", pad=12)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_histogram(values: np.ndarray, output_file: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
    plt.title("Jaccard Overlap Distribution (User Pairs)", fontsize=12, fontweight="bold")
    plt.xlabel("Jaccard overlap")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def save_head_frequency(sets: List[set[Head]], output_file: Path) -> None:
    freq: Dict[Head, int] = {}
    for s in sets:
        for head in s:
            freq[head] = freq.get(head, 0) + 1

    rows = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    with output_file.open("w") as f:
        f.write("layer,head,count\n")
        for (layer, head), count in rows:
            f.write(f"{layer},{head},{count}\n")


def summarize_matrix(matrix: np.ndarray) -> Dict[str, float]:
    n = matrix.shape[0]
    values = [matrix[i, j] for i in range(n) for j in range(n) if i != j]
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze per-user overlaps from PCS.")
    parser.add_argument("--pcs_file", required=True, help="PCS JSON file path")
    parser.add_argument(
        "--top_percent",
        type=float,
        default=0.04,
        help="Top percent of heads per user (default: 0.04)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Optional fixed top-k per user (overrides top_percent)",
    )
    parser.add_argument(
        "--output_dir",
        default="./visualizations/k1_users",
        help="Output directory",
    )
    parser.add_argument(
        "--label_prefix",
        default="user",
        help="Prefix for user labels in CSV/plots",
    )
    parser.add_argument(
        "--heatmap_cmap",
        default="viridis",
        help="Colormap for the Jaccard heatmap.",
    )
    parser.add_argument(
        "--heatmap_vmax",
        type=float,
        default=None,
        help="Optional vmax for heatmap color scale (default: max off-diagonal).",
    )
    parser.add_argument(
        "--show_diagonal",
        action="store_false",
        dest="mask_diagonal",
        help="Show diagonal cells in the heatmap.",
    )
    parser.set_defaults(mask_diagonal=True)
    parser.add_argument(
        "--max_users",
        type=int,
        default=None,
        help="Optional: limit number of users for visualization.",
    )
    parser.add_argument(
        "--sample_method",
        choices=["random", "first", "stride"],
        default="random",
        help="Sampling method when --max_users is set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed used for sampling.",
    )
    args = parser.parse_args()

    pcs_file = Path(args.pcs_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    heads, pcs_matrix = load_pcs_matrix(pcs_file)
    user_sets = build_user_head_sets(heads, pcs_matrix, args.top_percent, args.top_k)

    if args.max_users is not None and args.max_users < len(user_sets):
        if args.sample_method == "first":
            indices = np.arange(args.max_users)
        elif args.sample_method == "stride":
            indices = np.linspace(0, len(user_sets) - 1, args.max_users, dtype=int)
            indices = np.unique(indices)
            if len(indices) < args.max_users:
                remaining = args.max_users - len(indices)
                extra = np.setdiff1d(np.arange(len(user_sets)), indices)[:remaining]
                indices = np.concatenate([indices, extra])
        else:
            rng = np.random.default_rng(args.seed)
            indices = rng.choice(len(user_sets), size=args.max_users, replace=False)
        user_sets = [user_sets[i] for i in indices]
        labels = [f"{args.label_prefix}_{i:02d}" for i in indices]
    else:
        labels = [f"{args.label_prefix}_{i:02d}" for i in range(len(user_sets))]
    matrix = jaccard_matrix(user_sets)

    write_csv(matrix, labels, output_dir / "user_jaccard.csv")
    if len(user_sets) <= 120:
        plot_heatmap(
            matrix,
            labels,
            output_dir / "user_jaccard_heatmap.png",
            cmap=args.heatmap_cmap,
            vmax=args.heatmap_vmax,
            mask_diagonal=args.mask_diagonal,
        )

    summary = summarize_matrix(matrix)
    with (output_dir / "user_jaccard_summary.txt").open("w") as f:
        f.write(f"num_users: {len(user_sets)}\n")
        f.write(f"min: {summary['min']:.4f}\n")
        f.write(f"mean: {summary['mean']:.4f}\n")
        f.write(f"max: {summary['max']:.4f}\n")

    values = matrix[np.triu_indices(matrix.shape[0], k=1)]
    if values.size:
        plot_histogram(values, output_dir / "user_jaccard_hist.png")

    save_head_frequency(user_sets, output_dir / "head_frequency.csv")

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
