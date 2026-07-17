#!/usr/bin/env python3
"""
Plot PCS heatmap grid for selected users (k=1-style) from a single PCS file.

Each panel is layer x head for one user/sample index.
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

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


def head_to_matrix(
    heads: List[Head],
    values: np.ndarray,
    num_layers: int,
    num_heads: int,
) -> np.ndarray:
    mat = np.full((num_layers, num_heads), np.nan, dtype=float)
    for (layer, head), score in zip(heads, values):
        mat[layer, head] = score
    return mat


def select_user_indices(
    num_users: int,
    max_users: int,
    method: str,
    seed: int,
) -> np.ndarray:
    if max_users >= num_users:
        return np.arange(num_users)
    if method == "first":
        return np.arange(max_users)
    if method == "stride":
        indices = np.linspace(0, num_users - 1, max_users, dtype=int)
        return np.unique(indices)
    rng = np.random.default_rng(seed)
    return rng.choice(num_users, size=max_users, replace=False)


def plot_grid(
    matrices: List[np.ndarray],
    labels: List[str],
    output_file: Path,
    cmap: str,
    cols: int,
    title_fontsize: int = 16,
    axis_label_fontsize: int = 12,
    suptitle_fontsize: int = 20,
    cbar_label_fontsize: int = 12,
) -> None:
    n = len(matrices)
    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5.6 * cols + 1.2, 4.8 * rows),
        squeeze=False,
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.90,
        bottom=0.08,
        top=0.88,
        wspace=0.18,
        hspace=0.28,
    )

    all_vals = np.concatenate([m[~np.isnan(m)].ravel() for m in matrices])
    vmin = float(np.min(all_vals)) if all_vals.size else 0.0
    vmax = float(np.max(all_vals)) if all_vals.size else 1.0

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#f0f0f0")

    for idx, (mat, label) in enumerate(zip(matrices, labels)):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, cmap=cmap_obj, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(label, fontsize=title_fontsize, fontweight="bold", pad=8)
        ax.set_xlabel("Head Index", fontsize=axis_label_fontsize)
        ax.set_ylabel("Layer Index", fontsize=axis_label_fontsize)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    cax = fig.add_axes([0.92, 0.16, 0.018, 0.64])
    cbar = fig.colorbar(
        im,
        cax=cax,
    )
    cbar.set_label("PCS", fontsize=cbar_label_fontsize)
    cbar.ax.tick_params(labelsize=max(cbar_label_fontsize - 1, 10))
    fig.suptitle("PCS Heatmaps (Per-User)", fontsize=suptitle_fontsize, fontweight="bold", y=0.965)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PCS heatmap grid for users.")
    parser.add_argument("--pcs_file", required=True, help="PCS JSON file path")
    parser.add_argument(
        "--num_users",
        type=int,
        default=10,
        help="Number of users to plot.",
    )
    parser.add_argument(
        "--sample_method",
        choices=["random", "first", "stride"],
        default="random",
        help="Sampling method for user indices.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--output_dir",
        default="results/preference_head/visualizations/k1_users_10",
        help="Output directory.",
    )
    parser.add_argument(
        "--label_prefix",
        default="user",
        help="Prefix for user labels.",
    )
    parser.add_argument(
        "--cmap",
        default="YlOrRd",
        help="Colormap for heatmaps.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=5,
        help="Number of columns in the grid.",
    )
    parser.add_argument(
        "--indices",
        default=None,
        help="Comma-separated explicit user indices to plot.",
    )
    parser.add_argument(
        "--indices_file",
        default=None,
        help="Path to a text file containing comma-separated user indices.",
    )
    parser.add_argument("--title_fontsize", type=int, default=16)
    parser.add_argument("--axis_label_fontsize", type=int, default=12)
    parser.add_argument("--suptitle_fontsize", type=int, default=20)
    parser.add_argument("--cbar_label_fontsize", type=int, default=12)
    args = parser.parse_args()

    pcs_file = Path(args.pcs_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    heads, pcs_matrix = load_pcs_matrix(pcs_file)
    num_heads_total, num_users_total = pcs_matrix.shape

    num_layers = max(layer for layer, _ in heads) + 1
    num_heads = max(head for _, head in heads) + 1
    if num_heads_total != num_layers * num_heads:
        raise ValueError("PCS file does not match expected layer/head grid.")

    if args.indices_file:
        raw = Path(args.indices_file).read_text().strip()
        indices = np.array([int(x) for x in raw.split(",") if x.strip()], dtype=int)
    elif args.indices:
        indices = np.array([int(x) for x in args.indices.split(",") if x.strip()], dtype=int)
    else:
        indices = select_user_indices(
            num_users_total,
            args.num_users,
            args.sample_method,
            args.seed,
        )

    if np.any(indices < 0) or np.any(indices >= num_users_total):
        raise ValueError(
            f"User indices out of range for PCS matrix with {num_users_total} users: {indices.tolist()}"
        )
    labels = [f"{args.label_prefix}_{i:02d}" for i in indices]

    matrices = []
    for idx in indices:
        user_scores = pcs_matrix[:, idx]
        matrices.append(head_to_matrix(heads, user_scores, num_layers, num_heads))

    plot_grid(
        matrices,
        labels,
        output_dir / "pcs_heatmap_grid.png",
        cmap=args.cmap,
        cols=args.cols,
        title_fontsize=args.title_fontsize,
        axis_label_fontsize=args.axis_label_fontsize,
        suptitle_fontsize=args.suptitle_fontsize,
        cbar_label_fontsize=args.cbar_label_fontsize,
    )

    with (output_dir / "user_indices.txt").open("w") as f:
        f.write(",".join(str(i) for i in indices) + "\n")

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
