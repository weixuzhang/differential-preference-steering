#!/usr/bin/env python3
"""
Plot Spearman rank correlation heatmap across users based on PCS vectors.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

Head = Tuple[int, int]


def load_pcs_matrix(pcs_file: Path) -> np.ndarray:
    with pcs_file.open("r") as f:
        scores_dict = json.load(f)

    values = []
    for key in sorted(scores_dict.keys(), key=lambda k: tuple(map(int, k.split("-")))):
        values.append(scores_dict[key])

    lengths = {len(v) for v in values}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent PCS lengths found: {sorted(lengths)}")

    matrix = np.asarray(values, dtype=float)  # shape [num_heads, num_users]
    return matrix


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


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def spearman_matrix(pcs_matrix: np.ndarray, user_indices: np.ndarray) -> np.ndarray:
    # pcs_matrix: [num_heads, num_users]
    rank_vectors = []
    for idx in user_indices:
        vec = pcs_matrix[:, idx]
        rank_vectors.append(rankdata(vec))
    rank_matrix = np.vstack(rank_vectors)
    return np.corrcoef(rank_matrix)


def plot_heatmap(
    matrix: np.ndarray,
    labels: List[str],
    output_file: Path,
    cmap: str,
    vmin: float,
    vmax: float,
    mask_diagonal: bool,
) -> None:
    plot_matrix = matrix.copy()
    if mask_diagonal:
        np.fill_diagonal(plot_matrix, np.nan)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#f0f0f0")

    fig, ax = plt.subplots(figsize=(10, 8))
    masked = np.ma.masked_invalid(plot_matrix)
    im = ax.imshow(masked, cmap=cmap_obj, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman")
    ax.set_title("Per-User PCS Spearman Correlation", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-user PCS Spearman heatmap.")
    parser.add_argument("--pcs_file", required=True, help="PCS JSON file path")
    parser.add_argument(
        "--num_users",
        type=int,
        default=50,
        help="Number of users to include in the heatmap.",
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
        default="results/preference_head/visualizations/k1_users_spearman",
        help="Output directory.",
    )
    parser.add_argument(
        "--label_prefix",
        default="user",
        help="Prefix for user labels.",
    )
    parser.add_argument(
        "--cmap",
        default="coolwarm",
        help="Colormap for heatmap.",
    )
    parser.add_argument(
        "--mask_diagonal",
        action="store_true",
        help="Mask diagonal cells in the heatmap.",
    )
    args = parser.parse_args()

    pcs_file = Path(args.pcs_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pcs_matrix = load_pcs_matrix(pcs_file)
    num_users_total = pcs_matrix.shape[1]
    indices = select_user_indices(
        num_users_total,
        args.num_users,
        args.sample_method,
        args.seed,
    )
    labels = [f"{args.label_prefix}_{i:02d}" for i in indices]

    matrix = spearman_matrix(pcs_matrix, indices)

    plot_heatmap(
        matrix,
        labels,
        output_dir / "pcs_spearman_heatmap.png",
        cmap=args.cmap,
        vmin=-1.0,
        vmax=1.0,
        mask_diagonal=args.mask_diagonal,
    )

    with (output_dir / "user_indices.txt").open("w") as f:
        f.write(",".join(str(i) for i in indices) + "\n")

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
