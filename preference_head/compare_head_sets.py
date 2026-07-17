#!/usr/bin/env python3
"""
Compare preference head sets across groups (global vs cluster/user).

Outputs:
  - PCS heatmap grid (layer x head) for each group
  - Similarity matrix (Jaccard overlap or Spearman rank correlation)
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # type: ignore
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

Head = Tuple[int, int]


def _task_token(task: str) -> str:
    return task.replace("-", "_")


def _find_single_file(group_dir: Path, pattern: str, model: str | None) -> Path:
    candidates = sorted(group_dir.glob(pattern))
    if model:
        candidates = [p for p in candidates if p.name.startswith(f"{model}_")]
    if not candidates:
        raise FileNotFoundError(f"No files matching '{pattern}' in {group_dir}")
    if len(candidates) > 1:
        names = "\n".join(str(p) for p in candidates)
        raise ValueError(
            "Multiple matches found; specify --model to disambiguate:\n" + names
        )
    return candidates[0]


def find_group_files(group_dir: Path, task: str, model: str | None) -> Dict[str, Path]:
    task_token = _task_token(task)
    pcs = _find_single_file(group_dir, f"*_{task_token}_pcs.json", model)
    ranked = _find_single_file(group_dir, f"*_{task_token}_ranked.json", model)
    top_heads = _find_single_file(group_dir, f"*_{task_token}_top_heads.json", model)
    return {"pcs": pcs, "ranked": ranked, "top_heads": top_heads}


def load_pcs_matrix(pcs_file: Path) -> np.ndarray:
    with pcs_file.open("r") as f:
        scores_dict = json.load(f)

    items = []
    for key, values in scores_dict.items():
        layer, head = map(int, key.split("-"))
        items.append(((layer, head), float(np.mean(values))))

    max_layer = max(layer for (layer, _), _ in items)
    max_head = max(head for (_, head), _ in items)
    matrix = np.full((max_layer + 1, max_head + 1), np.nan, dtype=float)

    for (layer, head), value in items:
        matrix[layer, head] = value

    return matrix


def load_top_heads(top_heads_file: Path) -> List[Head]:
    with top_heads_file.open("r") as f:
        data = json.load(f)
    return [tuple(h) for h in data["preference_heads"]]


def load_ranked_scores(ranked_file: Path) -> Dict[Head, float]:
    with ranked_file.open("r") as f:
        data = json.load(f)
    scores = {}
    for entry in data["ranked_heads"]:
        scores[(entry["layer"], entry["head"])] = float(entry["avg_pcs"])
    return scores


def pad_matrices(matrices: List[np.ndarray]) -> List[np.ndarray]:
    max_rows = max(mat.shape[0] for mat in matrices)
    max_cols = max(mat.shape[1] for mat in matrices)
    padded = []
    for mat in matrices:
        pad_rows = max_rows - mat.shape[0]
        pad_cols = max_cols - mat.shape[1]
        if pad_rows or pad_cols:
            mat = np.pad(
                mat,
                ((0, pad_rows), (0, pad_cols)),
                mode="constant",
                constant_values=np.nan,
            )
        padded.append(mat)
    return padded


def plot_pcs_heatmap_grid(
    matrices: List[np.ndarray],
    labels: List[str],
    output_file: Path,
    cmap: str = "YlOrRd",
):
    matrices = pad_matrices(matrices)
    all_vals = np.concatenate([m[~np.isnan(m)].ravel() for m in matrices])
    vmin = float(np.min(all_vals)) if all_vals.size else 0.0
    vmax = float(np.max(all_vals)) if all_vals.size else 1.0

    n = len(matrices)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))

    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])

    for idx, (mat, label) in enumerate(zip(matrices, labels)):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        mask = np.isnan(mat)
        if _HAS_SEABORN:
            sns.heatmap(
                mat,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                cbar=False,
                ax=ax,
                mask=mask,
                xticklabels=False,
                yticklabels=False,
            )
        else:
            cmap_obj = plt.get_cmap(cmap).copy()
            cmap_obj.set_bad(color="#f0f0f0")
            masked = np.ma.masked_invalid(mat)
            ax.imshow(masked, cmap=cmap_obj, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Head Index")
        ax.set_ylabel("Layer Index")

    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    mappable.set_array([])
    fig.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        fraction=0.02,
        pad=0.02,
        label="Average PCS",
    )

    fig.suptitle("PCS Heatmaps by Group", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _rankdata(values: np.ndarray) -> np.ndarray:
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


def spearman_matrix(score_maps: List[Dict[Head, float]]) -> np.ndarray:
    head_keys = sorted(score_maps[0].keys())
    for scores in score_maps[1:]:
        if set(scores.keys()) != set(head_keys):
            raise ValueError("Ranked files must contain the same head keys.")

    rank_vectors = []
    for scores in score_maps:
        vec = np.array([scores[k] for k in head_keys], dtype=float)
        rank_vectors.append(_rankdata(vec))

    n = len(rank_vectors)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                corr = np.corrcoef(rank_vectors[i], rank_vectors[j])[0, 1]
                matrix[i, j] = float(corr)
    return matrix


def jaccard_matrix(head_sets: List[Iterable[Head]]) -> np.ndarray:
    sets = [set(hs) for hs in head_sets]
    n = len(sets)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
                continue
            union = sets[i] | sets[j]
            inter = sets[i] & sets[j]
            matrix[i, j] = float(len(inter) / len(union)) if union else 0.0
    return matrix


def write_matrix_csv(matrix: np.ndarray, labels: List[str], output_file: Path) -> None:
    with output_file.open("w") as f:
        f.write("," + ",".join(labels) + "\n")
        for label, row in zip(labels, matrix):
            row_str = ",".join(f"{v:.4f}" for v in row)
            f.write(f"{label},{row_str}\n")


def plot_similarity_heatmap(
    matrix: np.ndarray,
    labels: List[str],
    output_file: Path,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    center: float | None = None,
):
    fig, ax = plt.subplots(
        figsize=(8 + len(labels) * 0.3, 6 + len(labels) * 0.2)
    )
    if _HAS_SEABORN:
        sns.heatmap(
            matrix,
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            fmt=".2f",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            center=center,
            square=True,
            ax=ax,
        )
    else:
        cmap_obj = plt.get_cmap(cmap)
        im = ax.imshow(matrix, cmap=cmap_obj, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
        fig.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare preference head sets across groups."
    )
    parser.add_argument("--task", required=True, help="Task name, e.g., LaMP-1")
    parser.add_argument(
        "--model",
        default=None,
        help="Model prefix used in filenames, e.g., Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--group",
        nargs="+",
        required=True,
        help="Group directories to compare (each contains pcs/ranked/top_heads files).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels for groups (must match number of --group entries).",
    )
    parser.add_argument(
        "--similarity",
        nargs="+",
        choices=["jaccard", "spearman"],
        default=["jaccard"],
        help="Similarity metrics to compute.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="If set, build head sets from top-k ranked heads instead of top_heads files.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/preference_head/visualizations",
        help="Directory to save outputs.",
    )
    args = parser.parse_args()

    group_dirs = [Path(p) for p in args.group]
    labels = args.labels if args.labels else [d.name for d in group_dirs]
    if len(labels) != len(group_dirs):
        raise ValueError("--labels must match number of --group entries.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pcs_matrices = []
    head_sets = []
    ranked_scores = []

    for group_dir in group_dirs:
        files = find_group_files(group_dir, args.task, args.model)
        pcs_matrices.append(load_pcs_matrix(files["pcs"]))
        ranked_scores.append(load_ranked_scores(files["ranked"]))
        if args.top_k:
            ranked = sorted(
                ranked_scores[-1].items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[: args.top_k]
            head_sets.append([head for head, _ in ranked])
        else:
            head_sets.append(load_top_heads(files["top_heads"]))

    plot_pcs_heatmap_grid(
        pcs_matrices,
        labels,
        output_dir / "pcs_heatmap_grid.png",
    )

    if "jaccard" in args.similarity:
        matrix = jaccard_matrix(head_sets)
        plot_similarity_heatmap(
            matrix,
            labels,
            output_dir / "headset_jaccard_heatmap.png",
            "Head Set Jaccard Overlap",
            vmin=0.0,
            vmax=1.0,
            cmap="Blues",
        )
        write_matrix_csv(matrix, labels, output_dir / "headset_jaccard.csv")

    if "spearman" in args.similarity:
        matrix = spearman_matrix(ranked_scores)
        plot_similarity_heatmap(
            matrix,
            labels,
            output_dir / "pcs_spearman_heatmap.png",
            "PCS Spearman Rank Correlation",
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
            center=0.0,
        )
        write_matrix_csv(matrix, labels, output_dir / "pcs_spearman.csv")

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
