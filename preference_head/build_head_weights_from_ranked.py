#!/usr/bin/env python3
"""
Build head_weights.json from ranked head files, selecting top-k heads.

This is useful for head-count sweeps when you want to detect once (PCS) and
re-slice the top-k heads without re-running detection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _load_ranked(path: Path) -> Tuple[Dict, List[Dict]]:
    with path.open("r") as f:
        data = json.load(f)
    ranked = data.get("ranked_heads", [])
    if not ranked:
        raise ValueError(f"No ranked_heads in {path}")
    ranked_sorted = sorted(
        ranked,
        key=lambda r: (r.get("rank", 1_000_000_000), -float(r.get("avg_pcs", 0.0))),
    )
    return data, ranked_sorted


def _infer_shape(ranked: List[Dict]) -> Tuple[int, int]:
    max_layer = max(int(r["layer"]) for r in ranked)
    max_head = max(int(r["head"]) for r in ranked)
    return max_layer + 1, max_head + 1


def _normalize(weights: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return weights
    if mode == "max":
        denom = float(weights.max()) if float(weights.max()) > 0 else 1.0
        return weights / denom
    if mode == "sum":
        denom = float(weights.sum()) if float(weights.sum()) > 0 else 1.0
        return weights / denom
    raise ValueError(f"Unknown norm: {mode}")


def build_weights(
    ranked: List[Dict],
    num_layers: int,
    num_heads: int,
    keep_heads: int,
    pcs_min: float,
    pcs_norm: str,
    pcs_power: float,
) -> np.ndarray:
    keep_heads = max(1, min(keep_heads, num_layers * num_heads))
    weights = np.zeros((num_layers, num_heads), dtype=np.float32)
    for entry in ranked[:keep_heads]:
        layer = int(entry["layer"])
        head = int(entry["head"])
        avg_pcs = float(entry.get("avg_pcs", 0.0))
        if avg_pcs < pcs_min:
            continue
        weights[layer, head] = max(avg_pcs, 0.0)

    weights = _normalize(weights, pcs_norm)
    if pcs_power != 1.0:
        weights = np.power(weights, pcs_power)
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build head weights from ranked head files"
    )
    parser.add_argument("--src_dir", required=True, help="Source cluster_heads dir")
    parser.add_argument("--out_dir", required=True, help="Output cluster_heads dir")
    parser.add_argument("--num_heads", type=int, default=None, help="Top-k heads")
    parser.add_argument("--top_percent", type=float, default=None, help="Top percent")
    parser.add_argument("--pcs_norm", default="max", choices=["max", "sum", "none"])
    parser.add_argument("--pcs_min", type=float, default=0.0)
    parser.add_argument("--pcs_power", type=float, default=1.0)
    args = parser.parse_args()

    if args.num_heads is None and args.top_percent is None:
        raise ValueError("Provide --num_heads or --top_percent")
    if args.num_heads is not None and args.num_heads <= 0:
        raise ValueError("--num_heads must be positive")

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_dirs = sorted([p for p in src_dir.glob("cluster_*") if p.is_dir()])
    if not cluster_dirs:
        raise FileNotFoundError(f"No cluster_* dirs found in {src_dir}")

    for cluster_dir in cluster_dirs:
        ranked_files = list(cluster_dir.glob("*_ranked.json"))
        if not ranked_files:
            print(f"Skipping {cluster_dir} (missing *_ranked.json)")
            continue
        ranked_path = ranked_files[0]
        meta, ranked = _load_ranked(ranked_path)
        num_layers, num_heads = _infer_shape(ranked)
        total_heads = num_layers * num_heads

        if args.num_heads is not None:
            keep_heads = int(args.num_heads)
        else:
            keep_heads = int(total_heads * float(args.top_percent))
        keep_heads = max(1, min(keep_heads, total_heads))

        weights = build_weights(
            ranked,
            num_layers,
            num_heads,
            keep_heads,
            args.pcs_min,
            args.pcs_norm,
            args.pcs_power,
        )

        cluster_id = int(cluster_dir.name.split("_")[-1])
        payload = {
            "model": meta.get("model"),
            "task": meta.get("task"),
            "cluster_id": cluster_id,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "top_percent": keep_heads / float(total_heads),
            "heads_count": keep_heads,
            "pcs_norm": args.pcs_norm,
            "pcs_min": args.pcs_min,
            "pcs_power": args.pcs_power,
            "head_weights": weights.tolist(),
            "source_ranked": str(ranked_path),
        }

        out_cluster = out_dir / cluster_dir.name
        out_cluster.mkdir(parents=True, exist_ok=True)
        with (out_cluster / "head_weights.json").open("w") as f:
            json.dump(payload, f, indent=2)
        np.save(out_cluster / "head_weights.npy", weights)

    print(f"Saved head weights to: {out_dir}")


if __name__ == "__main__":
    main()
