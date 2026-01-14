#!/usr/bin/env python3
"""
Create random binary head masks per cluster (true masking).

The output head_weights matrices contain 1.0 for masked heads and 0.0 otherwise.
Weighted DPS will then zero out those heads in the depersonalized pass.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def _pick_random_heads(
    rng: np.random.Generator, num_layers: int, num_heads: int, keep_heads: int
) -> list[list[int]]:
    total = num_layers * num_heads
    keep_heads = max(1, min(keep_heads, total))
    flat_indices = rng.choice(total, size=keep_heads, replace=False)
    return [[int(idx // num_heads), int(idx % num_heads)] for idx in flat_indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Random mask cluster head weights")
    parser.add_argument("--src_dir", required=True, help="Source cluster_heads dir")
    parser.add_argument("--out_dir", required=True, help="Output dir for random masks")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top_percent", type=float, default=None)
    parser.add_argument("--heads_count", type=int, default=None)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_dirs = sorted([p for p in src_dir.glob("cluster_*") if p.is_dir()])
    if not cluster_dirs:
        raise FileNotFoundError(f"No cluster_* dirs found in {src_dir}")

    for cluster_dir in cluster_dirs:
        weight_path = cluster_dir / "head_weights.json"
        if not weight_path.exists():
            print(f"Skipping {cluster_dir} (missing head_weights.json)")
            continue

        with open(weight_path, "r") as f:
            payload = json.load(f)

        weights = np.array(payload["head_weights"], dtype=np.float32)
        num_layers, num_heads = weights.shape
        total_heads = num_layers * num_heads

        if args.heads_count is not None:
            keep_heads = args.heads_count
        elif args.top_percent is not None:
            keep_heads = int(total_heads * args.top_percent)
        elif "top_percent" in payload:
            keep_heads = int(total_heads * float(payload["top_percent"]))
        else:
            keep_heads = int(np.count_nonzero(weights))
        keep_heads = max(1, min(keep_heads, total_heads))

        cluster_id = int(cluster_dir.name.split("_")[-1])
        rng = np.random.default_rng(args.seed + cluster_id)
        selected = _pick_random_heads(rng, num_layers, num_heads, keep_heads)

        mask = np.zeros((num_layers, num_heads), dtype=np.float32)
        for layer, head in selected:
            mask[layer, head] = 1.0

        new_payload = dict(payload)
        new_payload["head_weights"] = mask.tolist()
        new_payload["random_mask"] = True
        new_payload["random_seed"] = args.seed
        new_payload["heads_count"] = keep_heads
        new_payload["source_dir"] = str(src_dir)
        new_payload["masked_heads"] = selected

        out_cluster = out_dir / cluster_dir.name
        out_cluster.mkdir(parents=True, exist_ok=True)
        with open(out_cluster / "head_weights.json", "w") as f:
            json.dump(new_payload, f, indent=2)
        np.save(out_cluster / "head_weights.npy", mask)

    print(f"Saved random masks to: {out_dir}")


if __name__ == "__main__":
    main()
