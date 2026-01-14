#!/usr/bin/env python3
"""
Randomize per-cluster head weights by shuffling weights across heads.

This preserves the weight distribution while assigning weights to random heads.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def _randomize(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    flat = weights.flatten()
    rng.shuffle(flat)
    return flat.reshape(weights.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomize cluster head weights")
    parser.add_argument("--src_dir", required=True, help="Source cluster_heads dir")
    parser.add_argument("--out_dir", required=True, help="Output dir for randomized heads")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_dirs = sorted([p for p in src_dir.glob("cluster_*") if p.is_dir()])
    if not cluster_dirs:
        raise FileNotFoundError(f"No cluster_* dirs found in {src_dir}")

    rng = np.random.default_rng(args.seed)

    for cluster_dir in cluster_dirs:
        weight_path = cluster_dir / "head_weights.json"
        if not weight_path.exists():
            print(f"Skipping {cluster_dir} (missing head_weights.json)")
            continue

        with open(weight_path, "r") as f:
            payload = json.load(f)

        weights = np.array(payload["head_weights"], dtype=np.float32)
        randomized = _randomize(weights, rng)

        payload["head_weights"] = randomized.tolist()
        payload["randomized"] = True
        payload["random_seed"] = args.seed
        payload["source_dir"] = str(src_dir)

        out_cluster = out_dir / cluster_dir.name
        out_cluster.mkdir(parents=True, exist_ok=True)

        with open(out_cluster / "head_weights.json", "w") as f:
            json.dump(payload, f, indent=2)
        np.save(out_cluster / "head_weights.npy", randomized)

    print(f"Saved randomized head weights to: {out_dir}")


if __name__ == "__main__":
    main()
