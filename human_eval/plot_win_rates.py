#!/usr/bin/env python3
"""Plot win/lose/tie for human eval results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot win/lose/tie bars")
    parser.add_argument(
        "--input",
        default="/scratch/weixuz/dps/human_eval/LaMP-4_win_rates.json",
    )
    parser.add_argument(
        "--out",
        default="/scratch/weixuz/dps/human_eval/LaMP-4_win_lose_tie.png",
    )
    parser.add_argument("--title", default="LaMP-4 Human Eval (Win/Lose/Tie)")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())

    def _get_counts(section: str):
        sec = data.get(section, {})
        dps = sec.get("DPS", {}).get("count", 0)
        cad = sec.get("CAD", {}).get("count", 0)
        tie = sec.get("TIE", {}).get("count", 0)
        return dps, cad, tie

    primary = _get_counts("primary")
    alignment = _get_counts("alignment")

    labels = ["Primary", "Alignment"]
    dps_vals = [primary[0], alignment[0]]
    cad_vals = [primary[1], alignment[1]]
    tie_vals = [primary[2], alignment[2]]

    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, dps_vals, label="DPS win", color="#A6CEE3")
    ax.bar(x, tie_vals, bottom=dps_vals, label="Tie", color="#FDBF6F")
    bottoms = [d + t for d, t in zip(dps_vals, tie_vals)]
    ax.bar(x, cad_vals, bottom=bottoms, label="CAD win", color="#B2DF8A")

    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Count")
    ax.set_title(args.title)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
