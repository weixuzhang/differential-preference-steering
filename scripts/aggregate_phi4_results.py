#!/usr/bin/env python3
"""
Aggregate Phi-4 experiment results under a run root.

Scans prediction JSONL files, evaluates the newest file for each
(task, model, method), and writes:
  - JSON summary
  - TSV table
  - Markdown table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from evaluate_predictions import evaluate_predictions


def parse_prediction_filename(path: Path) -> Tuple[str, str, str] | None:
    name = path.name
    if not name.startswith("pred_") or not name.endswith(".json"):
        return None

    stem = name[len("pred_") : -len(".json")]
    if "__" not in stem:
        return None

    task_model, method = stem.split("__", 1)
    parts = task_model.split("_")
    if len(parts) < 3:
        return None
    if parts[0] not in {"LAMP", "LongLaMP"}:
        return None

    task_prefix = parts[0]
    task_id = parts[1]
    model = "_".join(parts[2:])

    if task_prefix == "LAMP":
        task = f"LaMP-{task_id}"
    else:
        task = f"LongLaMP-{task_id}"
    return task, model, method


def metric_string(result: Dict) -> str:
    if "accuracy" in result:
        parts = [f"acc={result['accuracy']:.4f}"]
        if "f1" in result:
            parts.append(f"f1={result['f1']:.4f}")
        return ", ".join(parts)
    if "mae" in result:
        parts = [f"mae={result['mae']:.4f}"]
        if "rmse" in result:
            parts.append(f"rmse={result['rmse']:.4f}")
        return ", ".join(parts)
    if "rouge-1" in result:
        parts = [f"r1={result['rouge-1']:.4f}"]
        if "rouge-L" in result:
            parts.append(f"rL={result['rouge-L']:.4f}")
        if "meteor" in result:
            parts.append(f"meteor={result['meteor']:.4f}")
        return ", ".join(parts)
    return ""


def sort_key(row: Dict) -> Tuple[int, str, str]:
    task = row["task"]
    prefix, num = task.split("-")
    return (int(num), row["model"], row["method"])


def build_markdown_table(rows: List[Dict]) -> str:
    header = "| task | model | method | metrics | samples |"
    sep = "|---|---|---|---|---:|"
    body = [
        f"| {row['task']} | {row['model']} | {row['method']} | {row['metrics']} | {row['samples']} |"
        for row in rows
    ]
    return "\n".join([header, sep] + body) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Phi-4 experiment results")
    parser.add_argument(
        "--run-root",
        default="runs/phi4_mini",
        help="Root directory containing Phi-4 run outputs",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write summaries (default: <run-root>/summary)",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir) if args.output_dir else run_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    newest: Dict[Tuple[str, str, str], Path] = {}
    for pred_path in run_root.rglob("pred_*.json"):
        parsed = parse_prediction_filename(pred_path)
        if parsed is None:
            continue
        key = parsed
        if key not in newest or pred_path.stat().st_mtime > newest[key].stat().st_mtime:
            newest[key] = pred_path

    if not newest:
        print(f"No prediction files found under {run_root}")
        return

    summary = {}
    rows: List[Dict] = []
    for (task, model, method), pred_path in sorted(newest.items()):
        result = evaluate_predictions(str(pred_path))
        key = f"{task}::{model}::{method}"
        summary[key] = {
            "task": task,
            "model": model,
            "method": method,
            "file": str(pred_path),
            "metrics": result,
        }
        rows.append(
            {
                "task": task,
                "model": model,
                "method": method,
                "metrics": metric_string(result),
                "samples": result.get("total_samples", 0),
                "file": str(pred_path),
            }
        )

    rows.sort(key=sort_key)

    json_path = output_dir / "phi4_results_summary.json"
    tsv_path = output_dir / "phi4_results_table.tsv"
    md_path = output_dir / "phi4_results_table.md"

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    with open(tsv_path, "w") as f:
        f.write("task\tmodel\tmethod\tmetrics\tsamples\tfile\n")
        for row in rows:
            f.write(
                f"{row['task']}\t{row['model']}\t{row['method']}\t{row['metrics']}\t{row['samples']}\t{row['file']}\n"
            )

    md_path.write_text(build_markdown_table(rows))

    print(f"Wrote JSON summary: {json_path}")
    print(f"Wrote TSV table:    {tsv_path}")
    print(f"Wrote Markdown:     {md_path}")
    print()
    print(build_markdown_table(rows))


if __name__ == "__main__":
    main()
