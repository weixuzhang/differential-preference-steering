#!/usr/bin/env python3
"""
Build a markdown report summarizing hyperparameter + ablation results.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))
from evaluate_predictions import evaluate_predictions


HYPARAM_DIR = REPO_ROOT / "outputs" / "hparam"


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rows.append({k: v for k, v in zip(header, parts)})
    return rows


def _eval_files(files: List[Path]) -> List[Dict[str, object]]:
    rows = []
    for pred in files:
        res = evaluate_predictions(str(pred))
        rows.append(
            {
                "task": res.get("task"),
                "samples": res.get("total_samples"),
                "accuracy": res.get("accuracy"),
                "f1": res.get("f1"),
                "mae": res.get("mae"),
                "rmse": res.get("rmse"),
                "rouge-1": res.get("rouge-1"),
                "rouge-L": res.get("rouge-L"),
                "meteor": res.get("meteor"),
                "pred_file": str(pred),
            }
        )
    return rows


def _find_pred_files(root: Path, pattern: str) -> List[Path]:
    return sorted([p for p in root.rglob(pattern) if not p.name.endswith("_eval.json")])


def _num(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value.strip() == "" or value.strip().lower() == "none":
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _format_metric(row: Dict[str, object]) -> str:
    acc = _num(row.get("accuracy"))
    f1 = _num(row.get("f1"))
    mae = _num(row.get("mae"))
    rmse = _num(row.get("rmse"))
    r1 = _num(row.get("rouge-1"))
    rL = _num(row.get("rouge-L"))
    meteor = _num(row.get("meteor"))

    if acc is not None and f1 is not None:
        return f"acc={acc:.4f}, f1={f1:.4f}"
    if mae is not None:
        return f"mae={mae:.4f}, rmse={rmse:.4f}"
    if rL is not None:
        return f"r1={r1:.4f}, rL={rL:.4f}, meteor={meteor:.4f}"
    return "n/a"


def _score_row(row: Dict[str, object]) -> tuple[Optional[float], str]:
    """Return (score, metric_name) for ranking. Higher is better."""
    acc = _num(row.get("accuracy"))
    if acc is not None:
        return acc, "accuracy"
    rL = _num(row.get("rouge-L"))
    if rL is not None:
        return rL, "rouge-L"
    mae = _num(row.get("mae"))
    if mae is not None:
        return -mae, "mae (lower is better)"
    return None, "n/a"


def _best_setting(rows: List[Dict[str, object]], label_key: str) -> str:
    """Pick best label(s) by score; return a compact string."""
    scored = []
    for row in rows:
        score, metric = _score_row(row)
        if score is None:
            continue
        scored.append((score, metric, row.get(label_key)))
    if not scored:
        return "n/a"
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score = scored[0][0]
    metric = scored[0][1]
    labels = sorted({str(label) for score, _, label in scored if abs(score - best_score) < 1e-6})
    return f"{', '.join(labels)} (best by {metric})"


def _compare_pairs(rows: List[Dict[str, object]], key: str, a: str, b: str) -> Dict[str, List[str]]:
    """Compare two settings per task and return wins/ties/losses lists for a vs b."""
    grouped: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        task = row.get("task")
        setting = row.get(key)
        if task and setting:
            grouped.setdefault(task, {})[setting] = row

    wins: List[str] = []
    losses: List[str] = []
    ties: List[str] = []
    for task, settings in grouped.items():
        if a not in settings or b not in settings:
            continue
        score_a, _ = _score_row(settings[a])
        score_b, _ = _score_row(settings[b])
        if score_a is None or score_b is None:
            continue
        if abs(score_a - score_b) < 1e-6:
            ties.append(task)
        elif score_a > score_b:
            wins.append(task)
        else:
            losses.append(task)
    return {"wins": wins, "losses": losses, "ties": ties}


def _find_true_heads(task: str) -> Optional[Path]:
    patterns = [
        REPO_ROOT / "outputs",
        ROOT / "outputs",
    ]
    matches = []
    for base in patterns:
        for pred in base.rglob("pred_LAMP_*__DPSWeightedSoft.json"):
            if "/hparam/" in str(pred):
                continue
            if f"pred_{task.replace('-', '_')}" in pred.name:
                matches.append(pred)
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _table(rows: List[List[str]]) -> str:
    if not rows:
        return "_No results._"
    col_count = max(len(r) for r in rows)
    widths = [0] * col_count
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = []
    for idx, row in enumerate(rows):
        padded = [row[i].ljust(widths[i]) for i in range(col_count)]
        lines.append("| " + " | ".join(padded) + " |")
        if idx == 0:
            lines.append("| " + " | ".join("-" * w for w in widths) + " |")
    return "\n".join(lines)


def main() -> None:
    report_path = REPO_ROOT / "docs" / "HYPERPARAM_ABLATION_RESULTS.md"

    heads_rows = _read_tsv(HYPARAM_DIR / "heads_quick" / "summary.tsv")
    groups_rows = _read_tsv(HYPARAM_DIR / "groupsize_quick" / "summary.tsv")

    gamma_files = _find_pred_files(HYPARAM_DIR / "gamma_full", "pred_LAMP_*__DPSWeightedSoft.json")
    gamma_rows = _eval_files(gamma_files)
    # tag gamma rows with setting
    for row in gamma_rows:
        pred_path = Path(row["pred_file"])
        if "fixed_alpha_" in pred_path.parent.name:
            row["setting"] = pred_path.parent.name
        elif pred_path.parent.name == "adaptive":
            row["setting"] = "adaptive"
        else:
            row["setting"] = pred_path.parent.name

    ablation_files = _find_pred_files(HYPARAM_DIR / "ablation_full", "pred_LAMP_*__DPSWeightedSoft.json")
    ablation_rows = _eval_files(ablation_files)
    for row in ablation_rows:
        pred_path = Path(row["pred_file"])
        row["setting"] = pred_path.parent.name

    # add true heads references if available
    ablation_with_true = list(ablation_rows)
    tasks_present = sorted({row["task"] for row in ablation_rows if row.get("task")})
    for task in tasks_present:
        true_path = _find_true_heads(task)
        if not true_path:
            continue
        res = evaluate_predictions(str(true_path))
        ablation_with_true.append(
            {
                "task": res.get("task"),
                "samples": res.get("total_samples"),
                "accuracy": res.get("accuracy"),
                "f1": res.get("f1"),
                "mae": res.get("mae"),
                "rmse": res.get("rmse"),
                "rouge-1": res.get("rouge-1"),
                "rouge-L": res.get("rouge-L"),
                "meteor": res.get("meteor"),
                "pred_file": str(true_path),
                "setting": "true_heads_ref",
            }
        )

    routing_rows = _read_tsv(HYPARAM_DIR / "routing_ablation" / "summary.tsv")

    # Build markdown sections
    lines = []
    lines.append("# Hyperparameter & Ablation Results (Current)")
    lines.append("")
    lines.append("This report summarizes all hyperparameter/ablation runs available so far.")
    lines.append("")

    lines.append("## Implementation Summary")
    lines.append("")
    lines.append("- **Heads sweep:** uses existing ranked heads (k≈25), evaluates DPSWeightedSoft on 50 samples.")
    lines.append("- **Group-size sweep:** uses existing k=250 (≈10 users) and k=25 (≈100 users), evaluates on 50 samples.")
    lines.append("- **Gamma sweep:** adaptive α vs fixed α=0.5, full dev where available.")
    lines.append("- **Heads ablation:** randomize head weights and random mask; includes reference DPSWeightedSoft where found.")
    lines.append("- **Routing ablation:** soft vs hard routing with 200 samples (per-task).")
    lines.append("")

    lines.append("## Key Takeaways (Auto)")
    lines.append("")

    if heads_rows:
        heads_best = _best_setting(heads_rows, "param")
        lines.append(f"- Heads sweep: {heads_best}.")
    else:
        lines.append("- Heads sweep: n/a.")

    if groups_rows:
        group_best = _best_setting(groups_rows, "param")
        lines.append(f"- Group size sweep: {group_best}.")
    else:
        lines.append("- Group size sweep: n/a.")

    gamma_cmp = _compare_pairs(gamma_rows, "setting", "adaptive", "fixed_alpha_0.5")
    if gamma_cmp["wins"] or gamma_cmp["losses"] or gamma_cmp["ties"]:
        lines.append(
            "- Gamma sweep: adaptive wins on "
            f"{len(gamma_cmp['wins'])} task(s) "
            f"({', '.join(gamma_cmp['wins']) or 'n/a'}), "
            "fixed α=0.5 wins on "
            f"{len(gamma_cmp['losses'])} task(s) "
            f"({', '.join(gamma_cmp['losses']) or 'n/a'}), "
            f"ties on {len(gamma_cmp['ties'])} task(s) "
            f"({', '.join(gamma_cmp['ties']) or 'n/a'})."
        )
    else:
        lines.append("- Gamma sweep: n/a.")

    ablation_cmp = _compare_pairs(ablation_with_true, "setting", "true_heads_ref", "random_heads")
    if ablation_cmp["wins"] or ablation_cmp["losses"] or ablation_cmp["ties"]:
        lines.append(
            "- Heads ablation (random heads vs true): true wins on "
            f"{len(ablation_cmp['wins'])} task(s) "
            f"({', '.join(ablation_cmp['wins']) or 'n/a'})."
        )
    else:
        lines.append("- Heads ablation: n/a (no true-head references found).")

    routing_cmp = _compare_pairs(routing_rows, "routing", "soft", "hard")
    if routing_cmp["wins"] or routing_cmp["losses"] or routing_cmp["ties"]:
        lines.append(
            "- Routing ablation: soft wins on "
            f"{len(routing_cmp['wins'])} task(s) "
            f"({', '.join(routing_cmp['wins']) or 'n/a'}), "
            "hard wins on "
            f"{len(routing_cmp['losses'])} task(s) "
            f"({', '.join(routing_cmp['losses']) or 'n/a'}), "
            f"ties on {len(routing_cmp['ties'])} task(s) "
            f"({', '.join(routing_cmp['ties']) or 'n/a'})."
        )
    else:
        lines.append("- Routing ablation: n/a.")

    lines.append("")

    lines.append("## Heads Sweep (50-sample quick)")
    lines.append("")
    if heads_rows:
        table = [["heads", "task", "samples", "metrics", "pred_file"]]
        for row in heads_rows:
            metrics = _format_metric(row)
            table.append(
                [
                    row["param"],
                    row["task"],
                    row["samples"],
                    metrics,
                    row["pred_file"].replace(str(ROOT) + "/", ""),
                ]
            )
        lines.append(_table(table))
    else:
        lines.append("_No heads sweep results found._")
    lines.append("")
    lines.append("Figure: `outputs/hparam/figures/heads/heads_sweep.png`")
    lines.append("")

    lines.append("## Group Size Sweep (50-sample quick)")
    lines.append("")
    if groups_rows:
        table = [["group_size", "task", "samples", "metrics", "pred_file"]]
        for row in groups_rows:
            metrics = _format_metric(row)
            table.append(
                [
                    row["param"],
                    row["task"],
                    row["samples"],
                    metrics,
                    row["pred_file"].replace(str(ROOT) + "/", ""),
                ]
            )
        lines.append(_table(table))
    else:
        lines.append("_No group size sweep results found._")
    lines.append("")
    lines.append("Figure: `outputs/hparam/figures/groupsize/groupsize_sweep.png`")
    lines.append("")

    lines.append("## Gamma Sweep (full dev where available)")
    lines.append("")
    if gamma_rows:
        table = [["task", "setting", "samples", "metrics", "pred_file"]]
        for row in sorted(gamma_rows, key=lambda r: (r["task"], r["setting"])):
            metrics = _format_metric(row)
            table.append(
                [
                    row["task"],
                    row["setting"],
                    str(row["samples"]),
                    metrics,
                    str(row["pred_file"]).replace(str(ROOT) + "/", ""),
                ]
            )
        lines.append(_table(table))
    else:
        lines.append("_No gamma sweep results found._")
    lines.append("")
    lines.append("Figure: `outputs/hparam/figures/gamma/gamma_sweep.png` (LaMP-1)")
    lines.append("")

    lines.append("## Heads Ablation (full dev where available)")
    lines.append("")
    if ablation_with_true:
        table = [["task", "setting", "samples", "metrics", "pred_file"]]
        for row in sorted(ablation_with_true, key=lambda r: (r["task"], r["setting"])):
            metrics = _format_metric(row)
            table.append(
                [
                    row["task"],
                    row["setting"],
                    str(row["samples"]),
                    metrics,
                    str(row["pred_file"]).replace(str(ROOT) + "/", ""),
                ]
            )
        lines.append(_table(table))
    else:
        lines.append("_No ablation results found._")
    lines.append("")
    lines.append("Figures: `outputs/hparam/figures/ablation/ablation_metrics.png`, `outputs/hparam/figures/ablation/ablation_head_heatmaps.png`")
    lines.append("")

    lines.append("## Routing Ablation (soft vs hard, 200 samples)")
    lines.append("")
    if routing_rows:
        table = [["task", "routing", "samples", "metrics", "pred_file"]]
        for row in routing_rows:
            metrics = _format_metric(row)
            table.append(
                [
                    row["task"],
                    row["routing"],
                    row["samples"],
                    metrics,
                    row["pred_file"].replace(str(ROOT) + "/", ""),
                ]
            )
        lines.append(_table(table))
    else:
        lines.append("_No routing ablation results found._")
    lines.append("")
    lines.append("Figure: `outputs/hparam/routing_ablation/routing_ablation.png`")
    lines.append("")

    # Coverage & missing
    tasks_all = ["LaMP-1", "LaMP-2", "LaMP-3", "LaMP-4", "LaMP-5", "LaMP-7"]

    heads_expected = [10, 20, 40, 80, 160]
    heads_present = sorted(
        {
            int(row.get("param"))
            for row in heads_rows
            if str(row.get("param", "")).isdigit()
        }
    )
    heads_missing = [str(h) for h in heads_expected if h not in heads_present]

    group_expected = [10, 50, 100, 200, 400]
    group_present = sorted(
        {
            int(row.get("param"))
            for row in groups_rows
            if str(row.get("param", "")).isdigit()
        }
    )
    group_missing = [str(g) for g in group_expected if g not in group_present]

    gamma_tasks = sorted({row.get("task") for row in gamma_rows if row.get("task")})
    gamma_missing = [t for t in tasks_all if t not in gamma_tasks]

    ablation_tasks = sorted({row.get("task") for row in ablation_rows if row.get("task")})
    ablation_missing = [t for t in tasks_all if t not in ablation_tasks]

    routing_by_task = {}
    for row in routing_rows:
        task = row.get("task")
        routing = row.get("routing")
        if task and routing:
            routing_by_task.setdefault(task, set()).add(routing)
    routing_missing = []
    for task in tasks_all:
        got = routing_by_task.get(task, set())
        if "soft" not in got or "hard" not in got:
            routing_missing.append(task)

    lines.append("## Coverage & Missing Points")
    lines.append("")
    lines.append(f"- Heads sweep missing: {', '.join(heads_missing) if heads_missing else 'none'}")
    lines.append(f"- Group size sweep missing: {', '.join(group_missing) if group_missing else 'none'}")
    lines.append(f"- Gamma sweep missing tasks: {', '.join(gamma_missing) if gamma_missing else 'none'}")
    lines.append(f"- Heads ablation missing tasks: {', '.join(ablation_missing) if ablation_missing else 'none'}")
    lines.append(f"- Routing ablation missing soft/hard pairs: {', '.join(routing_missing) if routing_missing else 'none'}")
    lines.append("")

    report_path.write_text("\n".join(lines))
    print(f"Wrote report to: {report_path}")


if __name__ == "__main__":
    main()
