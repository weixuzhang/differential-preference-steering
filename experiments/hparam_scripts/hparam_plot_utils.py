#!/usr/bin/env python3
"""
Helpers for hyperparameter sweep plots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def normalize_task(task_field) -> str:
    if isinstance(task_field, list) and task_field:
        return str(task_field[0])
    return str(task_field)


def _extract_answer_value(value) -> str:
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, list) and first:
            return str(first[0])
        return str(first)
    return str(value)


def extract_pred_target(sample: Dict) -> Tuple[str, str]:
    pred = str(sample.get("predicted_answer", "")).strip()
    answers = sample.get("answers")
    if answers is None:
        answers = sample.get("answer")
    if answers is None:
        answers = sample.get("target")
    target = _extract_answer_value(answers).strip() if answers is not None else ""
    return pred, target


def get_labels(task: str) -> List[str]:
    if task == "LaMP-1":
        return ["[1]", "[2]"]
    if task == "LaMP-2":
        return [
            "sci-fi",
            "based on a book",
            "comedy",
            "action",
            "twist ending",
            "dystopia",
            "dark comedy",
            "classic",
            "psychology",
            "fantasy",
            "romance",
            "thought-provoking",
            "social commentary",
            "violence",
            "true story",
        ]
    return []


def _label_index(label: str, labels: List[str]) -> int:
    try:
        return labels.index(label.strip())
    except ValueError:
        return -1


def accuracy_f1(preds: List[str], targets: List[str], task: str) -> Tuple[float, float]:
    labels = get_labels(task)
    if not labels:
        raise ValueError(f"Unsupported task for accuracy/f1: {task}")

    pred_idx = [_label_index(p, labels) for p in preds]
    true_idx = [_label_index(t, labels) for t in targets]

    correct = [int(p == t) for p, t in zip(pred_idx, true_idx)]
    acc = float(np.mean(correct)) if correct else 0.0

    f1s = []
    for label_id in range(len(labels)):
        tp = sum(1 for p, t in zip(pred_idx, true_idx) if p == label_id and t == label_id)
        fp = sum(1 for p, t in zip(pred_idx, true_idx) if p == label_id and t != label_id)
        fn = sum(1 for p, t in zip(pred_idx, true_idx) if p != label_id and t == label_id)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    f1_macro = float(np.mean(f1s)) if f1s else 0.0
    return acc, f1_macro


def bootstrap_ci(
    preds: List[str],
    targets: List[str],
    task: str,
    num_samples: int = 500,
    seed: int = 1234,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(preds)
    if n == 0:
        return {"acc": 0.0, "f1": 0.0, "acc_lo": 0.0, "acc_hi": 0.0, "f1_lo": 0.0, "f1_hi": 0.0}

    accs = []
    f1s = []
    indices = np.arange(n)
    for _ in range(num_samples):
        sample_idx = rng.choice(indices, size=n, replace=True)
        sample_preds = [preds[i] for i in sample_idx]
        sample_targets = [targets[i] for i in sample_idx]
        acc, f1 = accuracy_f1(sample_preds, sample_targets, task)
        accs.append(acc)
        f1s.append(f1)

    accs = np.array(accs)
    f1s = np.array(f1s)
    acc, f1 = accuracy_f1(preds, targets, task)
    return {
        "acc": acc,
        "f1": f1,
        "acc_lo": float(np.percentile(accs, 2.5)),
        "acc_hi": float(np.percentile(accs, 97.5)),
        "f1_lo": float(np.percentile(f1s, 2.5)),
        "f1_hi": float(np.percentile(f1s, 97.5)),
    }


def correct_map(samples: List[Dict], task: str) -> Dict[int, int]:
    preds = []
    targets = []
    idxs = []
    for i, sample in enumerate(samples):
        idx = sample.get("idx", i)
        try:
            idx = int(idx)
        except Exception:
            idx = i
        pred, target = extract_pred_target(sample)
        preds.append(pred)
        targets.append(target)
        idxs.append(idx)
    labels = get_labels(task)
    pred_idx = [_label_index(p, labels) for p in preds]
    true_idx = [_label_index(t, labels) for t in targets]
    return {idx: int(p == t) for idx, p, t in zip(idxs, pred_idx, true_idx)}


def win_tie_loss(ref: Dict[int, int], other: Dict[int, int]) -> Tuple[int, int, int]:
    common = sorted(set(ref.keys()) & set(other.keys()))
    win = tie = loss = 0
    for idx in common:
        r = ref[idx]
        o = other[idx]
        if r > o:
            win += 1
        elif r < o:
            loss += 1
        else:
            tie += 1
    return win, tie, loss


def average_head_weights(head_dir: Path) -> np.ndarray:
    cluster_dirs = sorted([p for p in head_dir.glob("cluster_*") if p.is_dir()])
    if not cluster_dirs:
        raise FileNotFoundError(f"No cluster_* dirs in {head_dir}")
    mats = []
    for cluster in cluster_dirs:
        weight_path = cluster / "head_weights.json"
        if not weight_path.exists():
            continue
        with weight_path.open("r") as f:
            payload = json.load(f)
        mats.append(np.array(payload["head_weights"], dtype=np.float32))
    if not mats:
        raise FileNotFoundError(f"No head_weights.json found in {head_dir}")
    return np.mean(np.stack(mats, axis=0), axis=0)


def topk_set(weights: np.ndarray, k: int) -> set[tuple[int, int]]:
    k = max(1, min(int(k), weights.size))
    flat = weights.flatten()
    idxs = np.argsort(flat)[::-1][:k]
    num_heads = weights.shape[1]
    return set((int(idx // num_heads), int(idx % num_heads)) for idx in idxs)


def jaccard(a: Iterable, b: Iterable) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))
