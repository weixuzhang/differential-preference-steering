#!/usr/bin/env python3
"""
Offline sanity checks for cached models and datasets.
"""

import argparse
import os
import sys
from typing import List

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BANDITPR_ROOT = os.environ.get(
    "BANDITPR_ROOT", os.path.join(_REPO_ROOT, "third_party", "banditpr")
)
sys.path.append(os.path.join(_BANDITPR_ROOT, "src"))
from lamp import load_lamp_dataset


def _log_ok(message: str) -> None:
    print(f"[OK] {message}")


def _log_fail(message: str) -> None:
    print(f"[FAIL] {message}")


def check_snapshot(repo_id: str, cache_dir: str) -> None:
    snapshot_download(repo_id=repo_id, cache_dir=cache_dir, local_files_only=True)


def check_tokenizer(repo_id: str) -> None:
    AutoTokenizer.from_pretrained(repo_id, local_files_only=True)


def check_embedding_model(repo_id: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(repo_id, local_files_only=True)
    model = AutoModel.from_pretrained(repo_id, local_files_only=True)
    model.eval()
    inputs = tokenizer("sanity check", return_tensors="pt")
    with torch.no_grad():
        _ = model(**inputs)


def check_lamp_tasks(tasks: List[str], split: str) -> None:
    for task in tasks:
        ds = load_lamp_dataset(task, split=split)
        _ = len(ds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline sanity check for caches")
    parser.add_argument(
        "--models",
        nargs="*",
        default=[
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "Qwen/Qwen2-7B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "mistralai/Mistral-7B-v0.3",
        ],
    )
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--lamp_tasks",
        nargs="*",
        default=["LaMP-1", "LaMP-2", "LaMP-3", "LaMP-4", "LaMP-5", "LaMP-7"],
    )
    parser.add_argument(
        "--longlamp_tasks",
        nargs="*",
        default=["LongLaMP-2", "LongLaMP-3", "LongLaMP-4"],
    )
    parser.add_argument("--skip_longlamp", action="store_true")
    parser.add_argument("--skip_embedding", action="store_true")
    parser.add_argument(
        "--cache_dir",
        default=os.environ.get("HF_HOME", "/scratch/weixuz/decore/.cache/huggingface"),
    )

    args = parser.parse_args()

    os.environ.setdefault("LAMP_DATA_ROOT", "/scratch/weixuz/lamp_data")

    failures = 0
    print("Offline mode:", os.environ.get("HF_OFFLINE", "unset"))
    print("Cache dir:", args.cache_dir)

    for model in args.models:
        try:
            check_snapshot(model, args.cache_dir)
            check_tokenizer(model)
            _log_ok(f"Model cached: {model}")
        except Exception as exc:
            _log_fail(f"Model missing: {model} ({exc})")
            failures += 1

    if not args.skip_embedding:
        try:
            check_embedding_model(args.embedding_model)
            _log_ok(f"Embedding model cached: {args.embedding_model}")
        except Exception as exc:
            _log_fail(f"Embedding model missing: {args.embedding_model} ({exc})")
            failures += 1

    try:
        check_lamp_tasks(args.lamp_tasks, split="dev")
        _log_ok("LaMP dev datasets available")
    except Exception as exc:
        _log_fail(f"LaMP dev datasets missing or unreadable ({exc})")
        failures += 1

    if not args.skip_longlamp:
        try:
            check_lamp_tasks(args.longlamp_tasks, split="dev")
            _log_ok("LongLaMP dev datasets available")
        except Exception as exc:
            _log_fail(f"LongLaMP dev datasets missing or unreadable ({exc})")
            failures += 1

    try:
        check_lamp_tasks(args.lamp_tasks, split="train")
        _log_ok("LaMP train datasets available")
    except Exception as exc:
        _log_fail(f"LaMP train datasets missing or unreadable ({exc})")
        failures += 1

    if not args.skip_longlamp:
        try:
            check_lamp_tasks(args.longlamp_tasks, split="train")
            _log_ok("LongLaMP train datasets available")
        except Exception as exc:
            _log_fail(f"LongLaMP train datasets missing or unreadable ({exc})")
            failures += 1

    if failures:
        print(f"\nSanity check failed: {failures} issue(s)")
        sys.exit(1)

    print("\nSanity check passed.")


if __name__ == "__main__":
    main()
