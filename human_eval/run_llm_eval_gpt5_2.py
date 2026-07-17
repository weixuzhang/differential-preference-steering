#!/usr/bin/env python3
"""
Run LLM evaluation (GPT-5.2) for DPS vs CAD human-eval JSONL files.

Example:
  python human_eval/run_llm_eval_gpt5_2.py \
    --inputs human_eval/LaMP-4_dps_vs_cad.jsonl human_eval/LaMP-7_dps_vs_cad.jsonl \
    --output human_eval/llm_eval_gpt5_2.jsonl \
    --summary human_eval/llm_eval_gpt5_2_summary.json \
    --model gpt-5.2 \
    --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI


SYSTEM_PROMPT = (
    "You are a careful evaluator. Follow the rubric and respond in JSON only."
)

USER_PROMPT_TEMPLATE = """\
Task: {task}
Profile:
{profile}

Input:
{input_text}

Output A:
{output_a}

Output B:
{output_b}

Please judge:
1) Which output is better overall for this task?
2) Which output is more aligned with the profile?
3) Provide 1-5 ratings for each output on the criteria below.

Return JSON with this schema:
{{
  "win": "A|B|tie",
  "alignment_win": "A|B|tie",
  "ratings": {{
    "A": {{"relevance": 1-5, "fluency": 1-5, "style": 1-5, "alignment": 1-5, "factuality": 1-5}},
    "B": {{"relevance": 1-5, "fluency": 1-5, "style": 1-5, "alignment": 1-5, "factuality": 1-5}}
  }},
  "notes": "brief reason"
}}
"""

CRITERIA = ["relevance", "fluency", "style", "alignment", "factuality"]


def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def deterministic_mapping(item_id: str, seed: int) -> Dict[str, str]:
    digest = hashlib.md5(f"{seed}:{item_id}".encode("utf-8")).digest()
    if digest[0] % 2 == 0:
        return {"A": "dps", "B": "cad"}
    return {"A": "cad", "B": "dps"}


def build_prompt(
    task: str,
    profile: str,
    input_text: str,
    output_a: str,
    output_b: str,
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        task=task,
        profile=profile,
        input_text=input_text,
        output_a=output_a,
        output_b=output_b,
    )


def extract_text_from_response(resp: Any) -> str:
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text
    try:
        # Responses API structure
        for item in resp.output:
            for content in item.content:
                if getattr(content, "type", "") == "output_text":
                    return content.text
    except Exception:
        pass
    try:
        # Chat completions
        return resp.choices[0].message.content
    except Exception:
        return str(resp)


def call_openai(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    temperature: float,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        resp = client.responses.create(
            model=model,
            input=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format={"type": "json_object"},
        )
        return extract_text_from_response(resp)
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_output_tokens,
            response_format={"type": "json_object"},
        )
        return extract_text_from_response(resp)


def parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
    raise ValueError("Could not parse JSON from response")


def normalize_rating(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        num = int(round(float(value)))
    except Exception:
        return None
    if num < 1:
        return 1
    if num > 5:
        return 5
    return num


def normalize_choice(choice: Any) -> Optional[str]:
    if choice is None:
        return None
    choice = str(choice).strip().lower()
    if choice in {"a", "b", "tie"}:
        return choice
    return None


def map_results(
    raw: Dict[str, Any], mapping: Dict[str, str]
) -> Dict[str, Any]:
    win = normalize_choice(raw.get("win"))
    alignment_win = normalize_choice(raw.get("alignment_win"))

    def map_choice(choice: Optional[str]) -> str:
        if choice == "a":
            return mapping["A"]
        if choice == "b":
            return mapping["B"]
        return "tie"

    mapped = {
        "winner": map_choice(win),
        "alignment_winner": map_choice(alignment_win),
        "ratings": {"dps": {}, "cad": {}},
    }

    ratings = raw.get("ratings", {})
    for label in ["A", "B"]:
        side = mapping[label]
        side_ratings = ratings.get(label, {}) if isinstance(ratings, dict) else {}
        for crit in CRITERIA:
            mapped["ratings"][side][crit] = normalize_rating(
                side_ratings.get(crit)
            )
    return mapped


def summarize(results_path: str, model: str) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    overall = _init_stats()

    for rec in load_jsonl(results_path):
        task = rec.get("task", "unknown")
        if task not in stats:
            stats[task] = _init_stats()
        _update_stats(stats[task], rec)
        _update_stats(overall, rec)

    summary = {
        "model": model,
        "per_task": stats,
        "overall": overall,
    }
    return summary


def _init_stats() -> Dict[str, Any]:
    return {
        "count": 0,
        "wins": {"dps": 0, "cad": 0, "tie": 0},
        "alignment_wins": {"dps": 0, "cad": 0, "tie": 0},
        "rating_sums": {
            "dps": {c: 0 for c in CRITERIA},
            "cad": {c: 0 for c in CRITERIA},
        },
        "rating_counts": {
            "dps": {c: 0 for c in CRITERIA},
            "cad": {c: 0 for c in CRITERIA},
        },
    }


def _update_stats(stats: Dict[str, Any], rec: Dict[str, Any]) -> None:
    parsed = rec.get("parsed", {})
    winner = parsed.get("winner", "tie")
    alignment = parsed.get("alignment_winner", "tie")

    stats["count"] += 1
    if winner in stats["wins"]:
        stats["wins"][winner] += 1
    else:
        stats["wins"]["tie"] += 1

    if alignment in stats["alignment_wins"]:
        stats["alignment_wins"][alignment] += 1
    else:
        stats["alignment_wins"]["tie"] += 1

    ratings = parsed.get("ratings", {})
    for side in ["dps", "cad"]:
        for crit in CRITERIA:
            val = None
            if isinstance(ratings, dict):
                val = ratings.get(side, {}).get(crit)
            if val is not None:
                stats["rating_sums"][side][crit] += val
                stats["rating_counts"][side][crit] += 1


def finalize_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    def finalize_block(block: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(block)
        out["avg_ratings"] = {"dps": {}, "cad": {}}
        for side in ["dps", "cad"]:
            for crit in CRITERIA:
                count = block["rating_counts"][side][crit]
                if count:
                    out["avg_ratings"][side][crit] = (
                        block["rating_sums"][side][crit] / count
                    )
                else:
                    out["avg_ratings"][side][crit] = None
        return out

    summary["overall"] = finalize_block(summary["overall"])
    for task, block in list(summary["per_task"].items()):
        summary["per_task"][task] = finalize_block(block)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input JSONL files with DPS vs CAD predictions",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL for LLM eval results",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional summary JSON output path",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="OpenAI model name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed for deterministic A/B assignment",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip items already in output file",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=-1,
        help="Limit number of items per input file",
    )
    parser.add_argument(
        "--num-passes",
        type=int,
        default=1,
        help="Number of evaluation passes per item",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep seconds between API calls",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=512,
        help="Max tokens for the LLM output",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (defaults to OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional OpenAI base URL (defaults to OPENAI_BASE_URL)",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")

    client = OpenAI(api_key=api_key, base_url=base_url)

    done_ids = set()
    if args.resume and os.path.exists(args.output):
        for rec in load_jsonl(args.output):
            if "id" in rec:
                done_ids.add(rec["id"])

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_f = open(args.output, "a", encoding="utf-8")

    total_processed = 0
    for input_path in args.inputs:
        count = 0
        for sample in load_jsonl(input_path):
            if args.max_items >= 0 and count >= args.max_items:
                break

            sample_id = str(sample.get("id", sample.get("idx", count)))
            if args.resume and sample_id in done_ids:
                count += 1
                continue

            mapping = deterministic_mapping(sample_id, args.seed)
            output_a = sample["dps_prediction"] if mapping["A"] == "dps" else sample["cad_prediction"]
            output_b = sample["dps_prediction"] if mapping["B"] == "dps" else sample["cad_prediction"]

            user_prompt = build_prompt(
                task=sample.get("task", "unknown"),
                profile=sample.get("profile", ""),
                input_text=sample.get("input", ""),
                output_a=output_a,
                output_b=output_b,
            )

            raw_responses: List[Dict[str, Any]] = []
            parsed_passes: List[Dict[str, Any]] = []
            errors: List[str] = []

            for _ in range(args.num_passes):
                try:
                    text = call_openai(
                        client=client,
                        model=args.model,
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        max_output_tokens=args.max_output_tokens,
                        temperature=args.temperature,
                    )
                    raw = parse_json(text)
                    raw_responses.append(raw)
                    parsed_passes.append(map_results(raw, mapping))
                except Exception as exc:
                    errors.append(str(exc))
                if args.sleep:
                    time.sleep(args.sleep)

            if parsed_passes:
                parsed = _aggregate_passes(parsed_passes)
            else:
                parsed = {}

            record = {
                "id": sample_id,
                "idx": sample.get("idx"),
                "task": sample.get("task"),
                "model": args.model,
                "mapping": mapping,
                "parsed": parsed,
                "raw": raw_responses,
                "errors": errors,
            }

            out_f.write(json.dumps(record, ensure_ascii=True) + "\n")
            out_f.flush()

            count += 1
            total_processed += 1

    out_f.close()

    if args.summary:
        summary_dir = os.path.dirname(args.summary)
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        summary = summarize(args.output, args.model)
        summary = finalize_summary(summary)
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=True, indent=2)

    print(f"Done. Wrote {total_processed} items to {args.output}")


def _aggregate_passes(passes: List[Dict[str, Any]]) -> Dict[str, Any]:
    def majority(choices: List[str]) -> str:
        counts = {"dps": 0, "cad": 0, "tie": 0}
        for c in choices:
            if c in counts:
                counts[c] += 1
            else:
                counts["tie"] += 1
        if counts["dps"] > counts["cad"] and counts["dps"] > counts["tie"]:
            return "dps"
        if counts["cad"] > counts["dps"] and counts["cad"] > counts["tie"]:
            return "cad"
        return "tie"

    wins = [p.get("winner", "tie") for p in passes]
    aligns = [p.get("alignment_winner", "tie") for p in passes]

    agg = {
        "winner": majority(wins),
        "alignment_winner": majority(aligns),
        "ratings": {"dps": {}, "cad": {}},
    }
    for side in ["dps", "cad"]:
        for crit in CRITERIA:
            vals = [
                p.get("ratings", {}).get(side, {}).get(crit)
                for p in passes
                if p.get("ratings", {}).get(side, {}).get(crit) is not None
            ]
            if vals:
                agg["ratings"][side][crit] = sum(vals) / len(vals)
            else:
                agg["ratings"][side][crit] = None
    return agg


if __name__ == "__main__":
    main()
