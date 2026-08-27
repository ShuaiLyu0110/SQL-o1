"""Merge schema-aware SFT pairs with a prioritized multi-model PSG pool.

The revised manuscript uses 16,428 original pairs and 4,826 deduplicated PSG
pairs (22.7% of the final cold-start set). This utility works with arbitrary
dataset sizes while preserving that ratio by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def load_examples(paths: list[str]) -> list[dict]:
    examples = []
    for filename in paths:
        with open(filename, encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {filename}")
        examples.extend(data)
    return examples


def example_key(example: dict) -> str:
    """Deduplicate the same SQL failure mined by different backbones."""
    stable = "\0".join(
        [str(example.get("db_id", "")), str(example.get("target", ""))]
    )
    # Older files lack db_id. Including the prompt prevents unrelated examples
    # with identical SQL strings from collapsing in that compatibility case.
    if not example.get("db_id"):
        stable += "\0" + str(example.get("input", ""))
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def build_cold_start_data(
    schema_examples: list[dict],
    failure_examples: list[dict],
    *,
    psg_ratio: float = 0.227,
    max_psg_examples: int | None = 4826,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    if not 0.0 <= psg_ratio < 1.0:
        raise ValueError("psg_ratio must be in [0, 1)")

    grouped: dict[str, list[dict]] = defaultdict(list)
    supporting_models: dict[str, set[str]] = defaultdict(set)
    for example in failure_examples:
        if "input" not in example or "target" not in example:
            raise ValueError("Every PSG example must contain input and target")
        key = example_key(example)
        grouped[key].append(example)
        model = str(example.get("failure_model", "unknown"))
        supporting_models[key].add(model)

    rng = random.Random(seed)
    tie_breakers = {key: rng.random() for key in grouped}
    ranked_keys = sorted(
        grouped,
        key=lambda key: (-len(supporting_models[key]), tie_breakers[key], key),
    )
    desired = round(len(schema_examples) * psg_ratio / (1.0 - psg_ratio)) if psg_ratio else 0
    if max_psg_examples is not None:
        desired = min(desired, max_psg_examples)
    selected_keys = ranked_keys[: min(desired, len(ranked_keys))]
    selected = [
        {"input": grouped[key][0]["input"], "target": grouped[key][0]["target"]}
        for key in selected_keys
    ]
    schema_clean = [
        {"input": example["input"], "target": example["target"]}
        for example in schema_examples
    ]
    combined = schema_clean + selected
    rng.shuffle(combined)
    metadata = {
        "schema_examples": len(schema_clean),
        "failure_pool_examples": len(failure_examples),
        "deduplicated_failures": len(grouped),
        "selected_psg_examples": len(selected),
        "actual_psg_ratio": len(selected) / len(combined) if combined else 0.0,
        "requested_psg_ratio": psg_ratio,
        "seed": seed,
    }
    return combined, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema_data", nargs="+", required=True)
    parser.add_argument("--failure_data", nargs="+", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--dataset_name", default="sql_o1_cold_start")
    parser.add_argument("--metadata_file", default="")
    parser.add_argument("--psg_ratio", type=float, default=0.227)
    parser.add_argument("--max_psg_examples", type=int, default=4826)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined, metadata = build_cold_start_data(
        load_examples(args.schema_data),
        load_examples(args.failure_data),
        psg_ratio=args.psg_ratio,
        max_psg_examples=args.max_psg_examples,
        seed=args.seed,
    )
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(combined, file, ensure_ascii=False, indent=2)
    dataset_info_file = output.parent / "dataset_info.json"
    if dataset_info_file.exists():
        with dataset_info_file.open(encoding="utf-8") as file:
            dataset_info = json.load(file)
    else:
        dataset_info = {}
    dataset_info[args.dataset_name] = {
        "file_name": output.name,
        "formatting": "alpaca",
        "columns": {"prompt": "input", "query": None, "response": "target"},
    }
    with dataset_info_file.open("w", encoding="utf-8") as file:
        json.dump(dataset_info, file, ensure_ascii=False, indent=2)
    metadata_file = Path(args.metadata_file) if args.metadata_file else output.with_suffix(".meta.json")
    with metadata_file.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
