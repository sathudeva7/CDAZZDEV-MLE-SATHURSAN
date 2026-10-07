"""Training records in chat format, and the 80/10/10 split.

A record is one JSONL line: the recipe fields (id, topic, kind, level) for the
diversity tables, and `messages`, the system/user/assistant turns that the
tokenizer's own chat template turns into Qwen's <|im_start|> format at
training time.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 2 data script as designed in the grilling rounds', Date: 2026-10-07

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from task2_genai.prompts import STUDENT_SYSTEM

DATA_DIR = Path(__file__).parent / "data"
SPLIT_SEED = 42
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10
SPLITS = ("train", "val", "test")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_record(function: dict[str, Any], docstring: str) -> dict[str, Any]:
    """One training example: the student's system prompt, the code, and the checked docstring."""
    return {
        "id": function["id"],
        "topic": function["topic"],
        "kind": function["kind"],
        "level": function["level"],
        "messages": [
            {"role": "system", "content": STUDENT_SYSTEM},
            {"role": "user", "content": function["code"]},
            {"role": "assistant", "content": docstring},
        ],
    }


def split(records: list[dict[str, Any]], seed: int = SPLIT_SEED) -> dict[str, list[dict[str, Any]]]:
    """80/10/10 train/val/test, stratified by kind so each split holds all four kinds.

    Sorting by id before shuffling makes the split depend only on the seed, not on
    the order the teacher's answers arrived in.
    """
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in sorted(records, key=lambda r: r["id"]):
        by_kind[record["kind"]].append(record)

    rng = random.Random(seed)
    out: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
    for kind in sorted(by_kind):
        group = by_kind[kind]
        rng.shuffle(group)
        n_test = round(len(group) * TEST_FRACTION)
        n_val = round(len(group) * VAL_FRACTION)
        out["test"] += group[:n_test]
        out["val"] += group[n_test : n_test + n_val]
        out["train"] += group[n_test + n_val :]
    return out
