"""Generate the Task 2 dataset with the teacher model, then check, filter and split it.

    python -m task2_genai.generate_data            # generate what is missing, then split
    python -m task2_genai.generate_data --split    # only rebuild the splits from saved data

Steps, each saved to task2_genai/data/ so a rerun resumes where it stopped:

1. functions.jsonl: call 1 per recipe cell writes new functions; each is
   validated (one def, right kind, 5-40 lines, no docstring, unique name).
   Rejects go to rejected.jsonl with the reason.
2. labels.jsonl: call 2 per function sees only the code and fills
   DocstringParts; we render it and label it with docstrings.check.
3. train/val/test.jsonl: only `correct` labels, split 80/10/10 by kind.

The teacher is GPT-6.1 Sol, a deliberate exception to the free-tier setup
(docs/adr/0002). Its calls are logged to task2_genai/logs/teacher_calls.jsonl.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 2 data script as designed in the grilling rounds', Date: 2026-10-07

from __future__ import annotations

import argparse
import ast
import json
import logging
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import common.llm_openai  # noqa: F401  registers the openai_dev profile the teacher is built from
from common.llm import StructuredLLM, enable_console_logging
from common.llm_config import PROFILES
from task2_genai import docstrings
from task2_genai.dataset import (
    DATA_DIR,
    SPLITS,
    build_record,
    read_jsonl,
    split,
    write_jsonl,
)
from task2_genai.prompts import TEACHER_DOCSTRING, TEACHER_FUNCTIONS
from task2_genai.recipes import (
    FUNCTIONS_PER_LEVEL,
    KINDS,
    LEVELS,
    TOPICS,
    Cell,
    all_cells,
    recipe_id,
)
from task2_genai.schemas import DocstringParts, FunctionBatch

logger = logging.getLogger("task2_genai")

LOG_DIR = Path(__file__).parent / "logs"
FUNCTIONS_FILE = DATA_DIR / "functions.jsonl"
REJECTED_FILE = DATA_DIR / "rejected.jsonl"
LABELS_FILE = DATA_DIR / "labels.jsonl"

# Sol on the openai_dev profile, logging to its own committed file so the
# teacher's calls are part of the submission evidence.
TEACHER_PROFILE = replace(PROFILES["openai_dev"], name="t2_teacher", log_file="teacher_calls.jsonl")
# The openai_dev profile sends one step above this (low -> medium): enough to write
# and read 40-line functions carefully without long, costly thinking.
TEACHER_EFFORT = "low"
# Parallel teacher calls. The paid tier's rate limits allow far more; 8 keeps a
# run to a few minutes without bursts that trigger 429s.
WORKERS = 8
MIN_LINES = 5
MAX_LINES = 40

_write_lock = threading.Lock()


def _append(path: Path, row: dict[str, Any]) -> None:
    """Append one row as soon as it is ready, so a crash mid-run never loses finished work."""
    with _write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate(code: str, kind: str) -> str | None:
    """Why a generated function is unusable, or None when it is fine."""
    try:
        facts = docstrings.analyze(code)
    except (SyntaxError, ValueError) as exc:
        return f"not one valid function: {exc}"
    fn = ast.parse(code).body[0]
    if ast.get_docstring(fn) is not None:
        return "already has a docstring"
    lines = [line for line in code.splitlines() if line.strip()]
    if not MIN_LINES <= len(lines) <= MAX_LINES:
        return f"{len(lines)} lines, outside {MIN_LINES}-{MAX_LINES}"
    first = fn.args.args[0].arg if fn.args.args else None
    expected = {
        "function": not facts.is_async and not facts.is_generator and first not in docstrings.RECEIVERS,
        "method": first == "self" and not facts.is_async and not facts.is_generator,
        "generator": facts.is_generator and not facts.is_async,
        "async": facts.is_async and not facts.is_generator,
    }
    if not expected[kind]:
        return f"does not match kind {kind!r}"
    return None


def generate_cell(llm: StructuredLLM, cell: Cell, seen_names: set[str]) -> None:
    """Call 1: write the cell's functions, keep the valid ones, record the rest as rejects."""
    result = llm.call(
        TEACHER_FUNCTIONS,
        {
            "topic": cell.topic,
            "topic_hint": TOPICS[cell.topic],
            "kind_hint": KINDS[cell.kind],
            "per_level": FUNCTIONS_PER_LEVEL,
            **LEVELS,
        },
        FunctionBatch,
        fallback=FunctionBatch(functions=[]),
        reasoning_effort=TEACHER_EFFORT,
    )
    if not result.ok:
        logger.error("cell %s-%s failed, rerun to retry it: %s", cell.topic, cell.kind, result.error)
        return

    counts: Counter[str] = Counter()
    for generated in result.value.functions:
        counts[generated.level] += 1
        n = counts[generated.level]
        if n > FUNCTIONS_PER_LEVEL:
            continue  # the teacher wrote more than asked; the recipe has no slot for it
        rid = recipe_id(cell.topic, cell.kind, generated.level, n)
        code = generated.code.strip("\n")
        reason = validate(code, cell.kind)
        if reason is None:
            name = docstrings.analyze(code).name
            with _write_lock:
                if name in seen_names:
                    reason = f"duplicate function name {name!r}"
                else:
                    seen_names.add(name)
        row = {"id": rid, "topic": cell.topic, "kind": cell.kind, "level": generated.level, "code": code}
        if reason:
            logger.info("rejected %s: %s", rid, reason)
            _append(REJECTED_FILE, {**row, "reason": reason})
        else:
            _append(FUNCTIONS_FILE, row)


def label_function(llm: StructuredLLM, function: dict[str, Any]) -> None:
    """Call 2: the teacher documents the code alone; we render and check the docstring."""
    result = llm.call(
        TEACHER_DOCSTRING,
        {"code": function["code"]},
        DocstringParts,
        fallback=DocstringParts(summary="", details=None, args=[], returns=None, yields=None, raises=[]),
        reasoning_effort=TEACHER_EFFORT,
    )
    if not result.ok:
        logger.error("label %s failed, rerun to retry it: %s", function["id"], result.error)
        return
    parts = result.value
    docstring = docstrings.render(
        parts.summary,
        parts.details,
        [(a.name, a.description) for a in parts.args],
        parts.returns,
        parts.yields,
        [(r.exception, r.condition) for r in parts.raises],
    )
    verdict = docstrings.check(function["code"], docstring)
    _append(
        LABELS_FILE,
        {
            "id": function["id"],
            "docstring": docstring,
            "label": verdict.label,
            "missing": verdict.missing,
            "invented": verdict.invented,
        },
    )


def generate(workers: int = WORKERS) -> None:
    llm = StructuredLLM(log_dir=LOG_DIR, profile=TEACHER_PROFILE)

    attempted = {(r["topic"], r["kind"]) for r in read_jsonl(FUNCTIONS_FILE) + read_jsonl(REJECTED_FILE)}
    todo = [cell for cell in all_cells() if (cell.topic, cell.kind) not in attempted]
    seen_names = {docstrings.analyze(f["code"]).name for f in read_jsonl(FUNCTIONS_FILE)}
    logger.info("call 1: %d of %d cells to generate", len(todo), len(all_cells()))
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda cell: generate_cell(llm, cell, seen_names), todo))

    labelled = {r["id"] for r in read_jsonl(LABELS_FILE)}
    unlabelled = [f for f in read_jsonl(FUNCTIONS_FILE) if f["id"] not in labelled]
    logger.info("call 2: %d functions to label", len(unlabelled))
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda f: label_function(llm, f), unlabelled))


def build_splits() -> dict[str, int]:
    """Keep the `correct` labels, split them, write train/val/test.jsonl, and return the sizes."""
    functions = {f["id"]: f for f in read_jsonl(FUNCTIONS_FILE)}
    labels = read_jsonl(LABELS_FILE)
    kept = [build_record(functions[r["id"]], r["docstring"]) for r in labels if r["label"] == "correct"]
    logger.info(
        "labels: %s; keeping %d correct",
        dict(Counter(r["label"] for r in labels)),
        len(kept),
    )
    parts = split(kept)
    for name in SPLITS:
        write_jsonl(DATA_DIR / f"{name}.jsonl", parts[name])
    sizes = {name: len(parts[name]) for name in SPLITS}
    logger.info("split sizes: %s", sizes)
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", action="store_true", help="only rebuild the splits from saved data")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    enable_console_logging()
    if not args.split:
        generate(args.workers)
    build_splits()


if __name__ == "__main__":
    main()
