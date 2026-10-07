"""Scores and dataset statistics for Task 2, in plain Python so they run and test without a GPU.

- rouge_l: ROUGE-L F1 per answer (Google's rouge-score package).
- structure_labels: the ast check (docstrings.check) per answer, the automatic
  side of the hallucination measurement.
- name_words, modules_used, near_duplicates: the dataset diversity report.

BERTScore runs in the notebook, because it needs a GPU-sized model.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 2 notebook: QLoRA training, merge, and evaluation against the base model', Date: 2026-10-07

from __future__ import annotations

import ast
import itertools
import re
from collections import Counter
from typing import Any

from task2_genai import docstrings

# Two functions sharing this share of their character 5-grams are near copies.
NEAR_DUPLICATE_JACCARD = 0.8
SHINGLE_CHARS = 5
# Words too common in function names to say anything about the topic spread.
NAME_STOPWORDS = frozenset({"get", "set", "to", "from", "by", "of", "and", "the", "for", "in", "is", "a"})


def rouge_l(predictions: list[str], references: list[str]) -> list[float]:
    """ROUGE-L F1 of each cleaned prediction against its reference, with Porter stemming."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [
        scorer.score(ref, docstrings.clean_output(pred))["rougeL"].fmeasure
        for pred, ref in zip(predictions, references, strict=True)
    ]


def structure_labels(codes: list[str], predictions: list[str]) -> list[dict[str, Any]]:
    """The ast check's label, plus what is missing or invented, for each answer."""
    rows = []
    for code, pred in zip(codes, predictions, strict=True):
        result = docstrings.check(code, pred)
        rows.append({"label": result.label, "missing": result.missing, "invented": result.invented})
    return rows


def name_words(codes: list[str]) -> Counter[str]:
    """Words in the function names (snake_case split), a keyword view of the topic spread."""
    words: Counter[str] = Counter()
    for code in codes:
        name = docstrings.analyze(code).name
        words.update(w for w in name.lower().split("_") if w and w not in NAME_STOPWORDS)
    return words


def modules_used(codes: list[str]) -> Counter[str]:
    """Names used as the base of an attribute call (`re.sub`, `asyncio.sleep`), counted once per function."""
    counts: Counter[str] = Counter()
    for code in codes:
        bases = {
            node.func.value.id
            for node in ast.walk(ast.parse(code))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id not in {"self", "cls"}
            and node.func.value.id[0].islower()
            and node.func.value.id in _MODULE_NAMES
        }
        counts.update(bases)
    return counts


# Standard-library modules the teacher prompt allows; restricting the count to them
# keeps local variables (`text.split`) out of the table.
_MODULE_NAMES = frozenset(
    {
        "asyncio", "bisect", "collections", "csv", "datetime", "decimal", "functools", "hashlib", "heapq",
        "itertools", "json", "math", "operator", "os", "pathlib", "random", "re", "shutil", "statistics",
        "string", "time", "urllib", "uuid", "zoneinfo",
    }
)


def _shingles(code: str) -> set[str]:
    text = re.sub(r"\s+", " ", code)
    return {text[i : i + SHINGLE_CHARS] for i in range(max(1, len(text) - SHINGLE_CHARS + 1))}


def near_duplicates(ids: list[str], codes: list[str], top: int = 5) -> list[tuple[str, str, float]]:
    """The `top` most similar pairs by Jaccard similarity of character 5-grams, most similar first."""
    shingles = [_shingles(code) for code in codes]
    pairs = []
    for i, j in itertools.combinations(range(len(codes)), 2):
        union = len(shingles[i] | shingles[j])
        pairs.append((ids[i], ids[j], len(shingles[i] & shingles[j]) / union if union else 0.0))
    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs[:top]
