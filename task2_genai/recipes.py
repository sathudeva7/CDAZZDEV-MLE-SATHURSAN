"""The recipe grid that makes the dataset varied by design: topic x kind x level.

Each (topic, kind) cell is one teacher call that writes FUNCTIONS_PER_LEVEL
functions at each level, so every recipe id is fixed before generation starts.
A rerun skips ids it already has, so an interrupted run resumes without duplicates.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 2 data script as designed in the grilling rounds', Date: 2026-10-07

from __future__ import annotations

from dataclasses import dataclass

# Each topic carries a hint so the teacher does not drift to the same toy example.
TOPICS: dict[str, str] = {
    "finance": "prices, returns, interest, loans, portfolios, invoices, currency",
    "text": "parsing, tokenising, slugs, templates, normalising and searching strings",
    "datetime": "dates, durations, time zones, schedules, business days, calendars",
    "files": "reading and writing files, paths, CSV, JSON, config files, directories",
    "statistics": "means, medians, percentiles, moving averages, outliers, histograms",
    "data_cleaning": "missing values, deduplication, type coercion, records, column mapping",
    "http": "URLs, query strings, headers, status codes, retries, pagination, rate limits",
    "validation": "emails, passwords, ranges, schemas, identifiers, user input",
    "collections": "grouping, chunking, flattening, merging dicts, sorting, priority queues",
    "geometry": "points, distances, polygons, bounding boxes, angles, grids",
    "caching": "memoisation, expiry, LRU eviction, keys, invalidation, counters",
    "simulation": "dice, cards, queues, random walks, inventories, game turns",
}

# What makes each kind of function different to document.
KINDS: dict[str, str] = {
    "function": "a plain module-level function (def)",
    "method": "an instance method of a class, written alone with `self` as its first parameter",
    "generator": "a generator function that uses `yield` (it may also take parameters with defaults)",
    "async": "an `async def` coroutine that awaits something such as asyncio.sleep or an injected client",
}

LEVELS: dict[str, str] = {
    "simple": "5 to 12 lines, one or two parameters, little branching",
    "medium": "10 to 25 lines, two to four parameters, at least one default value",
    "tricky": (
        "15 to 40 lines, keyword-only or *args/**kwargs parameters, several branches, "
        "and at least one explicit `raise`"
    ),
}

FUNCTIONS_PER_LEVEL = 2


@dataclass(frozen=True)
class Cell:
    """One teacher call: every level of one topic and kind."""

    topic: str
    kind: str

    def recipe_ids(self) -> list[str]:
        return [
            recipe_id(self.topic, self.kind, level, n)
            for level in LEVELS
            for n in range(1, FUNCTIONS_PER_LEVEL + 1)
        ]


def recipe_id(topic: str, kind: str, level: str, n: int) -> str:
    return f"{topic}-{kind}-{level}-{n:02d}"


def all_cells() -> list[Cell]:
    return [Cell(topic, kind) for topic in TOPICS for kind in KINDS]
