"""Offline tests for Task 2's data pipeline: the ast check, the validator, the split and both teacher steps."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 2 data script as designed in the grilling rounds', Date: 2026-10-07

from __future__ import annotations

import pytest

from task2_genai import docstrings, generate_data
from task2_genai.dataset import build_record, read_jsonl, split
from task2_genai.prompts import STUDENT_SYSTEM
from task2_genai.recipes import FUNCTIONS_PER_LEVEL, LEVELS, all_cells
from task2_genai.schemas import (
    ArgDoc,
    DocstringParts,
    FunctionBatch,
    GeneratedFunction,
    RaiseDoc,
)
from tests.fakes import ScriptedLLM

DISCOUNT = '''\
def apply_discount(price: float, pct: float, cap: float | None = None) -> float:
    if pct < 0 or pct > 100:
        raise ValueError("pct must be 0-100")
    cut = price * pct / 100
    return price - min(cut, cap) if cap else price - cut'''

GOOD_DOC = """Apply a percentage discount to a price, optionally capped.

Args:
    price: Original price.
    pct: Discount percentage, from 0 to 100.
    cap: Largest discount allowed; no cap when None.

Returns:
    The price after the discount.

Raises:
    ValueError: If pct is outside 0-100."""


# --- analyze -----------------------------------------------------------------


def test_analyze_reads_params_returns_and_raises():
    facts = docstrings.analyze(DISCOUNT)
    assert facts.params == ("price", "pct", "cap")
    assert facts.returns_value and not facts.is_generator and not facts.is_async
    assert facts.raises == {"ValueError"}


def test_analyze_drops_self_and_strips_stars():
    code = "def m(self, a, *args, key=1, **kwargs):\n    return a"
    assert docstrings.analyze(code).params == ("a", "args", "key", "kwargs")


def test_analyze_ignores_yield_and_raise_in_nested_functions():
    code = (
        "def outer(xs):\n"
        "    def inner():\n"
        "        yield 1\n"
        "        raise KeyError('x')\n"
        "    return list(inner())"
    )
    facts = docstrings.analyze(code)
    assert not facts.is_generator and facts.raises == frozenset()


def test_analyze_treats_return_none_as_no_value():
    assert not docstrings.analyze("def f(x):\n    if x:\n        return None\n    print(x)").returns_value


def test_analyze_rejects_more_than_one_function():
    with pytest.raises(ValueError):
        docstrings.analyze("def a():\n    pass\n\ndef b():\n    pass")


# --- parse and check ---------------------------------------------------------


def test_check_correct_docstring():
    assert docstrings.check(DISCOUNT, GOOD_DOC).label == "correct"


def test_check_missing_raises_is_partial():
    doc = GOOD_DOC.split("\n\nRaises:")[0]
    result = docstrings.check(DISCOUNT, doc)
    assert result.label == "partial" and result.missing == ["raises ValueError"]


def test_check_invented_arg_is_hallucinated_even_with_something_missing():
    doc = GOOD_DOC.replace("    cap:", "    limit:")
    result = docstrings.check(DISCOUNT, doc)
    assert result.label == "hallucinated"
    assert result.invented == ["arg limit"] and result.missing == ["arg cap"]


def test_check_generator_needs_yields_not_returns():
    code = "def count(n: int):\n    for i in range(n):\n        yield i"
    assert docstrings.check(code, "Count up.\n\nArgs:\n    n: Limit.\n\nYields:\n    The next number.").label == "correct"
    assert docstrings.check(code, "Count up.\n\nArgs:\n    n: Limit.\n\nReturns:\n    Numbers.").label == "hallucinated"


def test_check_accepts_typed_entries_fences_and_quotes():
    doc = '```\n"""' + GOOD_DOC.replace("    price:", "    price (float):") + '"""\n```'
    assert docstrings.check(DISCOUNT, doc).label == "correct"


def test_check_unparseable_without_summary():
    assert docstrings.check(DISCOUNT, "Args:\n    price: x").label == "unparseable"
    assert docstrings.check(DISCOUNT, "").label == "unparseable"


def test_render_matches_the_layout_the_checker_reads():
    doc = docstrings.render(
        "Apply a percentage discount to a price, optionally capped.",
        None,
        [("price", "Original price."), ("pct", "Discount percentage, from 0 to 100."),
         ("cap", "Largest discount allowed; no cap when None.")],
        "The price after the discount.",
        None,
        [("ValueError", "If pct is outside 0-100.")],
    )
    assert doc == GOOD_DOC


# --- validate ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "kind", "ok"),
    [
        (DISCOUNT, "function", True),
        (DISCOUNT, "method", False),
        ("def m(self, x):\n    y = x * 2\n    z = y + 1\n    w = z - 3\n    return w", "method", True),
        ("def g(n):\n    i = 0\n    while i < n:\n        yield i\n        i += 1", "generator", True),
        ("async def f(c):\n    await c.open()\n    data = await c.read()\n    await c.close()\n    return data", "async", True),
        ("def f(x):\n    return x", "function", False),  # too short
        ('def f(x):\n    """Doc."""\n    a = 1\n    b = 2\n    return x', "function", False),  # has a docstring
        ("def f(:\n    pass", "function", False),  # syntax error
    ],
)
def test_validate(code, kind, ok):
    assert (generate_data.validate(code, kind) is None) is ok


# --- dataset -----------------------------------------------------------------


def test_record_uses_the_student_system_prompt():
    record = build_record({"id": "a", "topic": "t", "kind": "function", "level": "simple", "code": "c"}, "d")
    assert [m["role"] for m in record["messages"]] == ["system", "user", "assistant"]
    assert record["messages"][0]["content"] == STUDENT_SYSTEM


def test_split_is_80_10_10_by_kind_and_stable():
    records = [{"id": f"{kind}-{i:03d}", "kind": kind} for kind in ("function", "method", "generator", "async") for i in range(50)]
    parts = split(records)
    assert {name: len(rows) for name, rows in parts.items()} == {"train": 160, "val": 20, "test": 20}
    assert {r["kind"] for r in parts["test"]} == {"function", "method", "generator", "async"}
    assert split(list(reversed(records))) == parts
    ids = [r["id"] for rows in parts.values() for r in rows]
    assert len(ids) == len(set(ids))


def test_grid_has_one_cell_per_topic_and_kind():
    cells = all_cells()
    assert len(cells) == 48
    assert len(cells[0].recipe_ids()) == len(LEVELS) * FUNCTIONS_PER_LEVEL


# --- teacher steps -----------------------------------------------------------


@pytest.fixture
def data_files(tmp_path, monkeypatch):
    for name in ("FUNCTIONS_FILE", "REJECTED_FILE", "LABELS_FILE"):
        monkeypatch.setattr(generate_data, name, tmp_path / f"{name.lower()}.jsonl")
    return tmp_path


def test_generate_cell_keeps_valid_functions_and_records_rejects(data_files):
    cell = all_cells()[0]  # finance / function
    batch = FunctionBatch(
        functions=[
            GeneratedFunction(level="simple", code=DISCOUNT),
            GeneratedFunction(level="simple", code="def tiny(x):\n    return x"),  # too short
            GeneratedFunction(level="simple", code=DISCOUNT),  # a third simple one has no slot
            GeneratedFunction(level="medium", code=DISCOUNT),  # same name as the first
        ]
    )
    generate_data.generate_cell(ScriptedLLM([batch]), cell, seen_names=set())

    kept = read_jsonl(generate_data.FUNCTIONS_FILE)
    rejected = read_jsonl(generate_data.REJECTED_FILE)
    assert [f["id"] for f in kept] == ["finance-function-simple-01"]
    assert [r["id"] for r in rejected] == ["finance-function-simple-02", "finance-function-medium-01"]
    assert "lines" in rejected[0]["reason"] and "duplicate" in rejected[1]["reason"]


def test_generate_cell_failure_writes_nothing(data_files):
    generate_data.generate_cell(ScriptedLLM([None]), all_cells()[0], seen_names=set())
    assert read_jsonl(generate_data.FUNCTIONS_FILE) == [] and read_jsonl(generate_data.REJECTED_FILE) == []


def test_label_function_renders_and_checks(data_files):
    parts = DocstringParts(
        summary="Apply a percentage discount to a price, optionally capped.",
        details=None,
        args=[ArgDoc(name="price", description="Original price."), ArgDoc(name="pct", description="Percent.")],
        returns="The price after the discount.",
        yields=None,
        raises=[RaiseDoc(exception="ValueError", condition="If pct is outside 0-100.")],
    )
    llm = ScriptedLLM([parts])
    generate_data.label_function(llm, {"id": "x", "code": DISCOUNT})

    (row,) = read_jsonl(generate_data.LABELS_FILE)
    assert row["label"] == "partial" and row["missing"] == ["arg cap"]
    assert llm.requests[0]["variables"] == {"code": DISCOUNT}  # the teacher sees only the code


# --- metrics -----------------------------------------------------------------

from task2_genai import metrics


def test_structure_labels_per_answer():
    rows = metrics.structure_labels([DISCOUNT, DISCOUNT], [GOOD_DOC, "nonsense\n\nArgs:\n    x: y"])
    assert rows[0]["label"] == "correct"
    assert rows[1]["label"] == "hallucinated" and rows[1]["invented"] == ["arg x"]


def test_near_duplicates_ranks_copies_first():
    other = "def total(xs: list[int]) -> int:\n    s = 0\n    for x in xs:\n        s += x\n    return s"
    pairs = metrics.near_duplicates(["a", "b", "c"], [DISCOUNT, DISCOUNT.replace("cut", "cut2"), other], top=1)
    assert pairs[0][:2] == ("a", "b") and pairs[0][2] > metrics.NEAR_DUPLICATE_JACCARD


def test_keyword_views():
    assert metrics.name_words([DISCOUNT])["discount"] == 1
    code = "def f(s):\n    t = s.strip()\n    return re.sub('a', 'b', t) + json.dumps(t)"
    assert metrics.modules_used([code]) == {"re": 1, "json": 1}


def test_rouge_l_is_one_for_identical_text():
    pytest.importorskip("rouge_score")
    assert metrics.rouge_l(['"""' + GOOD_DOC + '"""'], [GOOD_DOC]) == [pytest.approx(1.0)]
