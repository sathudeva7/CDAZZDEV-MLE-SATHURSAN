"""What a function really does, read with `ast`, and a check of a docstring against it.

The same check serves twice:

- It filters the teacher's labels: only docstrings labelled `correct` become
  training examples, so a teacher slip never reaches the student.
- It scores the student: every test answer gets a label and the facts behind it.

Labels (see task2_genai/README.md):

- correct: every real parameter documented and nothing extra, Returns or Yields
  matches the code, Raises lists exactly the explicitly raised exceptions.
- partial: nothing invented, but something is missing.
- hallucinated: names a parameter, exception, return value or yield the code
  does not have. Invented beats missing: an answer with both is hallucinated.
- unparseable: not a Google-style docstring at all (no summary line).

The summary line's meaning is not checked here; the manual review and the
LLM judge cover it.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 2 data script as designed in the grilling rounds', Date: 2026-10-07

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Literal

Label = Literal["correct", "partial", "hallucinated", "unparseable"]

SECTIONS = ("Args", "Returns", "Yields", "Raises")
# The receiver of a method or classmethod is never documented.
RECEIVERS = frozenset({"self", "cls"})
# A section entry is indented one level under its header: "    name (type): text".
ENTRY = re.compile(r"^ {4}(\*{0,2}[A-Za-z_][\w.]*)\s*(?:\([^)]*\))?\s*:")
HEADER = re.compile(r"^(Args|Arguments|Returns|Yields|Raises):\s*$")
FENCE = re.compile(r"^```[\w-]*\s*$|^\s*```\s*$", re.MULTILINE)


@dataclass(frozen=True)
class FunctionFacts:
    """What the code of one function shows, independent of any docstring."""

    name: str
    params: tuple[str, ...]  # in order, receiver dropped, * and ** stripped
    is_async: bool
    is_generator: bool
    returns_value: bool  # a `return <value>` other than `return None`
    raises: frozenset[str]  # exception names in the function's own `raise` statements


@dataclass(frozen=True)
class DocFacts:
    """What a Google-style docstring claims."""

    summary: str
    args: tuple[str, ...]
    has_returns: bool
    has_yields: bool
    raises: frozenset[str]


@dataclass(frozen=True)
class CheckResult:
    label: Label
    missing: list[str] = field(default_factory=list)
    invented: list[str] = field(default_factory=list)


def analyze(code: str) -> FunctionFacts:
    """Facts about the single top-level function in `code`.

    Raises SyntaxError or ValueError when `code` is not exactly one function.
    """
    tree = ast.parse(code)
    defs = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(defs) != 1 or len(tree.body) != 1:
        raise ValueError("code must be exactly one top-level function")
    fn = defs[0]

    a = fn.args
    names = [arg.arg for arg in (*a.posonlyargs, *a.args)]
    if names and names[0] in RECEIVERS:
        names = names[1:]
    if a.vararg:
        names.append(a.vararg.arg)
    names += [arg.arg for arg in a.kwonlyargs]
    if a.kwarg:
        names.append(a.kwarg.arg)

    own = list(_own_nodes(fn))
    return FunctionFacts(
        name=fn.name,
        params=tuple(names),
        is_async=isinstance(fn, ast.AsyncFunctionDef),
        is_generator=any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in own),
        returns_value=any(
            isinstance(n, ast.Return) and n.value is not None and not _is_none(n.value) for n in own
        ),
        raises=frozenset(name for n in own if isinstance(n, ast.Raise) and (name := _raised_name(n))),
    )


def _own_nodes(fn: ast.AST):
    """Every node in the function's body, skipping nested functions, lambdas and classes.

    A `yield` or `raise` inside a nested function belongs to that function.
    """
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(node))


def _is_none(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _raised_name(node: ast.Raise) -> str | None:
    """`raise ValueError(...)` and `raise errors.Bad` give the class name; a bare `raise` gives None."""
    exc = node.exc
    if isinstance(exc, ast.Call):
        exc = exc.func
    if isinstance(exc, ast.Name):
        return exc.id
    if isinstance(exc, ast.Attribute):
        return exc.attr
    return None  # bare re-raise, or a raised variable such as `raise err`


def clean_output(text: str) -> str:
    """A model answer with code fences and wrapping triple quotes removed.

    Applied the same way to the base and fine-tuned models, so neither is
    scored on wrapping the other also gets forgiven.
    """
    text = FENCE.sub("", text).strip()
    for quote in ('"""', "'''"):
        text = text.removeprefix(quote)
        text = text.removesuffix(quote)
    return text.strip()


def parse(docstring: str) -> DocFacts | None:
    """The claims in a Google-style docstring, or None when it has no summary line."""
    lines = clean_output(docstring).splitlines()
    if not lines or not lines[0].strip() or HEADER.match(lines[0].strip()):
        return None

    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines[1:]:
        header = HEADER.match(line.strip()) if not line.startswith(" ") else None
        if header:
            current = "Args" if header.group(1) == "Arguments" else header.group(1)
            sections.setdefault(current, [])
        elif current:
            sections[current].append(line)

    def entries(section: str) -> list[str]:
        return [m.group(1).lstrip("*") for line in sections.get(section, []) if (m := ENTRY.match(line))]

    return DocFacts(
        summary=lines[0].strip(),
        args=tuple(entries("Args")),
        has_returns="Returns" in sections,
        has_yields="Yields" in sections,
        raises=frozenset(name.rsplit(".", 1)[-1] for name in entries("Raises")),
    )


def check(code: str, docstring: str) -> CheckResult:
    """Label `docstring` against what `code` really does."""
    facts = analyze(code)
    doc = parse(docstring)
    if doc is None:
        return CheckResult("unparseable")

    missing: list[str] = []
    invented: list[str] = []

    missing += [f"arg {p}" for p in facts.params if p not in doc.args]
    invented += [f"arg {a}" for a in doc.args if a not in facts.params]

    if facts.is_generator:
        if not doc.has_yields:
            missing.append("Yields")
        if doc.has_returns:
            invented.append("Returns")  # a generator's return value is not what callers receive
    else:
        if doc.has_yields:
            invented.append("Yields")
        if facts.returns_value and not doc.has_returns:
            missing.append("Returns")
        if doc.has_returns and not facts.returns_value:
            invented.append("Returns")

    missing += [f"raises {e}" for e in sorted(facts.raises - doc.raises)]
    invented += [f"raises {e}" for e in sorted(doc.raises - facts.raises)]

    if invented:
        return CheckResult("hallucinated", missing, invented)
    if missing:
        return CheckResult("partial", missing, invented)
    return CheckResult("correct")


def render(
    summary: str,
    details: str | None,
    args: list[tuple[str, str]],
    returns: str | None,
    yields: str | None,
    raises: list[tuple[str, str]],
) -> str:
    """One Google-style docstring body, in the exact layout every training answer uses."""
    parts = [summary.strip()]
    if details and details.strip():
        parts.append(details.strip())
    if args:
        parts.append("Args:\n" + "\n".join(f"    {name}: {text.strip()}" for name, text in args))
    if returns and returns.strip():
        parts.append("Returns:\n    " + returns.strip())
    if yields and yields.strip():
        parts.append("Yields:\n    " + yields.strip())
    if raises:
        parts.append("Raises:\n" + "\n".join(f"    {name}: {text.strip()}" for name, text in raises))
    return "\n\n".join(parts)
