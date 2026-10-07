"""Every prompt Task 2 sends: the teacher's two prompts and the student's system prompt.

The notebook prints all three in full, because the brief asks for the teacher
prompts in the submission. The student's rules (STYLE_RULES) are shared word
for word with the teacher's docstring prompt, so the labels follow exactly
the rules the student is told.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 2 data script as designed in the grilling rounds', Date: 2026-10-07

from __future__ import annotations

from common.llm import Prompt

STYLE_RULES = """\
Docstring rules (Google style):
- First line: a one-line summary in the imperative mood ("Return ...", "Parse ..."), ending with a period.
- Keep it short. Add one paragraph after the summary only when a caller would be surprised without it, at most two sentences.
- Args: every parameter in signature order, one per line as "    name: description", at most 15 words each. Never document self or cls. Write *args and **kwargs without the stars.
- Returns: what the function returns. Leave the section out when it returns nothing, and for generators.
- Yields: what each iteration yields. Only for generators (functions that use yield).
- Raises: only exceptions raised by a raise statement in this function's own body, one per line as "    ExceptionName: If ...", naming the main conditions briefly. Leave the section out when there are none.
- Describe only what the code does. Never mention a parameter, exception or behaviour that is not in the code."""

# The student's system prompt, identical for the base and the fine-tuned model,
# so the comparison measures fine-tuning and nothing else.
STUDENT_SYSTEM = (
    "You write docstrings for Python functions. The user sends one function. Reply with only "
    "the docstring text: no quotes, no code fences, no code.\n\n" + STYLE_RULES
)

# Call 1: new functions for one recipe cell. Fresh code avoids testing on functions
# the student may have memorised, with their docstrings, during pretraining.
TEACHER_FUNCTIONS = Prompt(
    name="t2_teacher_functions",
    version="1",
    system=(
        "You write realistic Python functions for a dataset that teaches a small model to "
        "document code. Each request names a topic, a kind of function and how many functions "
        "to write at each difficulty level.\n\n"
        "Rules:\n"
        "- Each function does a different, realistic job within the topic; vary names, "
        "parameters and logic so no two are near copies.\n"
        "- Use type hints on parameters and return values.\n"
        "- No docstring, no comments, no decorators, no imports. Standard-library modules and "
        "names (json, re, math, asyncio, datetime, Path, Decimal, defaultdict ...) may be used as "
        "if already imported.\n"
        "- Exactly one top-level def per function. Helper logic goes inline, not in nested defs.\n"
        "- When a function raises, use a specific built-in exception (ValueError, KeyError, "
        "TypeError, FileNotFoundError, TimeoutError, ...) with a message.\n"
        "- Code must be valid Python 3.11 and correct."
    ),
    user=(
        "Topic: {topic} ({topic_hint})\n"
        "Kind: {kind_hint}\n"
        "Write {per_level} functions at each level:\n"
        "- simple: {simple}\n"
        "- medium: {medium}\n"
        "- tricky: {tricky}"
    ),
)

# Call 2 sees only the code, exactly as the student will, so the label describes the
# code as written rather than what the teacher meant it to do.
TEACHER_DOCSTRING = Prompt(
    name="t2_teacher_docstring",
    version="1",
    system=(
        "You are a senior Python reviewer writing the docstring for one function. Read the code "
        "carefully and describe exactly what it does. Fill the fields; they are assembled into a "
        "Google-style docstring.\n\n" + STYLE_RULES
    ),
    user="{code}",
)
