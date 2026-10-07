"""Pydantic schemas for the teacher's two answers (sent as strict JSON schemas by common/llm.py).

Call 1 answers FunctionBatch: new functions for one recipe cell.
Call 2 answers DocstringParts: the docstring as fields, which our code renders
into one fixed Google-style layout (docstrings.render). Every training answer
therefore has the same layout, and the student learns one style, not the
teacher's day-to-day variation.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 2 data script as designed in the grilling rounds', Date: 2026-10-07

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["simple", "medium", "tricky"]


class GeneratedFunction(BaseModel):
    level: Level = Field(description="Which level of the brief this function was written for.")
    code: str = Field(
        description=(
            "The complete function source, with no docstring, no comments, no decorators and no "
            "imports. Imported names may be used as if already imported."
        )
    )


class FunctionBatch(BaseModel):
    functions: list[GeneratedFunction] = Field(
        description="The requested functions, in the order requested: every simple one, then medium, then tricky."
    )


class ArgDoc(BaseModel):
    name: str = Field(description="The parameter name exactly as in the signature, without * or **.")
    description: str = Field(description="At most 15 words: what the caller passes and any default's meaning.")


class RaiseDoc(BaseModel):
    exception: str = Field(description="The exception class name exactly as in the raise statement.")
    condition: str = Field(description="When it is raised, starting with 'If'.")


class DocstringParts(BaseModel):
    summary: str = Field(description="One line in the imperative mood, ending with a period, at most 80 characters.")
    details: str | None = Field(
        description="At most two sentences a caller would be surprised without; null in most cases."
    )
    args: list[ArgDoc] = Field(description="Every parameter in signature order, except self and cls.")
    returns: str | None = Field(
        description="What the function returns; null for generators and when it returns nothing."
    )
    yields: str | None = Field(description="What each iteration yields; null unless the function is a generator.")
    raises: list[RaiseDoc] = Field(
        description="Only exceptions raised by a raise statement in this function's own body."
    )
