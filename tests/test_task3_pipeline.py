"""Offline tests for the Task 3B pipeline: tool restriction, typed handoffs, the critique loop and the cache.

The chat model and the LLM are scripted (tests/fakes.py), so these tests check
the pipeline's wiring, not the models' judgement.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 3: the 3B two-agent pipeline with the critique loop and the persistent cache, as designed in the grilling rounds', Date: 2026-10-07

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from task1_financial import news
from task3_agentic.cache import SCHEMA_VERSION, cache_path
from task3_agentic.handoff import (
    ANALYST_TOOLS,
    DEFAULT_REQUEST,
    PIPELINE_WHY,
    build_brief,
    write_request,
)
from task3_agentic.pipeline import WRITER_TOOLS, ResearchPipeline
from task3_agentic.printer import print_update
from task3_agentic.schemas import (
    ClarificationRequest,
    FinalReportAnswer,
    HeadlineLabel,
    HeadlineLabels,
    KeyObservationsAnswer,
)
from tests.fakes import (
    TODAY,
    ScriptedChat,
    ScriptedLLM,
    make_session,
    tool_turn,
    trace_lines,
)
from tests.test_task3_agent import report_answer

CLARIFY_VOLATILITY = ClarificationRequest(question="63-day volatility?", needs="volatility", period=None, window=63, headlines=None, why="horizon")


def rss(count: int = 3) -> bytes:
    items = "".join(f"<item><title>Story {i}</title><pubDate>Mon, 05 Oct 2026 0{i}:00:00 GMT</pubDate></item>" for i in range(count))
    return f"<rss><channel>{items}</channel></rss>".encode()


@pytest.fixture
def feeds(monkeypatch, fake_prices, fake_search):
    monkeypatch.setattr(news, "_download", lambda url: rss())
    fake_search.replies["text"] = [{"title": "Analyst cuts target", "href": "https://x", "body": "b"}]
    return fake_search


def analyst_turns(answer_turn=None) -> list:
    """A: price + sentiment, then stops. B: one web search, then stops. A answers the request (63-day volatility)."""
    return [
        tool_turn(("get_price_data", {"why": "trend first", "ticker": "AAPL"}), ("llm_sentiment", {"why": "score the headlines", "headlines": ["Story 0", "Story 1"]})),
        AIMessage("Brief ready."),
        tool_turn(("web_search", {"why": "commentary the brief lacks", "query": "AAPL risks"})),
        AIMessage("Research done."),
        answer_turn or tool_turn(("calculate_volatility", {"why": "B asked for 63-day volatility", "ticker": "AAPL", "window": 63})),
        AIMessage("63-day volatility is 24.4%."),
    ]


def final_answer() -> FinalReportAnswer:
    return FinalReportAnswer(**report_answer().model_dump(), clarification_used="The 63-day volatility widened the expected move.")


def make_pipeline(tmp_path, *, turns=None, llm_replies=None, failing_tools=()):
    labels = HeadlineLabels(labels=[HeadlineLabel(index=1, sentiment="negative", confidence=0.7, reason="r"), HeadlineLabel(index=2, sentiment="neutral", confidence=0.5, reason="r")])
    session = make_session(tmp_path, ScriptedLLM([labels]), failing_tools=failing_tools)
    chat = ScriptedChat(replies=turns if turns is not None else analyst_turns())
    llm = ScriptedLLM(llm_replies if llm_replies is not None else [KeyObservationsAnswer(key_observations=["a", "b", "c"]), CLARIFY_VOLATILITY, final_answer()])
    return ResearchPipeline(session, models=[chat], llm=llm, cache_dir=tmp_path / "cache"), chat, llm


def events(tmp_path) -> list[str]:
    return [line["event"] for line in trace_lines(tmp_path)]


# --- the full run ----------------------------------------------------------------------------


def test_each_agent_is_bound_to_its_own_tools_only(tmp_path, feeds):
    pipeline, chat, _ = make_pipeline(tmp_path)

    pipeline.run("AAPL")

    bound = [call["tools"] for call in chat.calls if call["tools"]]
    assert bound[0] == list(ANALYST_TOOLS)  # A's first turn
    assert bound[2] == list(WRITER_TOOLS)  # B's first turn
    assert all("web_search" not in tools for tools in (bound[0], bound[4]))


def test_writer_cannot_call_price_tools(tmp_path, feeds):
    turns = analyst_turns()
    turns[2] = tool_turn(("get_price_data", {"why": "want prices", "ticker": "AAPL"}))
    pipeline, _, _ = make_pipeline(tmp_path, turns=turns)

    pipeline.run("AAPL")

    refused = [line for line in trace_lines(tmp_path) if line.get("status") == "refused"]
    assert refused[0]["agent"] == "B" and refused[0]["tool"] == "get_price_data"


def test_brief_request_answer_and_report_flow_in_order(tmp_path, feeds):
    pipeline, _, _ = make_pipeline(tmp_path)

    run = pipeline.run("AAPL")

    assert run.brief.tools_called == ["get_price_data", "llm_sentiment"]
    assert run.brief.sentiment.score == pytest.approx(-0.35)  # (-0.7 + 0) / 2
    assert run.brief.hedge_levels.volatility_source == "get_price_data hv_30"
    assert run.request == CLARIFY_VOLATILITY
    assert run.response.fulfilled_by == "agent" and run.response.volatility.window == 63
    assert run.report.clarification_used.startswith("The 63-day volatility")
    assert "calculate_volatility" in run.report.tools_called  # cited via A's answer, though B never called it
    trace = events(tmp_path)
    assert trace[0] == "cache" and trace[-3:] == ["critique", "report", "cache"]  # lookup first, save last
    assert trace.count("critique") == 2  # one request, one response: the loop ran exactly once
    handoff = next(line for line in trace_lines(tmp_path) if line["event"] == "handoff")
    assert handoff["sender"] == "A" and handoff["receiver"] == "B" and handoff["content"]["ticker"] == "AAPL"


def test_news_intake_runs_once_for_agent_a(tmp_path, feeds):
    pipeline, chat, _ = make_pipeline(tmp_path)

    pipeline.run("AAPL")

    intake = [line for line in trace_lines(tmp_path) if line.get("tool") == "get_news"]
    assert len(intake) == 1 and intake[0]["agent"] == "pipeline"
    assert "- Story 0" in chat.calls[0]["messages"][-1].content


def test_pipeline_fetches_requested_data_when_agent_a_does_not(tmp_path, feeds):
    pipeline, _, _ = make_pipeline(tmp_path, turns=analyst_turns(answer_turn=AIMessage("I already know it.")))

    run = pipeline.run("AAPL")

    assert run.response.fulfilled_by == "pipeline"
    assert run.response.volatility.window == 63
    fallback = [line for line in trace_lines(tmp_path) if line.get("why") == PIPELINE_WHY]
    assert fallback[0]["agent"] == "A" and fallback[0]["tool"] == "calculate_volatility"


def test_every_model_failing_still_finishes_and_is_not_cached(tmp_path, feeds):
    pipeline, _, _ = make_pipeline(tmp_path, turns=[RuntimeError("down")] * 6, llm_replies=[None, None, None])

    run = pipeline.run("AAPL")

    assert run.brief.generated_by == "template"
    assert run.request == DEFAULT_REQUEST
    assert run.response.fulfilled_by == "pipeline"
    assert run.report.generated_by == "template"
    assert not run.cache_path.exists()
    assert any("not cached" in w for w in run.warnings)


def test_no_headlines_means_no_sentiment_in_the_brief(tmp_path, fake_prices, fake_search, monkeypatch):
    monkeypatch.setattr(news, "_download", lambda url: b"<rss><channel></channel></rss>")
    turns = analyst_turns()
    turns[0] = tool_turn(("get_price_data", {"why": "trend first", "ticker": "AAPL"}))
    pipeline, _, _ = make_pipeline(tmp_path, turns=turns)

    run = pipeline.run("AAPL")

    assert run.brief.headlines_given == 0 and run.brief.sentiment is None
    assert any("news intake found no headlines" in w for w in run.warnings)


# --- the cache ----------------------------------------------------------------------------------


def test_second_run_loads_the_cache_without_calling_a_tool(tmp_path, feeds):
    pipeline, chat, _ = make_pipeline(tmp_path)
    first = pipeline.run("AAPL")
    tool_calls, model_calls = events(tmp_path).count("tool_call"), len(chat.calls)

    second = pipeline.run("AAPL")

    assert first.cached is False and second.cached is True
    assert second.report == first.report and second.brief == first.brief
    assert events(tmp_path).count("tool_call") == tool_calls and len(chat.calls) == model_calls
    assert trace_lines(tmp_path)[-1]["action"] == "hit"
    assert first.cache_path == cache_path(tmp_path / "cache", "AAPL", TODAY)
    assert json.loads(first.cache_path.read_text())["schema_version"] == SCHEMA_VERSION


def test_force_refresh_skips_the_cache(tmp_path, feeds):
    make_pipeline(tmp_path)[0].run("AAPL")
    rebuilt, chat, _ = make_pipeline(tmp_path)  # fresh scripted replies for the second full run

    run = rebuilt.run("AAPL", force_refresh=True)

    assert run.cached is False and chat.calls
    assert any(line.get("action") == "skip" for line in trace_lines(tmp_path))


@pytest.mark.parametrize("content", ["{not json", json.dumps({"schema_version": SCHEMA_VERSION})])
def test_a_corrupt_cache_file_is_a_miss(tmp_path, feeds, content):
    path = cache_path(tmp_path / "cache", "AAPL", TODAY)
    path.parent.mkdir(parents=True)
    path.write_text(content)
    pipeline, chat, _ = make_pipeline(tmp_path)

    run = pipeline.run("AAPL")

    assert run.cached is False and chat.calls
    miss = next(line for line in trace_lines(tmp_path) if line.get("action") == "miss")
    assert miss["reason"].startswith("unreadable")


def test_an_old_schema_version_is_a_miss(tmp_path, feeds):
    pipeline, _, _ = make_pipeline(tmp_path)
    first = pipeline.run("AAPL")
    saved = json.loads(first.cache_path.read_text())
    first.cache_path.write_text(json.dumps({**saved, "schema_version": SCHEMA_VERSION - 1}))
    rebuilt, chat, _ = make_pipeline(tmp_path)

    run = rebuilt.run("AAPL")

    assert run.cached is False and chat.calls


# --- handoff builders and printing -------------------------------------------------------------------


def test_brief_falls_back_to_template_observations(tmp_path):
    brief, warnings = build_brief("AAPL", TODAY, [], 0, None)

    assert brief.generated_by == "template" and len(brief.key_observations) == 3
    assert brief.price is None and brief.hedge_levels is None
    assert "get_price_data was not called" in brief.data_gaps
    assert warnings


def test_a_sentiment_request_without_headlines_becomes_the_default(tmp_path):
    brief, _ = build_brief("AAPL", TODAY, [], 0, None)
    bad = ClarificationRequest(question="q", needs="sentiment", period=None, window=None, headlines=None, why="w")

    request, warnings = write_request(brief, [], ScriptedLLM([bad]))

    assert request == DEFAULT_REQUEST and warnings


def test_printer_shows_each_handoff(tmp_path, feeds):
    lines: list[str] = []
    pipeline, _, _ = make_pipeline(tmp_path)

    pipeline.run("AAPL", on_update=lambda agent, step: print_update(agent, step, write=lines.append))

    text = "\n".join(lines)
    for expected in (
        "[pipeline] cache miss",
        "[pipeline] news intake for Agent A (ok): 3 headlines",
        "[A -> B] handoff: DataBrief",
        "[B -> A] clarification request (volatility, {'window': 63})",
        "[A -> B] clarification answer (fulfilled by agent)",
        "clarification used: The 63-day volatility",
        "[pipeline] saved the research brief to",
    ):
        assert expected in text
