"""Offline tests for the Task 3A agent loop, report, hedge levels, printer and short-term memory.

The chat model is ScriptedChat (tests/fakes.py): each test scripts the
model's turns, so these tests check the loop's wiring (budgets, refusals,
hints reaching the model, memory), not the model's judgement.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 2: the 3A agent loop, report, hedge levels, printer and short-term memory, as designed in the grilling rounds', Date: 2026-10-07

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from common.llm_config import PROFILES
from task3_agentic import agent as agent_module
from task3_agentic.agent import MAX_TOOL_CALLS, MAX_TURNS, ResearchAgent, chat_models
from task3_agentic.printer import print_replan_cycles, print_update, replan_cycles
from task3_agentic.report import (
    ONE_SD_HIGH,
    ONE_SD_LOW,
    compute_hedge_levels,
    data_gaps,
    expected_move,
    write_report,
)
from task3_agentic.schemas import (
    EvidenceAnswer,
    HedgeAnswer,
    HedgeLegAnswer,
    ReportAnswer,
    RiskAnswer,
)
from task3_agentic.tools import (
    CALCULATE_VOLATILITY,
    GET_NEWS,
    GET_PRICE_DATA,
    LLM_SENTIMENT,
    TOOL_NAMES,
    WEB_SEARCH,
)
from task3_agentic.trace import TRACE_FILE_NAME, read_trace
from tests.fakes import ScriptedChat, ScriptedLLM, make_session, tool_turn, trace_lines

AGENT_A_TOOLS = (GET_PRICE_DATA, CALCULATE_VOLATILITY, LLM_SENTIMENT)
SUMMARY = "Price is above both averages. Volatility is calm. Sentiment is mixed."


def call(tool: str, why: str = "because", **args) -> tuple[str, dict]:
    return tool, {"why": why, **args}


def report_answer(*, evidence_tool="get_price_data", put_level=ONE_SD_LOW) -> ReportAnswer:
    risk = RiskAnswer(title="Risk", explanation="It could fall.", evidence=[EvidenceAnswer(fact="close 100", source_tool=evidence_tool)])
    return ReportAnswer(
        financial_health_summary=SUMMARY,
        top_risks=[risk, risk, risk],
        hedge=HedgeAnswer(strategy="collar", legs=[HedgeLegAnswer(action="buy", instrument="put", level=put_level), HedgeLegAnswer(action="sell", instrument="call", level=ONE_SD_HIGH)], rationale="r"),
    )


def observations_from(session, *calls) -> list[dict]:
    """Observations shaped like the agent loop's, from real tool calls on the fake data."""
    results = [session.call(tool, args) for tool, args in calls]
    return [
        {"tool": r.tool, "args": args, "why": None, "status": r.status, "digest": r.for_llm(), "result": r.model_dump(mode="json")}
        for r, (_, args) in zip(results, calls, strict=True)
    ]


def make_agent(tmp_path, replies, llm_replies=(), **kwargs) -> tuple[ResearchAgent, ScriptedChat, ScriptedLLM]:
    session = make_session(tmp_path, failing_tools=kwargs.pop("failing_tools", ()))
    chat = ScriptedChat(replies=list(replies))
    llm = ScriptedLLM(list(llm_replies))
    return ResearchAgent(session, models=[chat], llm=llm, **kwargs), chat, llm


# --- hedge levels ----------------------------------------------------------------------


def test_expected_move_hand_checked():
    # 200 x 0.30 x sqrt(63/252) = 200 x 0.30 x 0.5 = 30
    assert expected_move(200.0, 30.0) == pytest.approx(30.0)


def test_hedge_levels_bracket_the_price_by_one_expected_move(tmp_path, fake_prices):
    obs = observations_from(make_session(tmp_path), (GET_PRICE_DATA, {"ticker": "AAPL"}), (CALCULATE_VOLATILITY, {"ticker": "AAPL", "window": 30}))

    levels = compute_hedge_levels(obs)

    named = levels.by_name()
    assert levels.volatility_source == "calculate_volatility, 30-day"
    assert levels.expected_move == pytest.approx(expected_move(levels.price, levels.volatility_pct), abs=0.01)
    assert named[ONE_SD_LOW].level == pytest.approx(levels.price - levels.expected_move, abs=0.01)
    assert named[ONE_SD_HIGH].level == pytest.approx(levels.price + levels.expected_move, abs=0.01)
    assert {"sma_200", "bb_lower", "low_52w"} <= set(named)


def test_hedge_levels_use_hv_30_without_a_volatility_call_and_need_prices(tmp_path, fake_prices):
    session = make_session(tmp_path)

    assert compute_hedge_levels(observations_from(session, (GET_PRICE_DATA, {"ticker": "AAPL"}))).volatility_source == "get_price_data hv_30"
    assert compute_hedge_levels([]) is None


# --- the report ---------------------------------------------------------------------------


def test_report_drops_evidence_from_uncalled_tools_and_fills_levels_by_name(tmp_path, fake_prices):
    session = make_session(tmp_path)
    obs = observations_from(session, (GET_PRICE_DATA, {"ticker": "AAPL"}))
    llm = ScriptedLLM([report_answer(evidence_tool="web_search", put_level="made_up_level")])

    report = write_report("AAPL", session.today, obs, llm)

    assert report.generated_by == "llm"
    assert all(risk.evidence == [] for risk in report.top_risks)
    assert any("citing web_search" in w for w in report.warnings)
    put, sold_call = report.hedge.legs
    assert put.level is None and any("made_up_level" in w for w in report.warnings)
    assert sold_call.level == compute_hedge_levels(obs).by_name()[ONE_SD_HIGH].level
    assert report.tools_called == [GET_PRICE_DATA]


def test_report_falls_back_to_a_template_with_three_risks(tmp_path, fake_prices):
    session = make_session(tmp_path)
    obs = observations_from(session, (GET_PRICE_DATA, {"ticker": "AAPL"}), (CALCULATE_VOLATILITY, {"ticker": "AAPL", "window": 30}))

    for llm in (ScriptedLLM([None]), None):
        report = write_report("AAPL", session.today, obs, llm)
        assert report.generated_by == "template"
        assert len(report.top_risks) == 3
        assert report.hedge.strategy == "protective_put" and report.hedge.legs[0].level is not None
        assert "written from a template" in report.to_markdown()


def test_template_without_prices_suggests_trimming(tmp_path):
    report = write_report("ZZZ", make_session(tmp_path).today, [], None)

    assert report.hedge.strategy == "trim_and_stop"
    assert report.hedge.levels is None


# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'ya run fallback to template report (no LLM report when no price data came back)', Date: 2026-10-07
def test_no_price_data_means_a_template_report_without_asking_the_llm(tmp_path, fake_search):
    fake_search.replies["text"] = [{"title": "FiscalNote (NOTE) stock price", "href": "https://x", "body": "b"}]
    session = make_session(tmp_path)
    obs = observations_from(session, (WEB_SEARCH, {"query": "ZZZ stock"}))
    llm = ScriptedLLM([report_answer(evidence_tool=WEB_SEARCH)])

    report = write_report("ZZZ", session.today, obs, llm)

    assert report.generated_by == "template"
    assert llm.replies  # the scripted answer was never asked for
    assert report.warnings[0].startswith("no price data came back")
    assert "Not written by the LLM: no price data came back, so the report was written from a template" in report.to_markdown()


def test_data_gaps_name_uncalled_and_failed_tools():
    obs = [{"tool": GET_NEWS, "status": "error", "result": {"error": "feed down", "warnings": []}}]

    gaps = data_gaps(obs, (GET_NEWS, WEB_SEARCH))

    assert gaps == ["get_news returned no usable result (error: feed down)", "web_search was not called"]


# --- the agent loop ----------------------------------------------------------------------


def test_agent_sees_a_failure_and_its_hint_then_switches_tool(tmp_path, fake_prices, fake_search):
    fake_search.replies["text"] = [{"title": "Analyst cuts target", "href": "https://x", "body": "b"}]
    agent, chat, _ = make_agent(
        tmp_path,
        [
            tool_turn(call(GET_PRICE_DATA, ticker="AAPL"), call(GET_NEWS, ticker="AAPL")),
            tool_turn(call(WEB_SEARCH, why="get_news failed; its hint says web_search", query="AAPL news")),
            AIMessage("Enough evidence."),
        ],
        [report_answer()],
        failing_tools=[GET_NEWS],
    )

    run = agent.research("AAPL")

    second_turn_saw = [m for m in chat.calls[1]["messages"] if isinstance(m, ToolMessage)]
    assert '"status": "error"' in second_turn_saw[1].content and "web_search" in second_turn_saw[1].content
    assert [line["tool"] for line in trace_lines(tmp_path) if line["event"] == "tool_call"] == [GET_PRICE_DATA, GET_NEWS, WEB_SEARCH]
    assert run.observations[2]["why"] == "get_news failed; its hint says web_search"
    assert run.report.generated_by == "llm"
    assert any("get_news returned no usable result" in gap for gap in run.report.data_gaps)


def test_tool_budget_caps_calls_then_goes_straight_to_the_report(tmp_path, fake_prices):
    # A new message per turn, as a real model sends: add_messages replaces a message whose id it has seen.
    turns = [tool_turn(*[call(GET_PRICE_DATA, ticker="AAPL")] * 3) for _ in range(3)]
    agent, chat, _ = make_agent(tmp_path, turns, [report_answer()])

    run = agent.research("AAPL")

    assert run.tool_calls == MAX_TOOL_CALLS
    assert [obs["status"] for obs in run.observations].count("skipped") == 9 - MAX_TOOL_CALLS
    assert len(chat.calls) == 3  # no fourth model turn: the budget sent the run to the report
    assert run.report is not None


def test_turn_cap_ends_the_research(tmp_path, fake_prices):
    agent, chat, _ = make_agent(tmp_path, [tool_turn(call(CALCULATE_VOLATILITY, ticker="AAPL", window=w)) for w in range(10, 20)], [report_answer()])

    run = agent.research("AAPL")

    assert len(chat.calls) == MAX_TURNS
    assert run.tool_calls == MAX_TURNS


def test_an_agent_cannot_use_a_tool_it_was_not_given(tmp_path, fake_prices):
    agent, chat, _ = make_agent(tmp_path, [tool_turn(call(WEB_SEARCH, query="AAPL")), AIMessage("ok")], [report_answer()], tool_names=AGENT_A_TOOLS)

    run = agent.research("AAPL")

    assert chat.calls[0]["tools"] == list(AGENT_A_TOOLS)
    assert run.observations[0]["status"] == "refused"
    assert run.tool_calls == 0
    assert trace_lines(tmp_path)[1]["status"] == "refused"  # line 0 is the agent_turn


def test_bad_arguments_go_back_to_the_model_as_an_error(tmp_path, fake_prices):
    agent, _, _ = make_agent(tmp_path, [tool_turn((GET_PRICE_DATA, {"why": "w"})), AIMessage("ok")], [report_answer()])

    run = agent.research("AAPL")

    assert run.observations[0]["status"] == "invalid_args"
    assert "ticker" in run.observations[0]["digest"]


def test_every_model_failing_still_ends_with_a_report(tmp_path, fake_prices):
    agent, _, _ = make_agent(tmp_path, [RuntimeError("Groq 503 and OpenRouter 429")], [None])

    run = agent.research("AAPL")

    assert run.report.generated_by == "template"
    assert any("agent model unavailable" in w for w in run.warnings)


# --- short-term memory ---------------------------------------------------------------------


def test_follow_up_is_answered_from_memory_without_new_tool_calls(tmp_path, fake_prices):
    agent, chat, llm = make_agent(
        tmp_path,
        [tool_turn(call(GET_PRICE_DATA, ticker="AAPL"), call(CALCULATE_VOLATILITY, ticker="AAPL", window=30)), AIMessage("Done."), AIMessage("It was 24.7%.")],
        [report_answer()],
    )
    first = agent.research("AAPL")
    tool_lines = len([line for line in trace_lines(tmp_path) if line["event"] == "tool_call"])

    follow_up = agent.ask("What volatility did you find?", thread_id=first.thread_id)

    assert follow_up.answer == "It was 24.7%."
    assert follow_up.tool_calls == 0
    assert len([line for line in trace_lines(tmp_path) if line["event"] == "tool_call"]) == tool_lines
    assert any(isinstance(m, ToolMessage) and "current_pct" in m.content for m in chat.calls[-1]["messages"])
    assert len(llm.requests) == 1  # the report was not rewritten
    assert follow_up.report == first.report


def test_follow_up_may_still_fetch_new_data(tmp_path, fake_prices):
    agent, _, _ = make_agent(
        tmp_path,
        [AIMessage("Done."), tool_turn(call(GET_PRICE_DATA, ticker="AAPL", period="1y")), AIMessage("Up 20% in a year.")],
        [report_answer()],
    )
    first = agent.research("AAPL")

    follow_up = agent.ask("How did it do over a year?", thread_id=first.thread_id)

    assert follow_up.tool_calls == 1 and follow_up.answer == "Up 20% in a year."


# --- printer and model setup ------------------------------------------------------------------


def test_printer_shows_why_hint_and_report(tmp_path, fake_prices, fake_search):
    lines: list[str] = []
    agent, _, _ = make_agent(tmp_path, [tool_turn(call(GET_PRICE_DATA, ticker="AAPL"), call(GET_NEWS, why="start with news", ticker="AAPL")), AIMessage("ok")], [report_answer()], failing_tools=[GET_NEWS])

    agent.research("AAPL", on_update=lambda name, step: print_update(name, step, write=lines.append))

    text = "\n".join(lines)
    assert "-> get_news(ticker='AAPL', n=10)" not in text  # the printer shows what the model sent
    assert "-> get_news(ticker='AAPL')" in text and "why: start with news" in text
    assert "x get_news error: injected failure" in text and "hint: Alternatives: web_search" in text
    assert "report written by llm: 3 risks, hedge collar" in text


def test_chat_models_need_the_primary_key_and_skip_a_missing_fallback(monkeypatch):
    profile = PROFILES["free"]
    monkeypatch.delenv(profile.primary.api_key_env, raising=False)
    with pytest.raises(RuntimeError, match=profile.primary.api_key_env):
        chat_models(profile)

    monkeypatch.setenv(profile.primary.api_key_env, "test-key")
    monkeypatch.delenv(profile.fallback.api_key_env, raising=False)
    assert len(chat_models(profile)) == 1

    monkeypatch.setenv(profile.fallback.api_key_env, "test-key")
    groq, openrouter = chat_models(profile)
    assert groq.reasoning_effort == agent_module.AGENT_EFFORT
    assert openrouter.extra_body["reasoning"] == {"effort": agent_module.AGENT_EFFORT}
    assert openrouter.extra_body["provider"] == {"require_parameters": True}


def test_every_tool_can_be_bound(tmp_path):
    session = make_session(tmp_path)

    assert [tool.name for tool in session.langchain_tools(TOOL_NAMES, agent="single")] == list(TOOL_NAMES)


# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Add a stronger paid OpenAI model (gpt-6.1-sol) for testing, in a separate file so it can be deleted before submission', Date: 2026-10-07
def test_printer_shows_text_sent_as_content_blocks():
    # OpenAI's Responses API sends a reply as a list of content blocks, not a string.
    reply = AIMessage(content=[{"type": "reasoning", "summary": []}, {"type": "text", "text": "Volatility was 24%."}])
    lines: list[str] = []

    print_update("single", {"agent": {"messages": [reply], "turns": 1}}, write=lines.append)

    assert any("says: Volatility was 24%." in line for line in lines)


# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Start PR 4: the Task 3 notebook, laid out as settled in the grilling rounds', Date: 2026-10-07
def test_replan_cycles_pair_a_turns_results_with_the_next_turns_choice(tmp_path, fake_prices, fake_search):
    fake_search.replies["text"] = [{"title": "Analyst cuts target", "href": "https://x", "body": "b"}]
    agent, _, _ = make_agent(
        tmp_path,
        [
            tool_turn(call(GET_NEWS, ticker="AAPL")),
            tool_turn(call(WEB_SEARCH, why="get_news failed; its hint says web_search", query="AAPL news")),
            AIMessage("Enough evidence."),
        ],
        [report_answer()],
        failing_tools=[GET_NEWS],
    )
    agent.research("AAPL")
    lines: list[str] = []

    events = read_trace(tmp_path / TRACE_FILE_NAME, run_id=agent.session.run_id)
    [cycle] = replan_cycles(events, "single")
    print_replan_cycles(events, "single", write=lines.append)

    assert cycle["turn"] == 2
    assert [(c["tool"], c["status"]) for c in cycle["observed"]] == [(GET_NEWS, "error")]
    assert [c["tool"] for c in cycle["chose"]] == [WEB_SEARCH]
    assert any("why: get_news failed; its hint says web_search" in line for line in lines)


def test_read_trace_of_a_missing_file_is_empty(tmp_path):
    assert read_trace(tmp_path / "none.jsonl") == []


# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'yes fix the hedge (levels from the volatility window closest to the 63-day horizon)', Date: 2026-10-07
def test_hedge_levels_prefer_the_horizon_matched_volatility_over_the_latest(tmp_path, fake_prices):
    session = make_session(tmp_path)
    obs = observations_from(
        session,
        (GET_PRICE_DATA, {"ticker": "AAPL"}),
        (CALCULATE_VOLATILITY, {"ticker": "AAPL", "window": 63}),
        (CALCULATE_VOLATILITY, {"ticker": "AAPL", "window": 30}),  # latest, but further from the 63-day horizon
    )

    assert compute_hedge_levels(obs).volatility_source == "calculate_volatility, 63-day"
