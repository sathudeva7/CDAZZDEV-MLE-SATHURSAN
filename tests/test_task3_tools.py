"""Offline tests for task3_agentic/tools.py, with yfinance, the news feeds, DuckDuckGo and the LLM replaced by fakes."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 1: the five agent tools, their tests and the new-tool skill, as designed in the grilling rounds', Date: 2026-10-07
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 2: the 3A agent loop, report, hedge levels, printer and short-term memory, as designed in the grilling rounds', Date: 2026-10-07

from __future__ import annotations

import json

import pandas as pd
import pytest
from ddgs.exceptions import RatelimitException
from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from task1_financial import news
from task1_financial.indicators import (
    COL_VOLATILITY,
    add_indicators,
    annualised_volatility,
)
from task3_agentic import tools
from task3_agentic.schemas import HeadlineLabel, HeadlineLabels, ToolResult
from task3_agentic.tools import (
    CALCULATE_VOLATILITY,
    GET_NEWS,
    GET_PRICE_DATA,
    LLM_SENTIMENT,
    MAX_NEWS_COUNT,
    MAX_VOLATILITY_WINDOW,
    NO_ALTERNATIVE,
    TOOL_NAMES,
    VOLATILITY_HISTORY_DAYS,
    WEB_SEARCH,
    percentile_rank,
)
from task3_agentic.trace import TRACE_OUTPUT_CHARS
from tests.fakes import (
    TODAY,
    ScriptedLLM,
    make_session,
    trace_lines,
    two_years_of_bars,
)

AGENT_A_TOOLS = (GET_PRICE_DATA, CALCULATE_VOLATILITY, LLM_SENTIMENT)


def labels(*entries) -> HeadlineLabels:
    return HeadlineLabels(labels=[HeadlineLabel(index=i, sentiment=s, confidence=c, reason="r") for i, s, c in entries])


# --- get_price_data ------------------------------------------------------------


def test_price_data_is_trimmed_to_the_period_with_task1_indicators(tmp_path, fake_prices):
    result = make_session(tmp_path).get_price_data("aapl", "6mo")

    assert result.status == "ok"
    price = result.data
    full = add_indicators(two_years_of_bars())
    expected = full.loc[full.index >= pd.Timestamp(TODAY) - pd.DateOffset(months=6)]
    assert price.ticker == "AAPL"
    assert len(price.bars) == len(expected)
    assert price.bars[0].date == expected.index[0].date()
    assert price.latest.indicators[COL_VOLATILITY] == pytest.approx(full[COL_VOLATILITY].iloc[-1])
    assert price.period_return_pct == pytest.approx((expected["Close"].iloc[-1] / expected["Close"].iloc[0] - 1) * 100)
    assert price.period_high == pytest.approx(expected["High"].max())
    assert price.close_vs_sma_200_pct == pytest.approx((full["Close"].iloc[-1] / full["sma_200"].iloc[-1] - 1) * 100)


def test_price_digest_shows_latest_values_and_five_bars(tmp_path, fake_prices):
    digest = make_session(tmp_path).get_price_data("AAPL", "1y").data.digest()

    assert len(digest["last_bars"]) == 5
    assert digest["last_bars"][-1][0] == TODAY
    assert set(digest["indicators"]) >= {"rsi_14", "sma_50", "sma_200", "macd", "bb_pct_b", "hv_30"}


def test_unknown_period_falls_back_to_the_default_with_a_warning(tmp_path, fake_prices):
    result = make_session(tmp_path).get_price_data("AAPL", "5y")

    assert result.status == "ok"
    assert result.data.period == tools.DEFAULT_PERIOD
    assert "5y" in result.warnings[0]


def test_price_and_volatility_share_one_download(tmp_path, fake_prices):
    session = make_session(tmp_path)

    session.get_price_data("AAPL")
    session.calculate_volatility("AAPL")
    session.get_price_data("AAPL", "1y")

    assert fake_prices.calls == ["AAPL"]
    assert [line["cache_hit"] for line in trace_lines(tmp_path)] == [False, True, True]


def test_no_prices_is_empty_and_hints_at_alternatives(tmp_path, fake_prices):
    fake_prices.frames[0] = pd.DataFrame()
    session = make_session(tmp_path)

    result = session.get_price_data("NOTATICKERZZ")

    assert result.status == "empty" and result.data is None
    assert "price history" in result.warnings[0]
    assert "web_search" in result.hint and "NOTATICKERZZ" in result.hint


def test_hint_names_only_tools_the_agent_has(tmp_path, fake_prices):
    fake_prices.frames[0] = pd.DataFrame()
    session = make_session(tmp_path)

    price = session.call(GET_PRICE_DATA, {"ticker": "ZZZ"}, available=AGENT_A_TOOLS)
    vol = session.call(CALCULATE_VOLATILITY, {"ticker": "ZZZ"}, available=AGENT_A_TOOLS)

    assert "web_search" not in price.hint and NO_ALTERNATIVE in price.hint
    assert "get_price_data" in vol.hint and "web_search" not in vol.hint


def test_an_empty_download_is_not_cached(tmp_path, fake_prices):
    fake_prices.frames[:] = [pd.DataFrame(), two_years_of_bars()]
    session = make_session(tmp_path)
    session.get_price_data("AAPL")  # three empty attempts

    assert session.get_price_data("AAPL").status == "ok"


# --- calculate_volatility ------------------------------------------------------


def test_percentile_rank_hand_checked():
    history = pd.Series([10.0, 20.0, 30.0, 40.0])

    assert percentile_rank(history, 30.0) == 75.0
    assert percentile_rank(history, 5.0) == 0.0
    assert percentile_rank(history, 40.0) == 100.0


def test_volatility_matches_task1_and_ranks_today_in_the_past_year(tmp_path, fake_prices):
    result = make_session(tmp_path).calculate_volatility("AAPL", 30)

    series = annualised_volatility(two_years_of_bars()["Close"], 30).dropna()
    history = series.iloc[-VOLATILITY_HISTORY_DAYS:]
    stats = result.data
    assert result.status == "ok"
    assert stats.current_pct == pytest.approx(series.iloc[-1] * 100)
    assert stats.max_1y_pct == pytest.approx(history.max() * 100)
    assert stats.percentile_1y == pytest.approx((history <= series.iloc[-1]).mean() * 100)
    assert stats.observations == VOLATILITY_HISTORY_DAYS


def test_volatility_window_is_clamped(tmp_path, fake_prices):
    result = make_session(tmp_path).calculate_volatility("AAPL", 1000)

    assert result.data.window == MAX_VOLATILITY_WINDOW
    assert "1000" in result.warnings[0]


# --- get_news --------------------------------------------------------------------


def rss_with(count: int) -> bytes:
    items = "".join(f"<item><title>Story {i}</title><pubDate>Mon, 05 Oct 2026 {i % 24:02d}:00:00 GMT</pubDate></item>" for i in range(count))
    return f"<rss><channel>{items}</channel></rss>".encode()


def test_news_returns_headlines_and_clamps_n(tmp_path, monkeypatch, no_sleep):
    monkeypatch.setattr(news, "_download", lambda url: rss_with(30))

    result = make_session(tmp_path).get_news("AAPL", 50)

    assert result.status == "ok"
    assert len(result.data.headlines) == MAX_NEWS_COUNT
    assert "50" in result.warnings[0]
    assert result.data.digest()["headlines"][0]["title"].startswith("Story")


def test_no_news_is_empty_and_suggests_web_search(tmp_path, monkeypatch, no_sleep):
    def offline(url):
        raise OSError("network down")

    monkeypatch.setattr(news, "_download", offline)

    result = make_session(tmp_path).get_news("AAPL")

    assert result.status == "empty"
    assert "web_search" in result.hint


# --- llm_sentiment -----------------------------------------------------------------


def test_sentiment_uses_one_call_and_task1s_score_rule(tmp_path):
    llm = ScriptedLLM([labels((1, "positive", 0.8), (2, "negative", 0.6), (3, "neutral", 0.9))])

    result = make_session(tmp_path, llm).llm_sentiment(["Beat", "Probe", "Event"])

    # Scores are sign x confidence: +0.8, -0.6, 0; the mean 0.0667 is inside +-0.2, so neutral.
    stats = result.data
    assert result.status == "ok"
    assert len(llm.requests) == 1
    assert llm.requests[0]["variables"] == {"subject": "AAPL", "headlines": "1. Beat\n2. Probe\n3. Event"}
    assert stats.summary.score == pytest.approx(0.0667)
    assert stats.summary.label == "neutral"
    assert [h["headline"] for h in stats.digest()["strongest"]] == ["Beat", "Probe", "Event"]


def test_an_unlabelled_headline_is_left_out_of_the_score(tmp_path):
    llm = ScriptedLLM([labels((1, "negative", 0.5))])

    result = make_session(tmp_path, llm).llm_sentiment(["Probe", "Ignored"])

    assert result.status == "ok"
    assert result.data.summary.scored == 1 and result.data.summary.failed == 1
    assert result.data.summary.score == pytest.approx(-0.5)
    assert any("no label for 1 of 2" in w for w in result.warnings)


def test_sentiment_llm_failure_is_an_error_with_a_hint(tmp_path):
    result = make_session(tmp_path, ScriptedLLM([None])).llm_sentiment(["Probe"])

    assert result.status == "error"
    assert "no LLM provider answered" in result.error
    assert "web_search" in result.hint


def test_no_headlines_skips_the_llm(tmp_path):
    llm = ScriptedLLM([])

    result = make_session(tmp_path, llm).llm_sentiment(["", "  "])

    assert result.status == "empty"
    assert llm.requests == []


def test_a_bare_headline_string_is_one_headline(tmp_path):
    llm = ScriptedLLM([labels((1, "positive", 0.7))])

    result = make_session(tmp_path, llm).llm_sentiment("Apple beats estimates")

    assert result.data.summary.scored == 1


# --- web_search --------------------------------------------------------------------


def test_web_search_maps_text_results(tmp_path, fake_search, no_sleep):
    rows = [{"title": f"T{i}", "href": f"https://x/{i}", "body": "snippet"} for i in range(8)]
    fake_search.replies["text"] = rows

    result = make_session(tmp_path).web_search("AAPL analyst outlook")

    assert result.status == "ok"
    assert result.data.backend == "text"
    assert len(result.data.hits) == tools.SEARCH_RESULTS
    assert result.data.hits[0].url == "https://x/0"


def test_rate_limited_web_search_falls_back_to_news_search(tmp_path, fake_search, no_sleep):
    fake_search.replies["text"] = RatelimitException("202 Ratelimit")
    fake_search.replies["news"] = [{"title": "N", "url": "https://n", "body": "b", "date": "2026-10-06T14:25:00+00:00"}]

    result = make_session(tmp_path).web_search("AAPL risks")

    assert result.status == "ok"
    assert result.data.backend == "news"
    assert result.data.hits[0].published.startswith("2026-10-06")
    assert any("RatelimitException" in w for w in result.warnings)
    assert len(no_sleep) == 2  # the text search was retried with backoff before switching


def test_failed_search_is_empty_and_suggests_get_news(tmp_path, fake_search, no_sleep):
    fake_search.replies["text"] = RatelimitException("202 Ratelimit")

    result = make_session(tmp_path).web_search("AAPL risks")

    assert result.status == "empty"
    assert "get_news" in result.hint


# --- the shared call path: injection, errors, trace ------------------------------------


def test_injected_failure_returns_an_error_without_running(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(tools, "fetch_headlines", lambda *a: called.append(a))

    result = make_session(tmp_path, failing_tools=[GET_NEWS]).get_news("AAPL")

    assert result.status == "error"
    assert "injected" in result.error
    assert called == []
    assert trace_lines(tmp_path)[0]["status"] == "error"


def test_unknown_failing_tool_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="get_quotes"):
        make_session(tmp_path, failing_tools=["get_quotes"])


def test_an_unexpected_exception_becomes_an_error_result(tmp_path, monkeypatch):
    def broken(*args):
        raise KeyError("pubDate")

    monkeypatch.setattr(tools, "fetch_headlines", broken)

    result = make_session(tmp_path).get_news("AAPL")

    assert result.status == "error"
    assert "KeyError" in result.error


def test_an_empty_ticker_is_an_error_not_an_exception(tmp_path):
    assert make_session(tmp_path).get_price_data("  ").status == "error"


def test_every_call_writes_one_trace_line(tmp_path, fake_prices):
    session = make_session(tmp_path)

    session.get_price_data("AAPL", agent="single")
    session.call(WEB_SEARCH, {"query": ""}, agent="B")

    first, second = trace_lines(tmp_path)
    assert first["event"] == "tool_call" and first["run_id"] == session.run_id
    assert first["agent"] == "single" and first["tool"] == GET_PRICE_DATA
    assert first["args"] == {"ticker": "AAPL", "period": "6mo"}
    assert len(first["output"]) == TRACE_OUTPUT_CHARS < first["output_chars"]
    assert first["duration_ms"] >= 0
    assert second["status"] == "error" and second["agent"] == "B"


# --- LangChain tools ---------------------------------------------------------------------


def test_langchain_tools_are_restricted_and_carry_the_typed_result(tmp_path, fake_prices):
    session = make_session(tmp_path)
    agent_tools = session.langchain_tools(AGENT_A_TOOLS, agent="A")

    assert [tool.name for tool in agent_tools] == list(AGENT_A_TOOLS)
    call = {"type": "tool_call", "id": "call_1", "name": GET_PRICE_DATA, "args": {"why": "Need the trend first", "ticker": "AAPL"}}
    message = agent_tools[0].invoke(call)

    assert isinstance(message, ToolMessage)
    assert json.loads(message.content)["status"] == "ok"
    assert isinstance(message.artifact, ToolResult) and message.artifact.data.ticker == "AAPL"
    line = trace_lines(tmp_path)[0]
    assert line["agent"] == "A" and line["args"] == {"ticker": "AAPL", "period": "6mo"}
    assert line["why"] == "Need the trend first"


def test_agent_arguments_put_why_first_and_require_it():
    schema = tools.AGENT_TOOL_ARGS[GET_PRICE_DATA].model_json_schema()

    assert next(iter(schema["properties"])) == tools.WHY_ARG
    assert tools.WHY_ARG in schema["required"]
    assert tools.split_agent_args(GET_NEWS, {"why": "w", "ticker": "AAPL"}) == ({"ticker": "AAPL", "n": tools.DEFAULT_NEWS_COUNT}, "w")
    with pytest.raises(ValidationError):
        tools.split_agent_args(GET_NEWS, {"ticker": "AAPL"})


def test_langchain_tools_reject_unknown_names(tmp_path):
    with pytest.raises(ValueError, match="get_quotes"):
        make_session(tmp_path).langchain_tools(["get_quotes"], agent="A")


def test_every_tool_has_a_description_and_argument_schema():
    for name in TOOL_NAMES:
        assert tools.TOOL_DESCRIPTIONS[name] and name in tools.TOOL_ARGS and name in tools.ALTERNATIVES
