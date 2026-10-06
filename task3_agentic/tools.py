"""The five tools Task 3's agents can call, built on Task 1's data, indicator, news and sentiment code.

    get_price_data(ticker, period)        adjusted daily OHLCV with every Task 1 indicator
    calculate_volatility(ticker, window)  annualised volatility now and within the past year
    get_news(ticker, n)                   recent headlines from the Task 1 RSS feeds
    llm_sentiment(headlines)              one batched LLM call and Task 1's Sentiment score
    web_search(query)                     DuckDuckGo results through `ddgs` (the renamed duckduckgo-search)

How every call runs (ToolSession.call):

- Never raises. A tool returns a ToolResult whose status is ok, empty or
  error. A failed result carries a `hint` naming the other tools that could
  fill the gap. The hint lists only tools the calling agent has, so Agent A
  is never told to use web_search, which it doesn't have.
- Traced. Every call, whatever its outcome, writes one tool_call line to
  agent_trace.jsonl, with the agent's name and the effective arguments.
- Shared prices. get_price_data and calculate_volatility read one cached
  download per (ticker, today), so a run asks Yahoo once per ticker. Only a
  non-empty download is cached, so a transient failure is retried by the next call.
- Demo failures. A tool listed in `failing_tools` returns status error
  without running, so the notebook can show the agent switching tools. The
  switch is off by default, and its error message says it was injected.

Prices are always fetched for Task 1's two-year window, so SMA-200 has data
before the first bar shown, then trimmed to `period`. Out-of-range arguments
are clamped or replaced with the default and reported as a warning, never
rejected, so the agent still gets an answer.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 1: the five agent tools, their tests and the new-tool skill, as designed in the grilling rounds', Date: 2026-10-07

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from ddgs import DDGS
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from common.llm import StructuredLLM
from task1_financial.data import fetch_ohlcv, with_retries
from task1_financial.indicators import (
    COL_BB_LOWER,
    COL_BB_MIDDLE,
    COL_BB_PCT_B,
    COL_BB_UPPER,
    COL_MACD,
    COL_MACD_HIST,
    COL_MACD_SIGNAL,
    COL_RSI,
    COL_SMA_LONG,
    COL_SMA_SHORT,
    COL_VOLATILITY,
    TRADING_DAYS_PER_YEAR,
    VOLATILITY_WINDOW,
    add_indicators,
    annualised_volatility,
)
from task1_financial.news import fetch_headlines
from task1_financial.pipeline import MARKET_TIMEZONE
from task1_financial.schemas import HeadlineSentiment
from task1_financial.sentiment import SCORE_DECIMALS, SIGN, summarise_sentiment
from task1_financial.summary import FIFTY_TWO_WEEKS, HIGH_COLUMN, LOW_COLUMN
from task3_agentic.prompts import BATCH_SENTIMENT
from task3_agentic.schemas import (
    HeadlineLabels,
    NewsList,
    PriceBar,
    PriceData,
    SearchBackend,
    SearchHit,
    SearchResults,
    SentimentStats,
    ToolResult,
    VolatilityStats,
)
from task3_agentic.trace import DEFAULT_LOG_DIR, TraceLog

logger = logging.getLogger(__name__)

GET_PRICE_DATA = "get_price_data"
CALCULATE_VOLATILITY = "calculate_volatility"
GET_NEWS = "get_news"
LLM_SENTIMENT = "llm_sentiment"
WEB_SEARCH = "web_search"
TOOL_NAMES = (GET_PRICE_DATA, CALCULATE_VOLATILITY, GET_NEWS, LLM_SENTIMENT, WEB_SEARCH)

# The agent name recorded for calls made by code rather than by an agent.
DIRECT_AGENT = "direct"

DEFAULT_PERIOD = "6mo"
PERIOD_OFFSETS = {
    "1mo": pd.DateOffset(months=1),
    "3mo": pd.DateOffset(months=3),
    "6mo": pd.DateOffset(months=6),
    "1y": pd.DateOffset(years=1),
    "2y": pd.DateOffset(years=2),
}
PRICE_COLUMN = "Close"
INDICATOR_COLUMNS = (
    COL_SMA_SHORT,
    COL_SMA_LONG,
    COL_RSI,
    COL_MACD,
    COL_MACD_SIGNAL,
    COL_MACD_HIST,
    COL_BB_UPPER,
    COL_BB_MIDDLE,
    COL_BB_LOWER,
    COL_BB_PCT_B,
    COL_VOLATILITY,
)
PERCENT = 100.0

DEFAULT_VOLATILITY_WINDOW = VOLATILITY_WINDOW
MIN_VOLATILITY_WINDOW = 5  # fewer returns than this make a noisy estimate
MAX_VOLATILITY_WINDOW = TRADING_DAYS_PER_YEAR
VOLATILITY_HISTORY_DAYS = TRADING_DAYS_PER_YEAR  # today's value is ranked against this many past values

DEFAULT_NEWS_COUNT = 10
MAX_NEWS_COUNT = 20  # every headline goes back to the model, so the count is capped for the token budget

MAX_SENTIMENT_HEADLINES = 20
SENTIMENT_EFFORT = "low"  # labelling is classification; reasoning tokens count against the free tier's limit
DEFAULT_SUBJECT = "the company each headline is mainly about"
NOT_LABELLED_REASON = "Not scored: the LLM returned no label for this headline."

SEARCH_RESULTS = 5
SEARCH_TIMEOUT_SECONDS = 10
SEARCH_BACKENDS: tuple[SearchBackend, ...] = ("text", "news")  # the web search first, news search if it fails

INJECTED_FAILURE = "injected failure: {tool} is in failing_tools (a demo switch that is off by default)"
NO_ALTERNATIVE = "No other tool you have can supply this; record it as a data gap and continue with what you have."

# What to try when a tool fails: advice about the tool itself, then alternative tools.
# {ticker} is filled from the call's arguments.
TOOL_ADVICE = {
    GET_PRICE_DATA: "Check that {ticker} is a valid Yahoo Finance symbol; calculate_volatility reads the same prices.",
    CALCULATE_VOLATILITY: "A shorter window needs less history.",
    GET_NEWS: "",
    LLM_SENTIMENT: "",
    WEB_SEARCH: "Try a shorter or differently worded query.",
}
ALTERNATIVES: dict[str, list[tuple[str, str]]] = {
    GET_PRICE_DATA: [(WEB_SEARCH, "search '{ticker} stock price performance' to describe recent moves in words")],
    CALCULATE_VOLATILITY: [
        (GET_PRICE_DATA, "its indicators include hv_30, the 30-day annualised volatility"),
        (WEB_SEARCH, "search '{ticker} implied volatility'"),
    ],
    GET_NEWS: [(WEB_SEARCH, "search '{ticker} stock news this week' for recent headlines")],
    LLM_SENTIMENT: [(WEB_SEARCH, "search for analyst commentary on {ticker} and judge its tone from the snippets")],
    WEB_SEARCH: [(GET_NEWS, "use recent headlines for {ticker} as commentary instead")],
}

# What the agent's model is told each tool does; LangChain sends these with the argument schemas.
TOOL_DESCRIPTIONS = {
    GET_PRICE_DATA: (
        "Daily adjusted prices for a ticker with technical indicators (SMA-50, SMA-200, RSI-14, MACD, "
        "Bollinger Bands and %B, 30-day volatility). Returns the latest indicator values, the period "
        "return, period and 52-week high and low, distance from the moving averages, and the last 5 bars."
    ),
    CALCULATE_VOLATILITY: (
        "Annualised historical volatility of daily log returns over `window` trading days, and where "
        "today's value sits within the past year (min, median, max, percentile). A high percentile "
        "means the stock is moving more than usual, and options cost more."
    ),
    GET_NEWS: "Recent news headlines (last 7 days) for a ticker, newest first, from Yahoo Finance and Google News.",
    LLM_SENTIMENT: (
        "Labels each headline positive, negative or neutral for the share price using an LLM, and returns "
        "the Sentiment score from -1 (uniformly negative) to +1 (uniformly positive) with the strongest headlines."
    ),
    WEB_SEARCH: (
        "Searches the web (DuckDuckGo) and returns the top 5 results with snippets. Use it for analyst "
        "commentary, price targets, upcoming events and risks that headlines don't cover."
    ),
}


# --- argument schemas the agent's model sees ---------------------------------
# Plain str and int rather than Literal or bounded fields: an out-of-range value
# then reaches the tool, which replaces or clamps it and says so, instead of
# LangChain rejecting the call before it is traced.


class PriceDataArgs(BaseModel):
    ticker: str = Field(description="Yahoo Finance ticker symbol, such as AAPL")
    period: str = Field(default=DEFAULT_PERIOD, description="How much history to return: 1mo, 3mo, 6mo, 1y or 2y")


class VolatilityArgs(BaseModel):
    ticker: str = Field(description="Yahoo Finance ticker symbol, such as AAPL")
    window: int = Field(
        default=DEFAULT_VOLATILITY_WINDOW,
        description=f"Trading days of returns per estimate, {MIN_VOLATILITY_WINDOW} to {MAX_VOLATILITY_WINDOW}",
    )


class NewsArgs(BaseModel):
    ticker: str = Field(description="Yahoo Finance ticker symbol, such as AAPL")
    n: int = Field(default=DEFAULT_NEWS_COUNT, description=f"How many headlines to return, 1 to {MAX_NEWS_COUNT}")


class SentimentArgs(BaseModel):
    headlines: list[str] = Field(description=f"Headline titles to label, at most {MAX_SENTIMENT_HEADLINES}")


class SearchArgs(BaseModel):
    query: str = Field(description="What to search for, such as 'AAPL analyst price target'")


TOOL_ARGS: dict[str, type[BaseModel]] = {
    GET_PRICE_DATA: PriceDataArgs,
    CALCULATE_VOLATILITY: VolatilityArgs,
    GET_NEWS: NewsArgs,
    LLM_SENTIMENT: SentimentArgs,
    WEB_SEARCH: SearchArgs,
}

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 2: the 3A agent loop, report, hedge levels, printer and short-term memory, as designed in the grilling rounds', Date: 2026-10-07
# Agents get one extra, required argument: `why`. Live tests showed gpt-oss (Groq) and
# the OpenRouter fallback both leave the message text empty when they call a tool,
# even when told to write a note, but both fill a required argument. `why` is
# traced and printed (the visible observe -> replan step), then dropped before the
# tool runs, so the Python methods keep the brief's exact signatures.
WHY_ARG = "why"
WHY_DESCRIPTION = "One sentence: what you have observed so far, and why this call is the right next step"


def _with_why(args: type[BaseModel]) -> type[BaseModel]:
    """`args` with `why` added as its first field, so the model states its reason before the parameters."""
    fields = {name: (field.annotation, field) for name, field in args.model_fields.items()}
    return create_model(args.__name__.replace("Args", "AgentArgs"), **{WHY_ARG: (str, Field(description=WHY_DESCRIPTION))}, **fields)


AGENT_TOOL_ARGS: dict[str, type[BaseModel]] = {name: _with_why(schema) for name, schema in TOOL_ARGS.items()}


def split_agent_args(tool: str, raw: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """An agent's raw arguments as (tool arguments with defaults filled in, the `why` note).

    Raises pydantic.ValidationError when a required argument is missing or the wrong type.
    """
    args = AGENT_TOOL_ARGS[tool](**raw).model_dump()
    return args, args.pop(WHY_ARG)


class ToolSession:
    """The five tools for one research run: shared price cache, one trace, one run id.

    Create one per run. Call tools directly (`session.get_price_data("AAPL")`)
    or hand an agent its subset with `langchain_tools(names, agent=...)`.
    `llm` and `search_factory` are replaceable so tests run offline.
    """

    def __init__(
        self,
        *,
        today: date | None = None,
        subject: str | None = None,
        run_id: str | None = None,
        log_dir: Path | str = DEFAULT_LOG_DIR,
        failing_tools: Sequence[str] = (),
        llm: StructuredLLM | None = None,
        search_factory: Callable[..., Any] = DDGS,
    ) -> None:
        unknown = set(failing_tools) - set(TOOL_NAMES)
        if unknown:
            raise ValueError(f"unknown tools in failing_tools: {sorted(unknown)}; tools are {TOOL_NAMES}")
        self.today = today or datetime.now(MARKET_TIMEZONE).date()
        self.subject = subject  # the ticker or company being researched; tells llm_sentiment whose share price matters
        self.run_id = run_id or uuid4().hex[:12]
        self.log_dir = Path(log_dir)
        self.trace = TraceLog(self.run_id, self.log_dir)
        self.failing_tools = frozenset(failing_tools)
        self._llm = llm
        self._search_factory = search_factory
        self._prices: dict[tuple[str, date], pd.DataFrame] = {}
        self._prices_lock = threading.Lock()  # parallel tool calls run in threads
        self._tools: dict[str, Callable[..., tuple[ToolResult, bool]]] = {
            GET_PRICE_DATA: self._get_price_data,
            CALCULATE_VOLATILITY: self._calculate_volatility,
            GET_NEWS: self._get_news,
            LLM_SENTIMENT: self._llm_sentiment,
            WEB_SEARCH: self._web_search,
        }

    # --- the five tools, called directly ---------------------------------------

    def get_price_data(self, ticker: str, period: str = DEFAULT_PERIOD, *, agent: str = DIRECT_AGENT) -> ToolResult[PriceData]:
        return self.call(GET_PRICE_DATA, {"ticker": ticker, "period": period}, agent=agent)

    def calculate_volatility(
        self, ticker: str, window: int = DEFAULT_VOLATILITY_WINDOW, *, agent: str = DIRECT_AGENT
    ) -> ToolResult[VolatilityStats]:
        return self.call(CALCULATE_VOLATILITY, {"ticker": ticker, "window": window}, agent=agent)

    def get_news(self, ticker: str, n: int = DEFAULT_NEWS_COUNT, *, agent: str = DIRECT_AGENT) -> ToolResult[NewsList]:
        return self.call(GET_NEWS, {"ticker": ticker, "n": n}, agent=agent)

    def llm_sentiment(self, headlines: list[str], *, agent: str = DIRECT_AGENT) -> ToolResult[SentimentStats]:
        return self.call(LLM_SENTIMENT, {"headlines": headlines}, agent=agent)

    def web_search(self, query: str, *, agent: str = DIRECT_AGENT) -> ToolResult[SearchResults]:
        return self.call(WEB_SEARCH, {"query": query}, agent=agent)

    # --- the one path every call takes -----------------------------------------

    def call(
        self,
        tool: str,
        args: dict[str, Any],
        *,
        agent: str = DIRECT_AGENT,
        available: Sequence[str] = TOOL_NAMES,
        why: str | None = None,
    ) -> ToolResult:
        """Run `tool` with `args`, trace it, and return its result. Never raises for a tool failure.

        `available` is the calling agent's tool list; a failure's hint names only those tools.
        `why` is the agent's reason for the call, recorded in the trace.
        An unknown tool name is a caller bug and raises ValueError.
        """
        if tool not in self._tools:
            raise ValueError(f"unknown tool {tool!r}; tools are {TOOL_NAMES}")
        started = time.perf_counter()
        cache_hit = False
        if tool in self.failing_tools:
            result = ToolResult(tool=tool, status="error", data=None, error=INJECTED_FAILURE.format(tool=tool))
        else:
            try:
                result, cache_hit = self._tools[tool](**args)
            # A tool must never stop the agent: an unforeseen bug or a wrong argument
            # from the model becomes an error result the agent can react to.
            except Exception as exc:
                logger.exception("%s failed unexpectedly", tool)
                result = ToolResult(tool=tool, status="error", data=None, error=f"{type(exc).__name__}: {exc}")

        if result.status != "ok":
            result.hint = self._hint(tool, args, available)
            logger.warning("%s returned %s: %s", tool, result.status, result.error or "; ".join(result.warnings))
        self.trace.tool_call(
            agent=agent,
            tool=tool,
            args=args,
            status=result.status,
            output=result.for_llm(),
            duration_ms=(time.perf_counter() - started) * 1000,
            cache_hit=cache_hit,
            why=why,
        )
        return result

    def langchain_tools(self, names: Sequence[str], *, agent: str) -> list[StructuredTool]:
        """LangChain tools for an agent that may use only `names`; their calls are traced under `agent`.

        Each tool returns (digest text, ToolResult): the model reads the text,
        and the typed result rides along as the ToolMessage's artifact. The
        argument schema includes the required `why` (see AGENT_TOOL_ARGS).
        """
        unknown = [name for name in names if name not in TOOL_NAMES]
        if unknown:
            raise ValueError(f"unknown tools {unknown}; tools are {TOOL_NAMES}")
        available = tuple(names)

        def make(name: str) -> StructuredTool:
            def run(**kwargs: Any) -> tuple[str, ToolResult]:
                args, why = split_agent_args(name, kwargs)  # fills in defaults, so the trace shows them
                result = self.call(name, args, agent=agent, available=available, why=why)
                return result.for_llm(), result

            return StructuredTool.from_function(
                func=run,
                name=name,
                description=TOOL_DESCRIPTIONS[name],
                args_schema=AGENT_TOOL_ARGS[name],
                response_format="content_and_artifact",
            )

        return [make(name) for name in names]

    # --- tool bodies: each returns (result, cache_hit) -------------------------

    def _get_price_data(self, ticker: str, period: str = DEFAULT_PERIOD) -> tuple[ToolResult, bool]:
        warnings: list[str] = []
        ticker = _clean_ticker(ticker)
        if period not in PERIOD_OFFSETS:
            _warn(warnings, f"period {period!r} is not one of {list(PERIOD_OFFSETS)}, so {DEFAULT_PERIOD} was used")
            period = DEFAULT_PERIOD
        prices, cache_hit = self._load_prices(ticker, warnings)
        if prices.empty:
            return ToolResult(tool=GET_PRICE_DATA, status="empty", data=None, warnings=warnings), cache_hit

        start = pd.Timestamp(self.today) - PERIOD_OFFSETS[period]
        in_period = prices.loc[prices.index >= start]
        year = prices.loc[prices.index >= pd.Timestamp(self.today) - FIFTY_TWO_WEEKS]
        bars = [_bar(timestamp, row) for timestamp, row in in_period.iterrows()]
        first, last = bars[0], bars[-1]
        data = PriceData(
            ticker=ticker,
            period=period,
            as_of=last.date,
            bars=bars,
            period_return_pct=(last.close / first.close - 1) * PERCENT if len(bars) > 1 else None,
            period_high=float(in_period[HIGH_COLUMN].max()),
            period_low=float(in_period[LOW_COLUMN].min()),
            high_52w=float(year[HIGH_COLUMN].max()),
            low_52w=float(year[LOW_COLUMN].min()),
            close_vs_sma_50_pct=_pct_from(last.close, last.indicators[COL_SMA_SHORT]),
            close_vs_sma_200_pct=_pct_from(last.close, last.indicators[COL_SMA_LONG]),
        )
        return ToolResult(tool=GET_PRICE_DATA, status="ok", data=data, warnings=warnings), cache_hit

    def _calculate_volatility(self, ticker: str, window: int = DEFAULT_VOLATILITY_WINDOW) -> tuple[ToolResult, bool]:
        warnings: list[str] = []
        ticker = _clean_ticker(ticker)
        clamped = min(max(int(window), MIN_VOLATILITY_WINDOW), MAX_VOLATILITY_WINDOW)
        if clamped != window:
            _warn(warnings, f"window={window} is outside {MIN_VOLATILITY_WINDOW}..{MAX_VOLATILITY_WINDOW}, using {clamped}")
        prices, cache_hit = self._load_prices(ticker, warnings)
        series = annualised_volatility(prices[PRICE_COLUMN], clamped).dropna() if not prices.empty else pd.Series(dtype=float)
        if series.empty:
            if not prices.empty:
                _warn(warnings, f"{ticker} has fewer than {clamped + 1} daily prices, too few for a {clamped}-day volatility")
            return ToolResult(tool=CALCULATE_VOLATILITY, status="empty", data=None, warnings=warnings), cache_hit

        history = series.iloc[-VOLATILITY_HISTORY_DAYS:]
        current = float(series.iloc[-1])
        data = VolatilityStats(
            ticker=ticker,
            window=clamped,
            as_of=series.index[-1].date(),
            current_pct=current * PERCENT,
            min_1y_pct=float(history.min()) * PERCENT,
            median_1y_pct=float(history.median()) * PERCENT,
            max_1y_pct=float(history.max()) * PERCENT,
            percentile_1y=percentile_rank(history, current),
            observations=len(history),
        )
        return ToolResult(tool=CALCULATE_VOLATILITY, status="ok", data=data, warnings=warnings), cache_hit

    def _get_news(self, ticker: str, n: int = DEFAULT_NEWS_COUNT) -> tuple[ToolResult, bool]:
        warnings: list[str] = []
        ticker = _clean_ticker(ticker)
        count = min(max(int(n), 1), MAX_NEWS_COUNT)
        if count != n:
            _warn(warnings, f"n={n} is outside 1..{MAX_NEWS_COUNT}, using {count}")
        news = fetch_headlines(ticker, self.today, count)
        warnings.extend(news.warnings)
        status = "ok" if news.headlines else "empty"
        return ToolResult(tool=GET_NEWS, status=status, data=NewsList(ticker=ticker, headlines=news.headlines), warnings=warnings), False

    def _llm_sentiment(self, headlines: list[str]) -> tuple[ToolResult, bool]:
        warnings: list[str] = []
        if isinstance(headlines, str):  # one headline passed bare, not in a list
            headlines = [headlines]
        titles = [title.strip() for title in headlines if isinstance(title, str) and title.strip()]
        if len(titles) > MAX_SENTIMENT_HEADLINES:
            _warn(warnings, f"{len(titles)} headlines given; only the first {MAX_SENTIMENT_HEADLINES} were labelled")
            titles = titles[:MAX_SENTIMENT_HEADLINES]
        if not titles:
            return ToolResult(tool=LLM_SENTIMENT, status="empty", data=None, error="no headlines were given"), False

        try:
            llm = self._get_llm()
        except RuntimeError as exc:  # StructuredLLM raises when the primary provider's key is missing
            return ToolResult(tool=LLM_SENTIMENT, status="error", data=None, error=str(exc)), False
        answer = llm.call(
            BATCH_SENTIMENT,
            {"subject": self.subject or DEFAULT_SUBJECT, "headlines": "\n".join(f"{i}. {t}" for i, t in enumerate(titles, 1))},
            HeadlineLabels,
            fallback=HeadlineLabels(labels=[]),
            reasoning_effort=SENTIMENT_EFFORT,
        )
        if not answer.ok:
            return ToolResult(tool=LLM_SENTIMENT, status="error", data=None, error=f"no LLM provider answered: {answer.error}"), False

        labels = {}
        for label in answer.value.labels:
            labels.setdefault(label.index, label)  # the first label for an index wins
        items = [_headline_sentiment(title, labels.get(i)) for i, title in enumerate(titles, 1)]
        missing = sum(item.score is None for item in items)
        if missing:
            _warn(warnings, f"the LLM returned no label for {missing} of {len(titles)} headlines; they are left out of the score")
        summary = summarise_sentiment(items, warnings)
        status = "ok" if summary.scored else "error"
        return ToolResult(tool=LLM_SENTIMENT, status=status, data=SentimentStats(summary=summary, items=items), warnings=warnings), False

    def _web_search(self, query: str) -> tuple[ToolResult, bool]:
        warnings: list[str] = []
        query = query.strip()
        if not query:
            return ToolResult(tool=WEB_SEARCH, status="error", data=None, error="the query is empty"), False
        for backend in SEARCH_BACKENDS:
            rows = with_retries(
                lambda backend=backend: self._search(backend, query),  # bound now, not at loop end
                what=f"{backend} search results for {query!r}",
                is_empty=lambda found: not found,
                warnings=warnings,
            )
            if rows:
                hits = [_search_hit(row) for row in rows[:SEARCH_RESULTS]]
                if backend != SEARCH_BACKENDS[0]:
                    _warn(warnings, "the web search failed, so these are news search results")
                data = SearchResults(query=query, backend=backend, hits=hits)
                return ToolResult(tool=WEB_SEARCH, status="ok", data=data, warnings=warnings), False
        return ToolResult(tool=WEB_SEARCH, status="empty", data=None, warnings=warnings), False

    # --- helpers -----------------------------------------------------------------

    def _load_prices(self, ticker: str, warnings: list[str]) -> tuple[pd.DataFrame, bool]:
        """Two years of bars with indicators, from the session cache when this run already fetched them."""
        key = (ticker, self.today)
        # Held during the download, so a parallel call for the same ticker waits and then reads the cache.
        with self._prices_lock:
            if key in self._prices:
                logger.info("price cache hit for %s", ticker)
                return self._prices[key], True
            history = fetch_ohlcv(ticker, self.today)
            warnings.extend(history.warnings)
            if history.prices.empty:
                return history.prices, False
            prices = add_indicators(history.prices)
            self._prices[key] = prices
            return prices, False

    def _get_llm(self) -> StructuredLLM:
        if self._llm is None:
            self._llm = StructuredLLM(log_dir=self.log_dir)
        return self._llm

    def _search(self, backend: SearchBackend, query: str) -> list[dict[str, Any]]:
        client = self._search_factory(timeout=SEARCH_TIMEOUT_SECONDS)
        return getattr(client, backend)(query, max_results=SEARCH_RESULTS)

    def _hint(self, tool: str, args: dict[str, Any], available: Sequence[str]) -> str:
        ticker = str(args.get("ticker") or self.subject or "the ticker").strip().upper()
        parts = [TOOL_ADVICE[tool].format(ticker=ticker)] if TOOL_ADVICE[tool] else []
        options = [f"{name}: {how.format(ticker=ticker)}" for name, how in ALTERNATIVES[tool] if name in available]
        parts.append(("Alternatives: " + "; ".join(options) + ".") if options else NO_ALTERNATIVE)
        return " ".join(parts)


def percentile_rank(history: pd.Series, current: float) -> float:
    """Percent of `history` at or below `current`: [10, 20, 30, 40] with current 30 gives 75.0."""
    return float((history <= current).mean() * PERCENT)


def _clean_ticker(ticker: str) -> str:
    cleaned = str(ticker).strip().upper()
    if not cleaned:
        raise ValueError("the ticker is empty")
    return cleaned


def _bar(timestamp: pd.Timestamp, row: pd.Series) -> PriceBar:
    return PriceBar(
        date=timestamp.date(),
        open=float(row["Open"]),
        high=float(row[HIGH_COLUMN]),
        low=float(row[LOW_COLUMN]),
        close=float(row[PRICE_COLUMN]),
        volume=float(row["Volume"]),
        indicators={name: _finite(row[name]) for name in INDICATOR_COLUMNS},
    )


def _finite(value: Any) -> float | None:
    """NaN (an indicator still warming up) becomes None, which JSON can carry."""
    number = float(value)
    return number if math.isfinite(number) else None


def _pct_from(price: float, reference: float | None) -> float | None:
    """How far `price` is above (+) or below (-) `reference`, in percent: 110 vs 100 gives 10.0."""
    if reference is None or reference == 0:
        return None
    return (price / reference - 1) * PERCENT


def _headline_sentiment(title: str, label) -> HeadlineSentiment:
    """A Task 1 HeadlineSentiment record; the score follows Task 1's LLM rule, sign x confidence."""
    if label is None:
        return HeadlineSentiment(
            headline=title, sentiment="neutral", confidence=0.0, brief_reason=NOT_LABELLED_REASON,
            probabilities=None, scored_by="none", score=None,
        )
    return HeadlineSentiment(
        headline=title,
        sentiment=label.sentiment,
        confidence=label.confidence,
        brief_reason=label.reason,
        probabilities=None,
        scored_by="llm",
        score=round(SIGN[label.sentiment] * label.confidence, SCORE_DECIMALS),
    )


def _search_hit(row: dict[str, Any]) -> SearchHit:
    """One ddgs row: text results have title/href/body, news results title/url/body/date."""
    return SearchHit(
        title=str(row.get("title") or ""),
        url=row.get("href") or row.get("url"),
        snippet=str(row.get("body") or ""),
        published=row.get("date"),
    )


def _warn(warnings: list[str], message: str) -> None:
    """Log a fallback and record it on the tool result, so the two always agree."""
    logger.warning(message)
    warnings.append(message)
