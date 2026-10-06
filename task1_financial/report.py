"""Bonus: a one-page equity research brief built from the Task 1A and 1B outputs.

The brief is written as Markdown from the template constants below, then
rendered to a styled HTML page with the price chart embedded as a base64 PNG,
so the HTML file opens anywhere without its neighbours. Sections, in order:
company snapshot, technical outlook (with the chart), news sentiment with the
top three headlines, the Recommendation, how the brief was made, data notes,
and the risk disclaimer.

No new model call is made here. The technical outlook reuses the same computed
facts the Recommendation prompt received (recommendation.technical_facts), so
the page cannot contradict the numbers or the LLM's own reasoning.

The top three headlines are the ones with the largest |score|, the strongest
sentiment either way, because they move the Sentiment score most. Ties go to
the newer headline. Picking the three that agree with the Recommendation
would be cherry-picking.

Like the rest of the pipeline this never raises for a data problem: a missing
value reads "unavailable", an empty news list or a placeholder Hold is said so
on the page, and a chart that cannot be drawn is left out with a data note.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 1 notebook and the bonus report as designed in the grilling rounds', Date: 2026-10-06

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from string import Template

import markdown
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch

from common.llm_config import PROFILES
from task1_financial.indicators import (
    BB_NUM_STD,
    BB_WINDOW,
    COL_BB_LOWER,
    COL_BB_UPPER,
    COL_MACD,
    COL_MACD_HIST,
    COL_MACD_SIGNAL,
    COL_RSI,
    COL_SMA_LONG,
    COL_SMA_SHORT,
    DEFAULT_PRICE_COLUMN,
    MACD_FAST_SPAN,
    MACD_SIGNAL_SPAN,
    MACD_SLOW_SPAN,
    RSI_PERIOD,
    SMA_LONG_WINDOW,
    SMA_SHORT_WINDOW,
)
from task1_financial.news import NEWS_MAX_AGE_DAYS, Headline
from task1_financial.pipeline import Analysis, MarketData
from task1_financial.recommendation import (
    RECOMMENDATION_HORIZON_DAYS,
    UNAVAILABLE,
    technical_facts,
)
from task1_financial.schemas import HeadlineSentiment
from task1_financial.signals import RSI_OVERBOUGHT, RSI_OVERSOLD

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent / "reports"
TOP_HEADLINES = 3

# --- chart -----------------------------------------------------------------

CHART_LOOKBACK_DAYS = 365  # calendar days shown: one year of context for the 200-day average
CHART_FIGSIZE = (10, 7.5)  # inches; fits the page width at CHART_DPI
CHART_DPI = 110
CHART_HEIGHT_RATIOS = (3, 1, 1)  # price panel, RSI panel, MACD panel
RSI_AXIS_LIMITS = (0, 100)
LINE_WIDTH = 1.5
THIN_LINE_WIDTH = 1.0
LEGEND_ALPHA = 0.9

# Colours from the data-viz reference palette (light mode): categorical slots
# in fixed order for the series, a gray for the band, and the diverging blue/red
# poles for the MACD histogram's sign. Text stays in ink colours, never series colours.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"  # blue: close, RSI, MACD line
SERIES_2 = "#eb6834"  # orange: SMA-50, MACD signal
SERIES_3 = "#1baf7a"  # aqua: SMA-200
BAND_FILL = "#e4e3df"
HIST_UP = "#86b6ef"
HIST_DOWN = "#f0a3a2"

# --- page --------------------------------------------------------------------

REPORT_TITLE = "{company} ({ticker}): equity research brief"
CHART_ALT = "{ticker} close with moving averages and Bollinger Bands, RSI and MACD"
NO_HEADLINES = "No headlines were retrieved, so there is no news sentiment for this brief."
NO_SCORED_HEADLINES = "Headlines were retrieved, but none could be scored."
NO_CHART = "_The chart could not be drawn; see Data notes._"
PLACEHOLDER_NOTE = (
    "> **Placeholder:** no language model answered, so this Hold is not an analysis. See Data notes."
)
RISK_DISCLAIMER = (
    "This brief was generated automatically from public market data and news headlines, "
    "with language models writing the reasoning. It is for information and assessment purposes only "
    "and is not investment advice, a solicitation, or a recommendation to buy or sell any security. "
    "Indicators describe past prices and do not predict future returns; headlines can be incomplete, "
    "late or wrong, and the models can misread them. Do your own research or consult a licensed adviser "
    "before making any investment decision."
)

# The brief as Markdown. {chart} is replaced by an image line pointing at the
# PNG file for the .md, and at an embedded data URI for the HTML.
REPORT_TEMPLATE = """\
# {title}

Generated {generated_at} · market data as of {as_of}

## Company snapshot

| Measure | Value |
|---|---|
| Current price | {price} |
| 52-week high | {high_52w} |
| 52-week low | {low_52w} |
| P/E ratio (trailing) | {pe_ratio} |
| Year-to-date return | {ytd} |
| Momentum signal (rule-based) | {momentum} |

## Technical outlook

{chart}

{technical_facts}

## News sentiment

{sentiment}

## Recommendation: {recommendation} over the next {horizon_days} days

{placeholder}{justification}

**Key factors**

{key_factors}

## How this brief was made

{method}
{data_notes}
---

**Risk disclaimer.** {disclaimer}
"""

# Inline CSS only, so the saved file is self-contained.
HTML_TEMPLATE = Template("""\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<style>
  body { margin: 0; background: #f4f3f0; color: #0b0b0b;
         font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  main { max-width: 860px; margin: 24px auto; padding: 32px 40px; background: #fcfcfb;
         border: 1px solid #e4e3df; border-radius: 8px; }
  h1 { font-size: 24px; margin: 0 0 4px; }
  h1 + p { color: #52514e; margin-top: 0; }
  h2 { font-size: 18px; margin: 28px 0 8px; padding-bottom: 4px; border-bottom: 1px solid #e4e3df; }
  table { border-collapse: collapse; width: 100%; max-width: 480px; }
  th, td { text-align: left; padding: 5px 10px; border-bottom: 1px solid #e4e3df; }
  th { color: #52514e; font-weight: 600; }
  td:last-child { font-variant-numeric: tabular-nums; }
  img { width: 100%; height: auto; border: 1px solid #e4e3df; border-radius: 4px; }
  blockquote { margin: 8px 0; padding: 8px 14px; background: #fdf3e7; border-left: 4px solid #eda100; }
  blockquote p { margin: 0; }
  hr { border: 0; border-top: 1px solid #e4e3df; margin: 28px 0 12px; }
  hr + p { color: #52514e; font-size: 13px; }
  @media (max-width: 640px) { main { margin: 0; padding: 20px 16px; border-radius: 0; } }
</style>
</head>
<body>
<main>
$body
</main>
</body>
</html>
""")


@dataclass(frozen=True)
class Report:
    """The brief in both formats, plus the chart it embeds (None when it could not be drawn)."""

    ticker: str
    markdown: str  # refers to the chart as <ticker>_chart.png beside it
    html: str  # self-contained: the chart is embedded
    chart_png: bytes | None


def plot_price_chart(prices: pd.DataFrame, ticker: str = "") -> Figure:
    """Close with SMA-50/200 and the Bollinger Bands, then RSI, then MACD, over the last year.

    Returns a Figure without touching pyplot's global state, so it is safe in
    tests and in a notebook (display(fig) shows it).
    """
    recent = _last_year(prices)
    fig = Figure(figsize=CHART_FIGSIZE, dpi=CHART_DPI, facecolor=SURFACE)
    price_ax, rsi_ax, macd_ax = fig.subplots(
        3, 1, sharex=True, gridspec_kw={"height_ratios": CHART_HEIGHT_RATIOS}
    )
    if recent.empty:
        price_ax.text(0.5, 0.5, "No price data", ha="center", va="center", color=INK_MUTED, transform=price_ax.transAxes)
    else:
        _price_panel(price_ax, recent)
        _rsi_panel(rsi_ax, recent)
        _macd_panel(macd_ax, recent)
        start, end = recent.index[0].date(), recent.index[-1].date()
        price_ax.set_title(f"{ticker} daily close and indicators, {start} to {end}".strip(), loc="left", color=INK)
    for ax in (price_ax, rsi_ax, macd_ax):
        _style(ax)
    fig.align_ylabels()
    fig.tight_layout()
    return fig


def top_headlines(
    market: MarketData, analysis: Analysis, n: int = TOP_HEADLINES
) -> list[tuple[HeadlineSentiment, Headline | None]]:
    """The `n` scored headlines with the largest |score|, ties to the newer one.

    Headlines arrive newest first and Python's sort is stable, so sorting on
    |score| alone keeps the newer headline ahead in a tie.
    """
    sources: list[Headline | None] = list(market.headlines)
    pairs = [
        (item, sources[i] if i < len(sources) else None)
        for i, item in enumerate(analysis.headline_sentiments)
        if item.score is not None
    ]
    return sorted(pairs, key=lambda pair: abs(pair[0].score), reverse=True)[:n]


def build_report(market: MarketData, analysis: Analysis, *, generated_at: datetime) -> Report:
    """The brief for one run. `generated_at` is passed in so the output is reproducible in tests."""
    summary = market.summary
    ticker = summary.get("ticker") or ""
    notes = [*summary.get("warnings", []), *analysis.warnings]

    chart_png = _chart_png(market.prices, ticker, notes)
    fields = _fields(market, analysis, generated_at, notes)
    alt = CHART_ALT.format(ticker=ticker)
    if chart_png is None:
        md_chart = html_chart = NO_CHART
    else:
        md_chart = f"![{alt}]({ticker}_chart.png)"
        html_chart = f"![{alt}](data:image/png;base64,{base64.b64encode(chart_png).decode('ascii')})"

    md = REPORT_TEMPLATE.format(chart=md_chart, **fields)
    body = markdown.markdown(REPORT_TEMPLATE.format(chart=html_chart, **fields), extensions=["tables"])
    html = HTML_TEMPLATE.substitute(title=fields["title"], body=body)
    return Report(ticker=ticker, markdown=md, html=html, chart_png=chart_png)


def save_report(report: Report, out_dir: Path | str = REPORTS_DIR) -> list[Path]:
    """Write <ticker>_brief.md, <ticker>_brief.html and <ticker>_chart.png. Returns the paths written."""
    out = Path(out_dir)
    files: dict[str, bytes] = {
        f"{report.ticker}_brief.md": report.markdown.encode("utf-8"),
        f"{report.ticker}_brief.html": report.html.encode("utf-8"),
    }
    if report.chart_png is not None:
        files[f"{report.ticker}_chart.png"] = report.chart_png
    written = []
    try:
        out.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            path = out / name
            path.write_bytes(content)
            written.append(path)
    except OSError as exc:
        logger.warning("could not save the report to %s: %s", out, exc)
    return written


# --- page sections -----------------------------------------------------------


def _fields(market: MarketData, analysis: Analysis, generated_at: datetime, notes: list[str]) -> dict[str, str]:
    summary = market.summary
    ticker = summary.get("ticker") or ""
    currency = f" {summary['currency']}" if summary.get("currency") else ""
    momentum = summary.get("momentum") or {}
    rec = analysis.recommendation
    return {
        "title": REPORT_TITLE.format(company=_escape(summary.get("company_name") or ticker), ticker=ticker),
        "generated_at": generated_at.strftime("%Y-%m-%d %H:%M %Z").strip(),
        "as_of": summary.get("as_of") or UNAVAILABLE,
        "price": _number(summary.get("current_price"), "{:.2f}" + currency),
        "high_52w": _number(summary.get("high_52w"), "{:.2f}" + currency),
        "low_52w": _number(summary.get("low_52w"), "{:.2f}" + currency),
        "pe_ratio": _number(summary.get("pe_ratio"), "{:.1f}"),
        "ytd": _number(summary.get("ytd_return_pct"), "{:+.1f}%"),
        "momentum": f"{momentum['label']} (score {momentum['score']:+d} of ±5)" if momentum else UNAVAILABLE,
        "technical_facts": _bullets(technical_facts(market.prices, summary)),
        "sentiment": _sentiment_section(market, analysis),
        "recommendation": rec.value.recommendation,
        "horizon_days": str(RECOMMENDATION_HORIZON_DAYS),
        "placeholder": "" if rec.ok else PLACEHOLDER_NOTE + "\n\n",
        "justification": _escape(rec.value.justification),
        "key_factors": _bullets([_escape(factor) for factor in rec.value.key_factors]),
        "method": _method(analysis),
        "data_notes": f"\n## Data notes\n\n{_bullets([_escape(n) for n in notes])}\n" if notes else "",
        "disclaimer": RISK_DISCLAIMER,
    }


def _sentiment_section(market: MarketData, analysis: Analysis) -> str:
    if not market.headlines:
        return NO_HEADLINES
    sentiment = analysis.sentiment
    if sentiment.score is None:
        return NO_SCORED_HEADLINES
    counts = sentiment.counts
    lines = [
        (
            f"**Sentiment score: {sentiment.score:+.2f} ({sentiment.label})** on a scale from -1 to +1, "
            f"the mean of {sentiment.scored} headlines from the last {NEWS_MAX_AGE_DAYS} days: "
            f"{counts.get('positive', 0)} positive, {counts.get('negative', 0)} negative, "
            f"{counts.get('neutral', 0)} neutral."
        ),
        "",
        f"Top {TOP_HEADLINES} headlines by strength of sentiment:",
        "",
    ]
    for rank, (item, source) in enumerate(top_headlines(market, analysis), start=1):
        byline = ""
        if source is not None:
            publisher = _escape(source.publisher) if source.publisher else "unknown publisher"
            byline = f" ({publisher}, {source.published:%Y-%m-%d})"
        lines.append(
            f"{rank}. **{_escape(item.headline)}**{byline}: {item.sentiment}, score {item.score:+.2f}. "
            f"{_escape(item.brief_reason)}"
        )
    return "\n".join(lines)


def _method(analysis: Analysis) -> str:
    sentiment = analysis.sentiment
    rec = analysis.recommendation
    writer = f"{_model_name(rec.provider)} via {rec.provider}" if rec.ok and rec.provider else "no model (placeholder)"
    return _bullets(
        [
            (
                f"Indicators computed from first principles with pandas and numpy: "
                f"SMA {SMA_SHORT_WINDOW} and {SMA_LONG_WINDOW}, RSI {RSI_PERIOD} (Wilder), "
                f"MACD ({MACD_FAST_SPAN}, {MACD_SLOW_SPAN}, {MACD_SIGNAL_SPAN}), "
                f"Bollinger Bands ({BB_WINDOW}, {BB_NUM_STD:g} standard deviations)."
            ),
            (
                f"Headline sentiment labelled by Jev for {sentiment.by_jev} headlines and by the LLM for "
                f"{sentiment.by_llm}; {sentiment.failed} could not be scored."
            ),
            f"Recommendation written by {writer}, from the computed facts above and the Sentiment score.",
        ]
    )


def _model_name(provider: str | None) -> str:
    """The model behind a provider name, looked up in the profiles; the provider name if unknown."""
    for profile in PROFILES.values():
        for candidate in (profile.primary, profile.fallback):
            if candidate is not None and candidate.name == provider:
                return candidate.model
    return provider or UNAVAILABLE


# --- chart panels ------------------------------------------------------------


def _chart_png(prices: pd.DataFrame, ticker: str, notes: list[str]) -> bytes | None:
    # Any plotting error is caught on purpose: the brief is still useful without its chart.
    try:
        fig = plot_price_chart(prices, ticker)
        buffer = BytesIO()
        fig.savefig(buffer, format="png", facecolor=SURFACE)
        return buffer.getvalue()
    except Exception as exc:  # noqa: BLE001
        message = f"the price chart could not be drawn: {exc}"
        logger.warning(message)
        notes.append(message)
        return None


def _last_year(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or prices.empty or DEFAULT_PRICE_COLUMN not in prices.columns:
        return pd.DataFrame()
    start = prices.index[-1] - pd.Timedelta(days=CHART_LOOKBACK_DAYS)
    return prices.loc[prices.index >= start]


def _price_panel(ax, df: pd.DataFrame) -> None:
    if {COL_BB_LOWER, COL_BB_UPPER} <= set(df.columns):
        ax.fill_between(
            df.index, df[COL_BB_LOWER], df[COL_BB_UPPER], color=BAND_FILL, linewidth=0,
            label=f"Bollinger Bands ({BB_WINDOW}, {BB_NUM_STD:g}σ)",
        )
    ax.plot(df.index, df[DEFAULT_PRICE_COLUMN], color=SERIES_1, linewidth=LINE_WIDTH, label="Close")
    for column, colour, window in ((COL_SMA_SHORT, SERIES_2, SMA_SHORT_WINDOW), (COL_SMA_LONG, SERIES_3, SMA_LONG_WINDOW)):
        if column in df.columns:
            ax.plot(df.index, df[column], color=colour, linewidth=LINE_WIDTH, label=f"SMA {window}")
    ax.set_ylabel("Price")
    _legend(ax, ncols=4)


def _rsi_panel(ax, df: pd.DataFrame) -> None:
    if COL_RSI in df.columns:
        ax.plot(df.index, df[COL_RSI], color=SERIES_1, linewidth=THIN_LINE_WIDTH)
    for level in (RSI_OVERSOLD, RSI_OVERBOUGHT):
        ax.axhline(level, color=INK_MUTED, linewidth=THIN_LINE_WIDTH, linestyle="--")
    ax.set_ylim(*RSI_AXIS_LIMITS)
    ax.set_yticks([RSI_OVERSOLD, RSI_OVERBOUGHT])
    ax.set_ylabel(f"RSI {RSI_PERIOD}")


def _macd_panel(ax, df: pd.DataFrame) -> None:
    if COL_MACD_HIST in df.columns:
        hist = df[COL_MACD_HIST]
        ax.bar(df.index, hist, color=[HIST_UP if v >= 0 else HIST_DOWN for v in hist.fillna(0)], width=1.0)
    if COL_MACD in df.columns:
        ax.plot(df.index, df[COL_MACD], color=SERIES_1, linewidth=THIN_LINE_WIDTH, label="MACD")
    if COL_MACD_SIGNAL in df.columns:
        ax.plot(df.index, df[COL_MACD_SIGNAL], color=SERIES_2, linewidth=THIN_LINE_WIDTH, label="Signal")
    ax.axhline(0, color=INK_MUTED, linewidth=THIN_LINE_WIDTH / 2)
    ax.set_ylabel("MACD")
    # The histogram's swatch shows both colours, since its colour carries the sign.
    handles, labels = ax.get_legend_handles_labels()
    handles.append((Patch(color=HIST_UP), Patch(color=HIST_DOWN)))
    labels.append("Histogram (above / below 0)")
    _legend(ax, ncols=3, handles=handles, labels=labels, handler_map={tuple: HandlerTuple(ndivide=None)})


def _legend(ax, ncols: int, **kwargs) -> None:
    """A legend on a near-opaque surface patch, so lines passing under it stay readable."""
    ax.legend(
        loc="upper left", ncols=ncols, fontsize="small", labelcolor=INK,
        frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=LEGEND_ALPHA, **kwargs,
    )


def _style(ax) -> None:
    """Recessive axes: no top or right spine, light horizontal grid, muted tick labels."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=THIN_LINE_WIDTH / 2)
    ax.tick_params(colors=INK_MUTED, labelsize="small")
    ax.yaxis.label.set_color(INK_MUTED)


# --- formatting --------------------------------------------------------------


def _number(value: float | None, spec: str) -> str:
    return spec.format(value) if value is not None else UNAVAILABLE


def _bullets(lines: list[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)


def _escape(text: str) -> str:
    """Make outside text (headlines, model output) inert in Markdown and in the HTML it becomes.

    Markdown control characters are backslash-escaped, and '<' becomes an
    entity so a headline cannot inject HTML into the page.
    """
    for char in "\\`*_[]":
        text = text.replace(char, "\\" + char)
    return text.replace("<", "&lt;")
