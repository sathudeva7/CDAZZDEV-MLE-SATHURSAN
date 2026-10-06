"""Recent news headlines for a ticker, from free public RSS feeds.

yfinance's own news call returned no items when probed, so this module reads
two RSS feeds directly (see GLOSSARY.md for Headline, Publisher and Feed):

- Feeds: Yahoo Finance's headline feed first, because it is finance-only.
  Google News search ("<ticker> stock") is read only when Yahoo leaves us
  short of `n`, which also covers Yahoo being down or rate-limiting (HTTP 429).
- Recency: a headline counts as recent when it was published on or after
  midnight UTC, NEWS_MAX_AGE_DAYS before `today`. Items with no date or an
  unreadable one are dropped, because they cannot be shown to be recent.
- Publisher: Google titles end in " - <Publisher>" and repeat the name in a
  <source> tag. The suffix is moved into `publisher` only when it matches the
  tag, so a title that merely contains " - " is left whole. Yahoo items carry
  no publisher.
- Duplicates: two headlines are the same story when their titles match after
  lowercasing and collapsing whitespace. The first one seen, Yahoo's, is kept.
- Failures: each feed is retried with data.with_retries. A feed that still
  fails becomes a warning; both failing gives an empty list, never an error.
- Parsing: the standard library's urllib and ElementTree. Python's bundled
  expat (2.4.1 and later) refuses entity-expansion ("billion laughs") XML, so
  no extra parser package is needed for remote feeds.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 1A step 2, news headlines, as designed in the grilling rounds', Date: 2026-10-06

from __future__ import annotations

import logging
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Literal
from urllib.parse import quote, quote_plus

from pydantic import BaseModel

from task1_financial.data import with_retries

logger = logging.getLogger(__name__)

NEWS_TARGET = 15  # headlines asked for by default: room above the minimum for dropped duplicates
MIN_HEADLINES = 10  # fewer than this is reported as a warning
NEWS_MAX_COUNT = 50  # largest n accepted; Google returns about 100 items per search
NEWS_MAX_AGE_DAYS = 7
NEWS_TIMEOUT_SECONDS = 10  # per request, so a stalled server cannot hang the notebook

FEED_YAHOO = "yahoo"
FEED_GOOGLE = "google"
YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
GOOGLE_NEWS_QUERY = "{ticker} stock"  # the company name alone also matches product reviews
# News sites often refuse Python's default User-Agent; the live probe used a browser's.
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
PUBLISHER_SEPARATOR = " - "


class Headline(BaseModel):
    """One recent article about the ticker: its title, who wrote it, when, and which feed it came from."""

    title: str
    publisher: str | None  # the outlet that wrote it; None when the feed does not say
    published: datetime  # timezone-aware, in UTC
    url: str | None
    feed: Literal["yahoo", "google"]


@dataclass
class NewsResult:
    """Headlines newest first, and any problems met while collecting them."""

    headlines: list[Headline]
    warnings: list[str] = field(default_factory=list)


def fetch_headlines(ticker: str, today: date, n: int = NEWS_TARGET) -> NewsResult:
    """Up to `n` recent, de-duplicated headlines for `ticker`, newest first. Never raises."""
    warnings: list[str] = []
    count = min(max(n, 1), NEWS_MAX_COUNT)
    if count != n:
        _warn(warnings, f"n={n} is outside 1..{NEWS_MAX_COUNT}, using {count}")
    cutoff = datetime.combine(today - timedelta(days=NEWS_MAX_AGE_DAYS), time.min, tzinfo=timezone.utc)

    feeds = [
        (FEED_YAHOO, YAHOO_RSS_URL.format(ticker=quote(ticker))),
        (FEED_GOOGLE, GOOGLE_NEWS_RSS_URL.format(query=quote_plus(GOOGLE_NEWS_QUERY.format(ticker=ticker)))),
    ]
    collected: list[Headline] = []
    seen: set[str] = set()
    for feed, url in feeds:
        if len(collected) >= count:
            break  # the earlier feed already had enough
        items = with_retries(
            lambda url=url, feed=feed: parse_feed(_download(url), feed),  # bound now, not at loop end
            what=f"{ticker} {feed} news",
            is_empty=lambda headlines: not headlines,
            warnings=warnings,
        ) or []
        recent = [h for h in items if h.published >= cutoff]
        added = 0
        for headline in recent:
            key = headline.title.lower()  # titles are already whitespace-collapsed
            if key not in seen:
                seen.add(key)
                collected.append(headline)
                added += 1
        logger.info("%s %s feed: %d items, %d recent, %d new", ticker, feed, len(items), len(recent), added)

    # sorted() is stable, so for equal times the earlier feed's headline stays first.
    headlines = sorted(collected, key=lambda h: h.published, reverse=True)[:count]
    expected = min(count, MIN_HEADLINES)
    if len(headlines) < expected:
        _warn(warnings, f"only {len(headlines)} headlines for {ticker} in the last {NEWS_MAX_AGE_DAYS} days, below the minimum of {expected}")
    return NewsResult(headlines, warnings)


def parse_feed(xml: bytes, feed: str) -> list[Headline]:
    """Every titled, dated <item> in an RSS document, in feed order. Raises ET.ParseError on broken XML."""
    root = ET.fromstring(xml)
    headlines: list[Headline] = []
    skipped = 0
    for item in root.iter("item"):
        title = " ".join((item.findtext("title") or "").split())
        published = _parse_date(item.findtext("pubDate"))
        if not title or published is None:
            skipped += 1
            continue
        publisher = " ".join((item.findtext("source") or "").split()) or None
        suffix = f"{PUBLISHER_SEPARATOR}{publisher}"
        if publisher and title.endswith(suffix):
            title = title[: -len(suffix)]
        url = (item.findtext("link") or "").strip() or None
        headlines.append(Headline(title=title, publisher=publisher, published=published, url=url, feed=feed))
    if skipped:
        logger.info("%s feed: skipped %d items with no title or no readable date", feed, skipped)
    return headlines


def _download(url: str) -> bytes:
    """The raw feed. Raises urllib's errors (HTTP 429, timeouts, DNS) for with_retries to handle."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=NEWS_TIMEOUT_SECONDS) as response:
        return response.read()


def _parse_date(text: str | None) -> datetime | None:
    """An RSS (RFC 822) date such as 'Mon, 05 Oct 2026 12:00:00 GMT' in UTC, or None if unreadable."""
    if not text:
        return None
    try:
        published = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if published.tzinfo is None:  # "-0000" means UTC with no zone stated
        published = published.replace(tzinfo=timezone.utc)
    return published.astimezone(timezone.utc)


def _warn(warnings: list[str], message: str) -> None:
    """Log a fallback and record it for the summary, so the two always agree."""
    logger.warning(message)
    warnings.append(message)
