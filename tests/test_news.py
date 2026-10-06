"""Offline tests for task1_financial/news.py, with the network replaced by fakes.

The two fixture files are hand-written in the shape of the real Yahoo and
Google News feeds; the `rss()` builder makes bigger feeds for the counting
rules. The real feeds are checked once by tests/test_live_news.py.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 1A step 2, news headlines, as designed in the grilling rounds', Date: 2026-10-06

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError

import pytest

from task1_financial import data, news
from task1_financial.data import FETCH_RETRIES
from task1_financial.news import (
    FEED_GOOGLE,
    FEED_YAHOO,
    MIN_HEADLINES,
    NEWS_MAX_AGE_DAYS,
    NEWS_MAX_COUNT,
    NEWS_TARGET,
    fetch_headlines,
    parse_feed,
)

TODAY = date(2026, 10, 6)
FIXTURES = Path(__file__).parent / "fixtures"
YAHOO_SAMPLE = (FIXTURES / "yahoo_rss_sample.xml").read_bytes()
GOOGLE_SAMPLE = (FIXTURES / "google_news_rss_sample.xml").read_bytes()


def rss(items: list[tuple[str, str]], publisher: str | None = None) -> bytes:
    """A minimal RSS feed of (title, pubDate) items; with `publisher`, Google-style titles and <source> tags."""
    body = []
    for title, published in items:
        if publisher:
            body.append(f"<item><title>{title} - {publisher}</title><pubDate>{published}</pubDate><source>{publisher}</source></item>")
        else:
            body.append(f"<item><title>{title}</title><pubDate>{published}</pubDate></item>")
    return f"<rss version='2.0'><channel>{''.join(body)}</channel></rss>".encode()


def hours_before_today(count: int, prefix: str) -> list[tuple[str, str]]:
    """`count` distinct titles published one hour apart, counting back from noon UTC on TODAY."""
    noon = datetime(TODAY.year, TODAY.month, TODAY.day, 12, tzinfo=timezone.utc)
    return [(f"{prefix} story {i}", (noon - timedelta(hours=i)).strftime("%a, %d %b %Y %H:%M:%S GMT")) for i in range(count)]


class FakeNetwork:
    """Stands in for news._download: answers each feed with fixed bytes or an exception, and records the calls."""

    def __init__(self, yahoo, google):
        self.answers = {FEED_YAHOO: yahoo, FEED_GOOGLE: google}
        self.calls: list[str] = []

    def __call__(self, url: str) -> bytes:
        feed = FEED_YAHOO if "yahoo.com" in url else FEED_GOOGLE
        self.calls.append(feed)
        answer = self.answers[feed]
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(data.time, "sleep", lambda seconds: None)


def use_network(monkeypatch, yahoo, google) -> FakeNetwork:
    network = FakeNetwork(yahoo, google)
    monkeypatch.setattr(news, "_download", network)
    return network


# --- parsing --------------------------------------------------------------


def test_parse_yahoo_sample_keeps_dated_titled_items():
    headlines = parse_feed(YAHOO_SAMPLE, FEED_YAHOO)

    # The undated item, the unreadable date and the blank title are dropped.
    assert [h.title for h in headlines] == [
        "Apple shares climb ahead of product event",
        "Analysts raise Apple price target & cite iPhone demand",  # whitespace collapsed, &amp; decoded
        "Apple faces EU antitrust fine - what it means",  # no <source> tag, so the " - " part stays
    ]
    first = headlines[0]
    assert first.publisher is None and first.feed == FEED_YAHOO
    assert first.published == datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
    assert first.url.startswith("https://finance.yahoo.com/news/")


def test_parse_google_sample_moves_publisher_out_of_the_title():
    headlines = parse_feed(GOOGLE_SAMPLE, FEED_GOOGLE)

    assert [(h.title, h.publisher) for h in headlines] == [
        ("Apple Shares Climb Ahead of Product Event", "Reuters"),
        # Found on Google's feed, written by Yahoo Finance: feed and publisher differ.
        ("Here's How Many Shares of Apple You'd Need for $10,000 in Dividends", "Yahoo Finance"),
        ("Is Apple stock a buy - or a sell - after the rally?", None),
    ]
    assert all(h.feed == FEED_GOOGLE for h in headlines)


def test_dates_without_a_zone_are_read_as_utc():
    headlines = parse_feed(rss([("Undated zone", "Mon, 05 Oct 2026 12:00:00 -0000")]), FEED_YAHOO)

    assert headlines[0].published == datetime(2026, 10, 5, 12, tzinfo=timezone.utc)


# --- combining the feeds --------------------------------------------------


def test_samples_combine_newest_first_without_duplicates_or_old_items(monkeypatch):
    use_network(monkeypatch, YAHOO_SAMPLE, GOOGLE_SAMPLE)

    result = fetch_headlines("AAPL", TODAY)

    # Yahoo's 20 Sep item is too old, and Google's Reuters copy of the
    # "shares climb" story is a duplicate of Yahoo's, so Yahoo's is kept.
    assert [(h.title, h.feed) for h in result.headlines] == [
        ("Analysts raise Apple price target & cite iPhone demand", FEED_YAHOO),
        ("Here's How Many Shares of Apple You'd Need for $10,000 in Dividends", FEED_GOOGLE),
        ("Apple shares climb ahead of product event", FEED_YAHOO),
        ("Is Apple stock a buy - or a sell - after the rally?", FEED_GOOGLE),
    ]
    assert len(result.warnings) == 1 and f"below the minimum of {MIN_HEADLINES}" in result.warnings[0]


def test_google_is_not_called_when_yahoo_has_enough(monkeypatch):
    network = use_network(monkeypatch, rss(hours_before_today(NEWS_TARGET, "Yahoo")), GOOGLE_SAMPLE)

    result = fetch_headlines("AAPL", TODAY)

    assert len(result.headlines) == NEWS_TARGET
    assert network.calls == [FEED_YAHOO]
    assert result.warnings == []


def test_google_tops_up_a_short_yahoo_feed(monkeypatch):
    use_network(
        monkeypatch,
        rss(hours_before_today(4, "Yahoo")),
        rss(hours_before_today(20, "Google"), publisher="Reuters"),
    )

    result = fetch_headlines("AAPL", TODAY)

    assert len(result.headlines) == NEWS_TARGET
    assert sum(h.feed == FEED_YAHOO for h in result.headlines) == 4
    published = [h.published for h in result.headlines]
    assert published == sorted(published, reverse=True)
    assert result.warnings == []


def test_yahoo_down_falls_back_to_google(monkeypatch):
    rate_limited = HTTPError("https://feeds.finance.yahoo.com", 429, "Too Many Requests", None, None)
    network = use_network(monkeypatch, rate_limited, rss(hours_before_today(20, "Google"), publisher="Reuters"))

    result = fetch_headlines("AAPL", TODAY)

    assert len(result.headlines) == NEWS_TARGET
    assert network.calls.count(FEED_YAHOO) == FETCH_RETRIES
    assert len(result.warnings) == 1 and "yahoo news" in result.warnings[0] and "429" in result.warnings[0]


def test_both_feeds_down_gives_an_empty_list_and_warnings(monkeypatch):
    use_network(monkeypatch, ConnectionError("no network"), ConnectionError("no network"))

    result = fetch_headlines("AAPL", TODAY)

    assert result.headlines == []
    assert any("yahoo news" in w for w in result.warnings)
    assert any("google news" in w for w in result.warnings)
    assert any("only 0 headlines" in w for w in result.warnings)


def test_malformed_xml_is_retried_then_skipped(monkeypatch):
    network = use_network(monkeypatch, b"<rss><channel><item>", rss(hours_before_today(20, "Google"), publisher="AP"))

    result = fetch_headlines("AAPL", TODAY)

    assert network.calls.count(FEED_YAHOO) == FETCH_RETRIES
    assert len(result.headlines) == NEWS_TARGET
    assert any("ParseError" in w for w in result.warnings)


# --- recency and count ----------------------------------------------------


def test_age_cutoff_is_midnight_utc_max_age_days_ago(monkeypatch):
    cutoff = datetime(TODAY.year, TODAY.month, TODAY.day, tzinfo=timezone.utc) - timedelta(days=NEWS_MAX_AGE_DAYS)
    fmt = "%a, %d %b %Y %H:%M:%S GMT"
    items = [
        ("On the cutoff", cutoff.strftime(fmt)),
        ("One second too old", (cutoff - timedelta(seconds=1)).strftime(fmt)),
    ]
    use_network(monkeypatch, rss(items), rss([]))

    result = fetch_headlines("AAPL", TODAY)

    assert [h.title for h in result.headlines] == ["On the cutoff"]


def test_result_is_cut_to_the_newest_n(monkeypatch):
    use_network(monkeypatch, rss(hours_before_today(20, "Yahoo")), GOOGLE_SAMPLE)

    result = fetch_headlines("AAPL", TODAY, n=12)

    assert [h.title for h in result.headlines] == [f"Yahoo story {i}" for i in range(12)]


@pytest.mark.parametrize(("asked", "expected"), [(0, 1), (-3, 1), (500, NEWS_MAX_COUNT)])
def test_out_of_range_n_is_clamped_with_a_warning(monkeypatch, asked, expected):
    use_network(monkeypatch, rss(hours_before_today(60, "Yahoo")), rss([]))

    result = fetch_headlines("AAPL", TODAY, n=asked)

    assert len(result.headlines) == expected
    assert any(f"n={asked}" in w and f"using {expected}" in w for w in result.warnings)
