"""News ingestion utilities for the AIDC pipeline."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import parse_qs, urlparse

import httpx
import feedparser
from dateutil import parser as dateparser

from .utils import (
    CONFIG_DIR,
    OUT_DIR,
    RateLimiter,
    alias_matches,
    load_companies,
    load_keywords,
    load_rss_feeds,
    log_json,
    save_jsonl,
    setup_json_logger,
)

GOOGLE_NEWS_URL = "https://news.google.com/rss/search"


@dataclass
class Article:
    id: str
    title: str
    url: str
    published_at: datetime
    source: str
    summary: str
    ticker: str
    company: str
    tier: str
    story_id: str
    raw: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["published_at"] = self.published_at.isoformat()
        return payload


def _hash_story(url: str, title: str) -> str:
    digest = hashlib.sha256((url + title).encode("utf-8")).hexdigest()
    return digest[:16]


def unwrap_google_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.endswith("news.google.com"):
        query = parse_qs(parsed.query)
        if "url" in query:
            return query["url"][0]
    return url


def _infer_tier(domain: str) -> str:
    tiers_path = CONFIG_DIR / "domain_tiers.json"
    if tiers_path.exists():
        tiers = json.loads(tiers_path.read_text(encoding="utf-8"))
        for tier, domains in tiers.items():
            if domain in domains:
                return tier
    return "tier3"


def parse_entry(entry: Dict[str, Any], ticker_map: Dict[str, Dict[str, Any]]) -> Optional[Article]:
    title = entry.get("title", "").strip()
    summary = entry.get("summary", "").strip()
    link = unwrap_google_url(entry.get("link", ""))
    published = entry.get("published") or entry.get("updated")
    if not (title and link and published):
        return None
    try:
        published_dt = dateparser.parse(published)
        if not published_dt.tzinfo:
            published_dt = published_dt.replace(tzinfo=timezone.utc)
        published_dt = published_dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        published_dt = datetime.now(tz=timezone.utc)

    lower_title = title.lower()
    matched_ticker = ""
    matched_company = ""
    matched_aliases: List[str] = []
    for ticker, row in ticker_map.items():
        aliases = [row["company"]] + row["aliases"].split("|")
        aliases = [alias.strip() for alias in aliases if alias.strip()]
        if alias_matches(lower_title, aliases) or alias_matches(summary.lower(), aliases):
            matched_ticker = ticker
            matched_company = row["company"]
            matched_aliases = aliases
            break
    if not matched_ticker:
        return None

    domain = urlparse(link).netloc
    tier = _infer_tier(domain)
    story_id = _hash_story(link, title)
    article_id = _hash_story(link, published_dt.isoformat())

    return Article(
        id=article_id,
        title=title,
        url=link,
        published_at=published_dt,
        source=domain,
        summary=summary,
        ticker=matched_ticker,
        company=matched_company,
        tier=tier,
        story_id=story_id,
        raw={"aliases": matched_aliases, "entry": entry},
    )


def _ingest_google_news(days: int, queries: List[str]) -> List[Article]:
    logger = setup_json_logger("ingest", OUT_DIR / "ingest.log")
    rate_limiter = RateLimiter(min_delay=float(int(os.environ.get("AIDC_NEWSAPI_THROTTLE_MS", "800"))) / 1000.0)
    articles: List[Article] = []
    client = httpx.Client(timeout=20.0, headers={"User-Agent": "aidc-signals/1.0"})
    companies = load_companies()
    ticker_map = {
        row["ticker"]: {"company": row["company"], "aliases": row.get("aliases", "")}
        for row in companies.to_dict(orient="records")
    }

    for query in queries:
        params = {"q": f"{query} when:{days}d", "hl": "en-US", "gl": "US", "ceid": "US:en"}
        url = GOOGLE_NEWS_URL + "?" + httpx.QueryParams(params).render()
        rate_limiter.wait()
        try:
            response = client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log_json(logger, event="google_news_error", query=query, error=str(exc))
            continue
        feed = feedparser.parse(response.text)
        for entry in feed.entries:
            article = parse_entry(entry, ticker_map)
            if article:
                articles.append(article)
                log_json(
                    logger,
                    event="article_hit",
                    ticker=article.ticker,
                    story_id=article.story_id,
                    source=article.source,
                    title=article.title,
                )
    client.close()
    return articles


def _ingest_rss(days: int) -> List[Article]:
    logger = setup_json_logger("ingest", OUT_DIR / "ingest.log")
    feeds = load_rss_feeds()
    companies = load_companies()
    ticker_map = {
        row["ticker"]: {"company": row["company"], "aliases": row.get("aliases", "")}
        for row in companies.to_dict(orient="records")
    }
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    articles: List[Article] = []
    for feed_url in feeds:
        try:
            parsed = feedparser.parse(feed_url)
        except Exception as exc:  # pylint: disable=broad-except
            log_json(logger, event="rss_error", feed=feed_url, error=str(exc))
            continue
        for entry in parsed.entries:
            published = entry.get("published") or entry.get("updated")
            if not published:
                continue
            try:
                published_dt = dateparser.parse(published)
                if not published_dt.tzinfo:
                    published_dt = published_dt.replace(tzinfo=timezone.utc)
                published_dt = published_dt.astimezone(timezone.utc)
            except (ValueError, TypeError):
                continue
            if published_dt < cutoff:
                continue
            article = parse_entry(entry, ticker_map)
            if article:
                articles.append(article)
    return articles


def deduplicate(articles: Iterable[Article]) -> List[Article]:
    seen: Dict[str, Article] = {}
    for article in articles:
        key = article.story_id
        if key not in seen or seen[key].published_at < article.published_at:
            seen[key] = article
    return list(seen.values())


def ingest(days: int = 2, use_google: bool = True, use_rss: bool = True) -> List[Dict[str, Any]]:
    keywords = load_keywords()
    if not keywords:
        companies = load_companies()
        keywords = companies["company"].tolist()
    google_articles: List[Article] = []
    rss_articles: List[Article] = []
    if use_google:
        google_articles = _ingest_google_news(days, keywords)
    if use_rss:
        rss_articles = _ingest_rss(days)
    all_articles = deduplicate([*google_articles, *rss_articles])
    payloads = [article.to_dict() for article in all_articles]
    save_jsonl(OUT_DIR / "articles.jsonl", payloads)
    return payloads


__all__ = ["Article", "ingest", "deduplicate", "unwrap_google_url"]
