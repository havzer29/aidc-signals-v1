"""Full-text content fetching utilities."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List

import httpx

try:
    import trafilatura  # type: ignore
except Exception:  # pylint: disable=broad-except
    trafilatura = None

from .utils import CACHE_DIR, OUT_DIR, RateLimiter, env_int, log_json, save_jsonl, setup_json_logger


@dataclass
class ArticleContent:
    article_id: str
    url: str
    title: str
    fetched_at: datetime
    text: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "article_id": self.article_id,
            "url": self.url,
            "title": self.title,
            "fetched_at": self.fetched_at.isoformat(),
            "text": self.text,
        }


async def _fetch_single(
    client: httpx.AsyncClient,
    limiter: RateLimiter,
    article: Dict[str, str],
    logger_name: str,
) -> ArticleContent:
    limiter.wait()
    logger = setup_json_logger(logger_name, OUT_DIR / "pipeline.log")
    try:
        response = await client.get(article["url"], timeout=float(env_int("AIDC_FETCH_TIMEOUT", 20)))
        response.raise_for_status()
        text = response.text
        if trafilatura is not None:
            try:
                extracted = trafilatura.extract(text, url=article["url"], include_comments=False)
                if extracted:
                    text = extracted
            except Exception as exc:  # pylint: disable=broad-except
                log_json(logger, event="trafilatura_error", url=article["url"], error=str(exc))
        return ArticleContent(
            article_id=article["id"],
            url=article["url"],
            title=article["title"],
            fetched_at=datetime.utcnow(),
            text=text,
        )
    except httpx.HTTPError as exc:
        log_json(logger, event="fetch_error", url=article["url"], error=str(exc))
        return ArticleContent(
            article_id=article["id"],
            url=article["url"],
            title=article["title"],
            fetched_at=datetime.utcnow(),
            text=article.get("summary", ""),
        )


async def fetch_contents_async(articles: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    concurrency = env_int("AIDC_FETCH_CONCURRENCY", 6)
    rate = RateLimiter(min_delay=0.0)
    logger_name = "fetch"
    async with httpx.AsyncClient(headers={"User-Agent": "aidc-signals/1.0"}) as client:
        tasks = []
        sem = asyncio.Semaphore(concurrency)

        async def bound_fetch(article: Dict[str, str]) -> ArticleContent:
            async with sem:
                return await _fetch_single(client, rate, article, logger_name)

        for article in articles:
            tasks.append(asyncio.create_task(bound_fetch(article)))
        results = await asyncio.gather(*tasks)

    payloads = [content.to_dict() for content in results]
    cache_path = CACHE_DIR / "articles_full.jsonl"
    save_jsonl(cache_path, payloads)
    save_jsonl(OUT_DIR / "articles_full.jsonl", payloads)
    return payloads


def fetch_contents(articles: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    return asyncio.run(fetch_contents_async(articles))


__all__ = ["fetch_contents", "fetch_contents_async", "ArticleContent"]
