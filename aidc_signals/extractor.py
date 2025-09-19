"""OpenAI-based event extractor."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from openai import OpenAI

from .taxonomy import TAXONOMY, taxonomy_names
from .utils import OUT_DIR, CACHE_DIR, Backoff, RateLimiter, env_float, log_json, save_jsonl, setup_json_logger


SYSTEM_PROMPT = """You are an analyst extracting supply/demand events linked to AI infrastructure companies. Only extract events explicitly supported by the taxonomy."""


@dataclass
class ExtractionResult:
    article_id: str
    ticker: str
    events: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {"article_id": self.article_id, "ticker": self.ticker, "events": self.events}


def _schema() -> Dict[str, Any]:
    return {
        "name": "events_schema",
        "schema": {
            "type": "object",
            "properties": {
                "events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "event_type": {"type": "string", "enum": taxonomy_names()},
                            "direction": {"type": "integer", "enum": [-1, 1]},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "relevance": {"type": "number", "minimum": 0, "maximum": 1},
                            "sentiment": {"type": "number", "minimum": -1, "maximum": 1},
                            "numeric_value": {"type": ["number", "null"]},
                            "unit": {"type": ["string", "null"]},
                        },
                        "required": [
                            "event_type",
                            "direction",
                            "confidence",
                            "relevance",
                            "sentiment",
                            "numeric_value",
                            "unit",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["events"],
            "additionalProperties": False,
        },
        "strict": True,
    }


_CLIENT: OpenAI | None = None
_LIMITER = RateLimiter(min_delay=env_float("AIDC_OPENAI_MIN_DELAY_MS", 300) / 1000.0)
_BACKOFF = Backoff(base_delay=env_float("AIDC_OPENAI_BACKOFF_MS", 800) / 1000.0)


def _client() -> OpenAI:
    global _CLIENT  # pylint: disable=global-statement
    if _CLIENT is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is required")
        _CLIENT = OpenAI(api_key=api_key)
    return _CLIENT


def extract_events(article: Dict[str, Any]) -> ExtractionResult:
    client = _client()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    logger = setup_json_logger("extract", OUT_DIR / "pipeline.log")

    prompt = (
        f"Ticker: {article['ticker']} (Company: {article['company']})\n"
        f"Title: {article['title']}\n"
        f"Published: {article['published_at']}\n"
        f"Content:\n{article.get('text') or article.get('summary', '')}"
    )

    _LIMITER.wait()
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_schema", "json_schema": _schema()},
            )
            content = response.choices[0].message.content or "{}"
            data = json.loads(content)
            events = data.get("events", [])
            filtered = [
                event
                for event in events
                if event.get("event_type") in TAXONOMY and isinstance(event.get("direction"), int)
            ]
            log_json(
                logger,
                event="extract_ok",
                article_id=article["id"],
                ticker=article["ticker"],
                events=len(filtered),
                prompt_tokens=getattr(response.usage, "prompt_tokens", None),
                completion_tokens=getattr(response.usage, "completion_tokens", None),
            )
            return ExtractionResult(article_id=article["id"], ticker=article["ticker"], events=filtered)
        except Exception as exc:  # pylint: disable=broad-except
            delay = _BACKOFF.compute(attempt)
            log_json(
                logger,
                event="extract_retry",
                article_id=article["id"],
                attempt=attempt,
                delay=delay,
                error=str(exc),
            )
            import time

            time.sleep(delay)
    return ExtractionResult(article_id=article["id"], ticker=article["ticker"], events=[])


def extract_batch(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for article in articles:
        result = extract_events(article)
        results.append(result.to_dict())
    save_jsonl(CACHE_DIR / "events_cache.jsonl", results)
    return results


__all__ = ["extract_events", "extract_batch", "ExtractionResult"]
