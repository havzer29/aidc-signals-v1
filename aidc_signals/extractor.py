"""OpenAI-powered extraction of structured supply/demand events."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from openai import OpenAI

from .taxonomy import is_valid_event, taxonomy_names, taxonomy_prompt
from .utils import (
    CACHE_DIR,
    OUT_DIR,
    Backoff,
    RateLimiter,
    env_float,
    env_int,
    load_json,
    log_json,
    save_jsonl,
    setup_json_logger,
)


SYSTEM_PROMPT = """You are an equity analyst focused on AI and datacenter supply-demand balances.
Extract only material events that map to the provided taxonomy. When in doubt,
return an empty list rather than guessing. Direction must be +1 (event pushes
in the nominal direction described) or -1 (event moves opposite). Confidence
and relevance are bounded in [0,1]. Sentiment ranges [-1,1]. Never fabricate
numeric values; use null when no explicit number is present."""

JSON_EXAMPLE = {
    "events": [
        {
            "event_type": "capacity_down",
            "direction": -1,
            "confidence": 0.8,
            "relevance": 0.7,
            "sentiment": -0.3,
            "numeric_value": None,
            "unit": None,
        }
    ]
}


@dataclass
class ExtractionResult:
    """Structured result from the extraction API."""

    article_id: str
    ticker: str
    events: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {"article_id": self.article_id, "ticker": self.ticker, "events": self.events}


def _tool_spec() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "events_schema",
            "description": "Return structured supply/demand events for the provided article.",
            "parameters": {
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
        },
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


def _build_prompt(article: Dict[str, Any]) -> str:
    taxonomy_text = taxonomy_prompt()
    return (
        "Taxonomy (role, impact_dir, weight):\n"
        f"{taxonomy_text}\n\n"
        "Return strictly valid JSON matching the schema. Example:\n"
        f"{json.dumps(JSON_EXAMPLE, ensure_ascii=False)}\n\n"
        f"Ticker: {article['ticker']} (Company: {article['company']})\n"
        f"Title: {article['title']}\n"
        f"Published: {article['published_at']}\n"
        f"Source: {article.get('source', '')}\n"
        f"Content:\n{article.get('text') or article.get('summary', '')}"
    )


def extract_events(article: Dict[str, Any]) -> ExtractionResult:
    logger = setup_json_logger("extract", OUT_DIR / "pipeline.log")
    try:
        client = _client()
    except EnvironmentError as exc:
        log_json(
            logger,
            event="extract_skipped",
            article_id=article.get("id"),
            ticker=article.get("ticker"),
            error=str(exc),
        )
        return ExtractionResult(article_id=article["id"], ticker=article["ticker"], events=[])

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    prompt = _build_prompt(article)

    _LIMITER.wait()
    attempts = env_int("AIDC_OPENAI_MAX_ATTEMPTS", 5)
    for attempt in range(attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tools=[_tool_spec()],
                tool_choice={"type": "function", "function": {"name": "events_schema"}},
            )
            message = response.choices[0].message
            data: Dict[str, Any] = {"events": []}
            if message.tool_calls:
                for call in message.tool_calls:
                    if call.function and call.function.name == "events_schema":
                        try:
                            args = call.function.arguments or "{}"
                            data = json.loads(args)
                        except json.JSONDecodeError:
                            data = {"events": []}
                        break
            elif message.content:
                try:
                    data = json.loads(message.content)
                except json.JSONDecodeError:
                    data = {"events": []}

            events = data.get("events", []) if isinstance(data, dict) else []
            filtered = [
                event
                for event in events
                if is_valid_event(event.get("event_type", ""))
                and isinstance(event.get("direction"), int)
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
                attempt=attempt + 1,
                delay=delay,
                error=str(exc),
            )
            time.sleep(delay)
    return ExtractionResult(article_id=article["id"], ticker=article["ticker"], events=[])


def extract_batch(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cache_path = CACHE_DIR / "events_cache.jsonl"
    cached_rows = load_json(cache_path)
    cache_index: Dict[str, Dict[str, Any]] = {row["article_id"]: row for row in cached_rows}
    force_refresh = bool(int(os.environ.get("AIDC_FORCE_REFRESH", "0")))

    results: List[Dict[str, Any]] = []
    for article in articles:
        cached = cache_index.get(article["id"]) if not force_refresh else None
        if cached is not None:
            results.append(cached)
            continue
        result = extract_events(article)
        payload = result.to_dict()
        cache_index[article["id"]] = payload
        results.append(payload)

    save_jsonl(cache_path, list(cache_index.values()))
    return results


__all__ = ["extract_events", "extract_batch", "ExtractionResult"]
