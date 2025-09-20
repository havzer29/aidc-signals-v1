"""End-to-end pipeline orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .content_fetcher import fetch_contents
from .extractor import extract_batch
from .ingestion import ingest
from .scorer import aggregate_article_scores, aggregate_company_scores, score_events
from .utils import OUT_DIR, env_int, log_json, setup_json_logger


def run_pipeline(days: int = 2, use_google: bool = True, use_rss: bool = True) -> Dict[str, Path]:
    logger = setup_json_logger("pipeline", OUT_DIR / "pipeline.log")
    log_json(logger, event="pipeline_start", days=days, use_google=use_google, use_rss=use_rss)

    articles = ingest(days=days, use_google=use_google, use_rss=use_rss)
    if not articles:
        log_json(logger, event="no_articles")
        return {}
    articles_df = pd.DataFrame(articles)

    full_texts = fetch_contents(articles)
    full_text_map = {item["article_id"]: item.get("text", "") for item in full_texts}
    articles_df["text"] = articles_df["id"].map(full_text_map).fillna(articles_df["summary"])

    extraction_inputs: List[Dict[str, str]] = articles_df.to_dict(orient="records")
    extraction_results = extract_batch(extraction_inputs)

    events: List[Dict[str, object]] = []
    for result in extraction_results:
        for event in result.get("events", []):
            payload = {"article_id": result["article_id"], "ticker": result["ticker"], **event}
            events.append(payload)
    events_path = OUT_DIR / "events.csv"
    events_df = score_events(articles_df, events)
    events_df.to_csv(events_path, index=False)

    article_scores = aggregate_article_scores(articles_df, events_df)
    article_scores_path = OUT_DIR / "article_verdicts.csv"
    article_scores.to_csv(article_scores_path, index=False)

    default_lookback = env_int("AIDC_DEFAULT_LOOKBACK", 14)
    company_scores = aggregate_company_scores(article_scores, events_df, lookback_days=default_lookback)
    signals_path = OUT_DIR / "signals.csv"
    company_scores.to_csv(signals_path, index=False)
    company_verdicts_path = OUT_DIR / "company_verdicts.csv"
    company_scores.to_csv(company_verdicts_path, index=False)

    log_json(
        logger,
        event="pipeline_complete",
        articles=len(articles_df),
        events=len(events_df),
    )
    return {
        "events": events_path,
        "article_verdicts": article_scores_path,
        "signals": signals_path,
        "company_verdicts": company_verdicts_path,
    }


__all__ = ["run_pipeline"]
