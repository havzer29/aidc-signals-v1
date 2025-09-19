"""Verdict generation from article scores."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

from .scorer import aggregate_article_scores, aggregate_company_scores
from .utils import OUT_DIR, load_json, setup_json_logger, log_json


def _load_articles() -> pd.DataFrame:
    path = OUT_DIR / "articles.jsonl"
    rows = load_json(path)
    if not rows:
        return pd.DataFrame(columns=["id", "ticker", "published_at", "tier", "summary"])
    return pd.DataFrame(rows)


def _load_events() -> pd.DataFrame:
    path = OUT_DIR / "events.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "article_id",
                "ticker",
                "event_type",
                "direction",
                "confidence",
                "relevance",
                "sentiment",
                "numeric_value",
                "unit",
                "raw_score",
            ]
        )
    return pd.read_csv(path)


def generate_verdicts(lookback: int = 14) -> Dict[str, Path]:
    logger = setup_json_logger("verdicts", OUT_DIR / "pipeline.log")
    log_json(logger, event="verdicts_start", lookback=lookback)

    articles = _load_articles()
    events = _load_events()

    article_scores = aggregate_article_scores(articles, events)
    article_scores_path = OUT_DIR / "article_verdicts.csv"
    article_scores.to_csv(article_scores_path, index=False)

    company_scores = aggregate_company_scores(article_scores, lookback_days=lookback)
    company_scores_path = OUT_DIR / "company_verdicts.csv"
    company_scores.to_csv(company_scores_path, index=False)

    log_json(
        logger,
        event="verdicts_complete",
        articles=len(article_scores),
        companies=len(company_scores),
        timestamp=datetime.utcnow().isoformat(),
    )
    return {"article_verdicts": article_scores_path, "company_verdicts": company_scores_path}


__all__ = ["generate_verdicts"]
