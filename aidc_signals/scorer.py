"""Scoring utilities converting events into signals."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import pandas as pd

from .taxonomy import TAXONOMY

TIER_WEIGHTS = {"tier1": 1.0, "tier2": 0.8, "tier3": 0.6}


@dataclass
class EventScore:
    article_id: str
    ticker: str
    event_type: str
    direction: int
    confidence: float
    relevance: float
    sentiment: float
    numeric_value: float | None
    unit: str | None
    raw_score: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "article_id": self.article_id,
            "ticker": self.ticker,
            "event_type": self.event_type,
            "direction": self.direction,
            "confidence": self.confidence,
            "relevance": self.relevance,
            "sentiment": self.sentiment,
            "numeric_value": self.numeric_value,
            "unit": self.unit,
            "raw_score": self.raw_score,
        }


def _tier_weight(tier: str) -> float:
    return TIER_WEIGHTS.get(tier.lower(), 0.5)


def score_events(articles: pd.DataFrame, events: List[Dict[str, object]]) -> pd.DataFrame:
    article_lookup = articles.set_index("id")
    scored: List[EventScore] = []
    for event in events:
        article_id = event["article_id"]
        if article_id not in article_lookup.index:
            continue
        article_row = article_lookup.loc[article_id]
        tier = article_row.get("tier", "tier3")
        taxonomy = TAXONOMY.get(event["event_type"])
        if not taxonomy:
            continue
        base = taxonomy.weight * event["direction"]
        confidence = float(event.get("confidence", 0))
        relevance = float(event.get("relevance", 0))
        sentiment = float(event.get("sentiment", 0))
        score = base * confidence * relevance * (1 + 0.5 * sentiment)
        score *= _tier_weight(tier)
        scored.append(
            EventScore(
                article_id=article_id,
                ticker=article_row["ticker"],
                event_type=event["event_type"],
                direction=event["direction"],
                confidence=confidence,
                relevance=relevance,
                sentiment=sentiment,
                numeric_value=event.get("numeric_value"),
                unit=event.get("unit"),
                raw_score=score,
            )
        )
    df = pd.DataFrame([item.to_dict() for item in scored])
    if df.empty:
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
    return df


def aggregate_article_scores(articles: pd.DataFrame, event_scores: pd.DataFrame) -> pd.DataFrame:
    if event_scores.empty:
        return pd.DataFrame(columns=["article_id", "ticker", "score", "published_at", "verdict"])
    agg = event_scores.groupby("article_id")["raw_score"].sum().rename("score")
    article_index = articles.set_index("id")
    merged = agg.to_frame().join(article_index[["ticker", "published_at"]], how="left")
    merged["published_at"] = pd.to_datetime(merged["published_at"])
    merged["verdict"] = merged["score"].apply(verdict_from_score)
    merged = merged.reset_index().rename(columns={"index": "article_id"})
    return merged


def verdict_from_score(score: float, up_threshold: float = 0.15, down_threshold: float = -0.15) -> str:
    if score >= up_threshold:
        return "UP"
    if score <= down_threshold:
        return "DOWN"
    return "NEUTRAL"


def aggregate_company_scores(article_scores: pd.DataFrame, lookback_days: int = 14) -> pd.DataFrame:
    if article_scores.empty:
        return pd.DataFrame(columns=["ticker", "score", "verdict", "as_of"])
    article_scores["published_at"] = pd.to_datetime(article_scores["published_at"])
    cutoff = article_scores["published_at"].max() - pd.Timedelta(days=lookback_days)
    recent = article_scores[article_scores["published_at"] >= cutoff]
    company_scores = recent.groupby("ticker")["score"].mean().rename("score").reset_index()
    company_scores["verdict"] = company_scores["score"].apply(verdict_from_score)
    company_scores["as_of"] = datetime.utcnow()
    return company_scores


__all__ = [
    "score_events",
    "aggregate_article_scores",
    "aggregate_company_scores",
    "verdict_from_score",
]
