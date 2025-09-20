"""Scoring utilities converting extracted events into supply/demand signals."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .taxonomy import TAXONOMY
from .utils import domain_quality, env_float, env_int

DEFAULT_CONFIDENCE = env_float("AIDC_DEFAULT_CONFIDENCE", 0.6)
DEFAULT_RELEVANCE = env_float("AIDC_DEFAULT_RELEVANCE", 0.6)
DEFAULT_SENTIMENT = env_float("AIDC_DEFAULT_SENTIMENT", 0.0)
DECAY_TAU = env_float("AIDC_DECAY_TAU_DAYS", 7.0)
SIG_ALPHA = env_float("AIDC_SIG_ALPHA", 1.2)
SCARCITY_BETA = env_float("AIDC_SCARCITY_BETA", 1.0)
EMA_HALFLIFE = env_float("AIDC_SIG_EMA_HALFLIFE", 7.0)
MIN_EVENTS = env_int("AIDC_GATE_MIN_EVENTS", 3)
MIN_DOMAINS = env_int("AIDC_GATE_MIN_DOMAINS", 2)
STRONG_SIG = env_float("AIDC_GATE_STRONG_SIG", 0.4)
UP_THRESHOLD = env_float("AIDC_VERDICT_UP", 0.25)
DOWN_THRESHOLD = env_float("AIDC_VERDICT_DOWN", -0.25)
STRONG_THRESHOLD = env_float("AIDC_VERDICT_STRONG", 0.5)
EXTREME_THRESHOLD = env_float("AIDC_VERDICT_EXTREME", 0.75)


@dataclass
class EventScore:
    """Container for scored events."""

    article_id: str
    ticker: str
    event_type: str
    role: str
    direction: int
    confidence: float
    relevance: float
    sentiment: float
    sentiment_factor: float
    domain: str
    domain_quality: float
    decay: float
    age_days: float
    magnitude: float
    supply_effect: float
    demand_effect: float
    scarcity_component: float
    numeric_value: Optional[float]
    unit: Optional[str]
    published_at: datetime

    def to_dict(self) -> Dict[str, object]:
        return {
            "article_id": self.article_id,
            "ticker": self.ticker,
            "event_type": self.event_type,
            "role": self.role,
            "direction": self.direction,
            "confidence": self.confidence,
            "relevance": self.relevance,
            "sentiment": self.sentiment,
            "sentiment_factor": self.sentiment_factor,
            "domain": self.domain,
            "domain_quality": self.domain_quality,
            "decay": self.decay,
            "age_days": self.age_days,
            "magnitude": self.magnitude,
            "supply_effect": self.supply_effect,
            "demand_effect": self.demand_effect,
            "scarcity_component": self.scarcity_component,
            "numeric_value": self.numeric_value,
            "unit": self.unit,
            "published_at": self.published_at.isoformat(),
        }


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _compute_decay(age_days: float) -> float:
    if DECAY_TAU <= 0:
        return 1.0
    return math.exp(-age_days / DECAY_TAU)


def _prepare_articles_frame(articles: pd.DataFrame) -> pd.DataFrame:
    frame = articles.copy()
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
    frame = frame.set_index("id")
    return frame


def score_events(articles: pd.DataFrame, events: List[Dict[str, object]], now: Optional[datetime] = None) -> pd.DataFrame:
    """Score raw events with supply/demand metadata."""

    if now is None:
        now = datetime.now(timezone.utc)

    article_lookup = _prepare_articles_frame(articles)
    scored: List[EventScore] = []

    for event in events:
        article_id = event.get("article_id")
        event_type = event.get("event_type")
        if not article_id or event_type not in TAXONOMY:
            continue
        if article_id not in article_lookup.index:
            continue
        taxonomy = TAXONOMY[event_type]
        direction = int(event.get("direction", 0))
        if direction not in (-1, 1):
            continue
        row = article_lookup.loc[article_id]
        ticker = row.get("ticker")
        if not isinstance(ticker, str):
            continue
        confidence = _clip(float(event.get("confidence", DEFAULT_CONFIDENCE)), 0.0, 1.0)
        relevance = _clip(float(event.get("relevance", DEFAULT_RELEVANCE)), 0.0, 1.0)
        sentiment = _clip(float(event.get("sentiment", DEFAULT_SENTIMENT)), -1.0, 1.0)
        sentiment_factor = 0.5 + 0.5 * sentiment
        magnitude = confidence * relevance * sentiment_factor
        domain = str(row.get("source", ""))
        quality = domain_quality(domain)
        magnitude *= quality
        published_at = row.get("published_at")
        if isinstance(published_at, pd.Timestamp):
            published_dt = published_at.to_pydatetime()
        elif isinstance(published_at, datetime):
            published_dt = published_at
        else:
            published_dt = now
        if published_dt.tzinfo is None:
            published_dt = published_dt.replace(tzinfo=timezone.utc)
        age_days = max((now - published_dt).total_seconds() / 86400.0, 0.0)
        decay = _compute_decay(age_days)
        magnitude *= decay

        impact_direction = taxonomy.impact * direction
        base_weight = taxonomy.weight
        if taxonomy.role.upper() == "S":
            supply_effect = base_weight * impact_direction * magnitude
            demand_effect = 0.0
            scarcity_component = -supply_effect
        else:
            demand_effect = base_weight * impact_direction * magnitude
            supply_effect = 0.0
            scarcity_component = demand_effect

        scored.append(
            EventScore(
                article_id=str(article_id),
                ticker=ticker,
                event_type=event_type,
                role=taxonomy.role,
                direction=direction,
                confidence=confidence,
                relevance=relevance,
                sentiment=sentiment,
                sentiment_factor=sentiment_factor,
                domain=domain,
                domain_quality=quality,
                decay=decay,
                age_days=age_days,
                magnitude=magnitude,
                supply_effect=supply_effect,
                demand_effect=demand_effect,
                scarcity_component=scarcity_component,
                numeric_value=event.get("numeric_value"),
                unit=event.get("unit"),
                published_at=published_dt,
            )
        )

    if not scored:
        return pd.DataFrame(
            columns=[
                "article_id",
                "ticker",
                "event_type",
                "role",
                "direction",
                "confidence",
                "relevance",
                "sentiment",
                "sentiment_factor",
                "domain",
                "domain_quality",
                "decay",
                "age_days",
                "magnitude",
                "supply_effect",
                "demand_effect",
                "scarcity_component",
                "numeric_value",
                "unit",
                "published_at",
            ]
        )

    df = pd.DataFrame([event.to_dict() for event in scored])
    return df


def _sig_to_verdict(sig: float) -> Tuple[str, Optional[int]]:
    if sig >= EXTREME_THRESHOLD:
        return "UP", 30
    if sig >= STRONG_THRESHOLD:
        return "UP", 20
    if sig >= UP_THRESHOLD:
        return "UP", 10
    if sig <= -EXTREME_THRESHOLD:
        return "DOWN", 30
    if sig <= -STRONG_THRESHOLD:
        return "DOWN", 20
    if sig <= DOWN_THRESHOLD:
        return "DOWN", 10
    return "NEUTRAL", None


def gate_signal(breadth: int, domain_diversity: int, sig_value: float) -> bool:
    return (breadth >= MIN_EVENTS and domain_diversity >= MIN_DOMAINS) or (
        abs(sig_value) >= STRONG_SIG and domain_diversity >= MIN_DOMAINS
    )


def aggregate_article_scores(articles: pd.DataFrame, event_scores: pd.DataFrame) -> pd.DataFrame:
    if event_scores.empty:
        return pd.DataFrame(
            columns=[
                "article_id",
                "ticker",
                "published_at",
                "supply_total",
                "demand_total",
                "scarcity",
                "sig",
                "breadth",
                "domain_diversity",
                "verdict",
                "hold_days",
            ]
        )

    articles_frame = _prepare_articles_frame(articles)
    grouped = event_scores.groupby(["article_id", "ticker"], dropna=False)
    supply = grouped["supply_effect"].sum().rename("supply_total")
    demand = grouped["demand_effect"].sum().rename("demand_total")
    breadth = grouped.size().rename("breadth")
    domains = event_scores.groupby("article_id")["domain"].nunique().rename("domain_diversity")

    summary = pd.concat([supply, demand, breadth], axis=1).reset_index()
    summary = summary.merge(domains.reset_index(), on="article_id", how="left")
    summary["domain_diversity"] = summary["domain_diversity"].fillna(1).astype(int)

    def fetch_meta(article_id: str, column: str):
        if article_id in articles_frame.index:
            value = articles_frame.loc[article_id].get(column)
            if isinstance(value, pd.Series):
                return value.iloc[0]
            return value
        return None

    summary["published_at"] = summary["article_id"].apply(
        lambda x: fetch_meta(x, "published_at") or datetime.now(timezone.utc)
    )
    summary["title"] = summary["article_id"].apply(lambda x: fetch_meta(x, "title"))
    summary["url"] = summary["article_id"].apply(lambda x: fetch_meta(x, "url"))
    summary["source"] = summary["article_id"].apply(lambda x: fetch_meta(x, "source"))
    summary["language"] = summary["article_id"].apply(lambda x: fetch_meta(x, "language"))
    summary["company"] = summary["article_id"].apply(lambda x: fetch_meta(x, "company"))
    summary["scarcity"] = summary.apply(
        lambda row: row["demand_total"] - SCARCITY_BETA * row["supply_total"], axis=1
    )
    summary["sig"] = summary["scarcity"].apply(lambda x: math.tanh(SIG_ALPHA * x))
    verdicts: List[str] = []
    holds: List[Optional[int]] = []
    for sig in summary["sig"]:
        verdict, hold = _sig_to_verdict(sig)
        verdicts.append(verdict)
        holds.append(hold)
    summary["verdict"] = verdicts
    summary["hold_days"] = holds
    summary = summary.sort_values("published_at", ascending=False)
    return summary


def aggregate_company_scores(
    article_scores: pd.DataFrame,
    event_scores: pd.DataFrame,
    lookback_days: int = 14,
) -> pd.DataFrame:
    if article_scores.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "demand_total",
                "supply_total",
                "scarcity",
                "ema_scarcity",
                "sig",
                "events",
                "domains",
                "gate_passed",
                "verdict",
                "hold_days",
                "as_of",
            ]
        )

    article_scores = article_scores.copy()
    article_scores["published_at"] = pd.to_datetime(article_scores["published_at"], utc=True, errors="coerce")
    cutoff = article_scores["published_at"].max() - pd.Timedelta(days=lookback_days)
    recent_articles = article_scores[article_scores["published_at"] >= cutoff]
    if recent_articles.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "demand_total",
                "supply_total",
                "scarcity",
                "ema_scarcity",
                "sig",
                "events",
                "domains",
                "gate_passed",
                "verdict",
                "hold_days",
                "as_of",
            ]
        )

    event_scores = event_scores.copy()
    event_scores["published_at"] = pd.to_datetime(event_scores["published_at"], utc=True, errors="coerce")
    recent_events = event_scores[event_scores["published_at"] >= cutoff]

    records: List[Dict[str, object]] = []
    as_of = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

    for ticker, frame in recent_articles.groupby("ticker"):
        frame = frame.sort_values("published_at")
        demand_total = frame["demand_total"].sum()
        supply_total = frame["supply_total"].sum()
        scarcity_series = frame.apply(
            lambda row: row["demand_total"] - SCARCITY_BETA * row["supply_total"], axis=1
        )
        if EMA_HALFLIFE > 0 and len(scarcity_series) > 1:
            ema_scarcity = float(
                scarcity_series.ewm(halflife=EMA_HALFLIFE, adjust=False).mean().iloc[-1]
            )
        else:
            ema_scarcity = float(scarcity_series.iloc[-1])
        sig_value = math.tanh(SIG_ALPHA * ema_scarcity)
        events_count = int(frame["breadth"].sum())
        domains = int(recent_events[recent_events["ticker"] == ticker]["domain"].nunique())
        gate_passed = gate_signal(events_count, max(domains, 1), sig_value)
        verdict, hold = _sig_to_verdict(sig_value) if gate_passed else ("NEUTRAL", None)
        records.append(
            {
                "ticker": ticker,
                "demand_total": float(demand_total),
                "supply_total": float(supply_total),
                "scarcity": float(scarcity_series.iloc[-1]),
                "ema_scarcity": ema_scarcity,
                "sig": sig_value,
                "events": events_count,
                "domains": max(domains, 1),
                "gate_passed": gate_passed,
                "verdict": verdict,
                "hold_days": hold,
                "as_of": as_of,
            }
        )

    return pd.DataFrame(records).sort_values("sig", ascending=False)


__all__ = [
    "score_events",
    "aggregate_article_scores",
    "aggregate_company_scores",
    "gate_signal",
]
