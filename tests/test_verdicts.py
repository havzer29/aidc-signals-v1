"""Unit tests for scoring and gating logic."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from aidc_signals.scorer import aggregate_company_scores, gate_signal, score_events


def test_score_events_basic() -> None:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    articles = pd.DataFrame(
        [
            {
                "id": "a1",
                "ticker": "NVDA",
                "company": "NVIDIA",
                "title": "Test article",
                "summary": "",
                "source": "www.reuters.com",
                "published_at": now.isoformat(),
                "language": "en",
            }
        ]
    )
    events = [
        {
            "article_id": "a1",
            "event_type": "demand_up",
            "direction": 1,
            "confidence": 1.0,
            "relevance": 1.0,
            "sentiment": 0.0,
            "numeric_value": None,
            "unit": None,
        }
    ]

    scored = score_events(articles, events, now=now)
    assert not scored.empty
    row = scored.iloc[0]
    # With confidence=relevance=1 and sentiment=0, magnitude should equal domain weight (1.2) * 0.5 = 0.6
    assert abs(row["demand_effect"] - 0.6) < 1e-6
    assert row["supply_effect"] == 0.0
    assert row["role"] == "D"


def test_gate_signal_thresholds() -> None:
    assert gate_signal(3, 2, 0.1) is True
    assert gate_signal(1, 2, 0.5) is True  # strong signal override
    assert gate_signal(2, 1, 0.5) is False


def test_aggregate_company_scores_gating() -> None:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    article_scores = pd.DataFrame(
        [
            {
                "article_id": "a1",
                "ticker": "NVDA",
                "published_at": now.isoformat(),
                "supply_total": -0.2,
                "demand_total": 0.5,
                "scarcity": 0.7,
                "sig": 0.3,
                "breadth": 2,
                "domain_diversity": 2,
            },
            {
                "article_id": "a2",
                "ticker": "NVDA",
                "published_at": now.isoformat(),
                "supply_total": -0.1,
                "demand_total": 0.4,
                "scarcity": 0.5,
                "sig": 0.25,
                "breadth": 2,
                "domain_diversity": 2,
            },
        ]
    )
    event_scores = pd.DataFrame(
        [
            {
                "article_id": "a1",
                "ticker": "NVDA",
                "domain": "www.reuters.com",
                "published_at": now,
            },
            {
                "article_id": "a2",
                "ticker": "NVDA",
                "domain": "www.ft.com",
                "published_at": now,
            },
        ]
    )

    company = aggregate_company_scores(article_scores, event_scores, lookback_days=14)
    assert not company.empty
    row = company.iloc[0]
    assert row["gate_passed"] is True
    assert row["verdict"] in {"UP", "DOWN", "NEUTRAL"}
