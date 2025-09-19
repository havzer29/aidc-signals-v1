"""Story clustering helpers."""
from __future__ import annotations

import pandas as pd


def cluster_stories(articles: pd.DataFrame) -> pd.DataFrame:
    if "story_id" not in articles.columns:
        articles["story_id"] = articles["url"].apply(hash)
    clusters = articles.groupby("story_id").agg(
        {
            "id": "first",
            "ticker": lambda vals: list(set(vals)),
            "title": "first",
            "published_at": "min",
        }
    )
    clusters = clusters.rename(columns={"id": "representative_article"}).reset_index()
    return clusters


__all__ = ["cluster_stories"]
