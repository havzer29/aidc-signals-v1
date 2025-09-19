"""CLI helper to run batch extraction."""
from __future__ import annotations

from pathlib import Path
from typing import List

from .extractor import extract_batch
from .utils import OUT_DIR, load_json, save_jsonl


def run_extract(input_path: Path | None = None) -> List[dict]:
    if input_path is None:
        input_path = OUT_DIR / "articles_full.jsonl"
    rows = load_json(input_path)
    if not rows:
        raise FileNotFoundError(f"No articles found at {input_path}")
    results = extract_batch(rows)
    output_path = OUT_DIR / "events_cache.jsonl"
    save_jsonl(output_path, results)
    return results


__all__ = ["run_extract"]
