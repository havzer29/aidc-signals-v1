"""Interactive ingestion console."""
from __future__ import annotations

import argparse
import json
from datetime import datetime

from aidc_signals.ingestion import ingest
from aidc_signals.utils import OUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live ingestion for AIDC signals")
    parser.add_argument("--days", type=int, default=2, help="Lookback window in days")
    parser.add_argument("--use-google", action="store_true", help="Enable Google News ingestion")
    parser.add_argument("--use-rss", action="store_true", help="Enable RSS ingestion")
    args = parser.parse_args()

    articles = ingest(days=args.days, use_google=args.use_google, use_rss=args.use_rss)
    print(f"Ingestion completed at {datetime.utcnow().isoformat()} UTC")
    print(f"Articles collected: {len(articles)}")
    sample = articles[:5]
    print(json.dumps(sample, indent=2))
    print(f"Articles saved to {OUT_DIR / 'articles.jsonl'}")


if __name__ == "__main__":
    main()
