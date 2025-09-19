"""CLI entrypoint for the AIDC signals system."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from aidc_signals.pipeline import run_pipeline
from aidc_signals.verdicts import generate_verdicts
from aidc_signals.utils import OUT_DIR


def cmd_ingest(args: argparse.Namespace) -> None:
    from aidc_signals.ingestion import ingest

    articles = ingest(days=args.days, use_google=args.use_google, use_rss=args.use_rss)
    print(json.dumps({"articles": len(articles), "output": str(OUT_DIR / 'articles.jsonl')}, indent=2))


def cmd_pipeline(args: argparse.Namespace) -> None:
    outputs = run_pipeline(days=args.days, use_google=args.use_google, use_rss=args.use_rss)
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


def cmd_verdicts(args: argparse.Namespace) -> None:
    outputs = generate_verdicts(lookback=args.lookback)
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


def cmd_eventstudy(args: argparse.Namespace) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt

    events_path = OUT_DIR / "events.csv"
    if not events_path.exists():
        raise FileNotFoundError("events.csv not found; run pipeline first")
    events = pd.read_csv(events_path)
    if events.empty:
        print("No events to analyze")
        return
    summary = events.groupby("event_type")["raw_score"].agg(["count", "mean", "sum"]).sort_values("sum", ascending=False)
    print(summary)

    fig, ax = plt.subplots(figsize=(10, 5))
    summary["sum"].plot(kind="bar", ax=ax, title="Signal contribution by event type")
    ax.set_ylabel("Cumulative score")
    fig.tight_layout()
    output_path = OUT_DIR / "eventstudy_signal_contribution.png"
    fig.savefig(output_path)
    print(f"Saved event study chart to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AIDC Signals pipeline controller")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_parser = sub.add_parser("ingest", help="Run ingestion only")
    ingest_parser.add_argument("--days", type=int, default=2)
    ingest_parser.add_argument("--use-google", action="store_true")
    ingest_parser.add_argument("--use-rss", action="store_true")
    ingest_parser.set_defaults(func=cmd_ingest)

    pipeline_parser = sub.add_parser("pipeline", help="Run full pipeline")
    pipeline_parser.add_argument("--days", type=int, default=2)
    pipeline_parser.add_argument("--use-google", action="store_true")
    pipeline_parser.add_argument("--use-rss", action="store_true")
    pipeline_parser.set_defaults(func=cmd_pipeline)

    verdicts_parser = sub.add_parser("verdicts", help="Generate verdict files")
    verdicts_parser.add_argument("--lookback", type=int, default=14)
    verdicts_parser.set_defaults(func=cmd_verdicts)

    eventstudy_parser = sub.add_parser("eventstudy", help="Produce event study analytics")
    eventstudy_parser.set_defaults(func=cmd_eventstudy)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
