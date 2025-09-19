# AIDC Signals

AIDC Signals builds long/short signals for AI and data center equities using live news ingestion, structured LLM extraction, and systematic scoring.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python run_all.py pipeline --days 2 --use-google --use-rss
python run_all.py verdicts --lookback 14
streamlit run ui_verdicts.py
```

## Commands

- `python mini_term_ingest.py --days 2 --use-google --use-rss`
- `python run_all.py ingest --days 1 --use-google --use-rss`
- `python run_all.py pipeline --days 2 --use-google --use-rss`
- `python run_all.py verdicts --lookback 14`
- `python run_all.py eventstudy`

## Outputs

Generated artefacts live in the `out/` directory:

- `articles.jsonl` — normalized ingestion records
- `articles_full.jsonl` — enriched full-text content
- `events.csv` — structured events and scores
- `signals.csv` — per-event scores
- `article_verdicts.csv` — article-level verdicts
- `company_verdicts.csv` — aggregate ticker verdicts
- `eventstudy_signal_contribution.png` — signal contribution chart

Logs are written to `out/ingest.log` and `out/pipeline.log` in JSON for streaming observability.
