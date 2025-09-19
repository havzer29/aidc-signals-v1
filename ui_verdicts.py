"""Streamlit UI for exploring AIDC verdicts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from aidc_signals.utils import OUT_DIR

st.set_page_config(page_title="AIDC Verdicts", layout="wide")
st.title("AIDC Signals Verdict Explorer")

article_path = OUT_DIR / "article_verdicts.csv"
company_path = OUT_DIR / "company_verdicts.csv"

if not article_path.exists() or not company_path.exists():
    st.warning("Run the pipeline and verdict generation before opening the UI.")
    st.stop()

articles = pd.read_csv(article_path)
companies = pd.read_csv(company_path)

st.sidebar.header("Filters")
selected_ticker = st.sidebar.selectbox("Ticker", options=["ALL", *sorted(companies["ticker"].unique())])
lookback = st.sidebar.slider("Lookback days", min_value=3, max_value=30, value=14)

st.header("Company Verdicts")
if selected_ticker != "ALL":
    st.dataframe(companies[companies["ticker"] == selected_ticker])
else:
    st.dataframe(companies)

st.header("Article Scores")
articles["published_at"] = pd.to_datetime(articles["published_at"])
cutoff = articles["published_at"].max() - pd.Timedelta(days=lookback)
filtered_articles = articles[articles["published_at"] >= cutoff]
if selected_ticker != "ALL":
    filtered_articles = filtered_articles[filtered_articles["ticker"] == selected_ticker]
st.dataframe(filtered_articles.sort_values("published_at", ascending=False))

st.header("Score Distribution")
st.bar_chart(filtered_articles.set_index("published_at")["score"])
