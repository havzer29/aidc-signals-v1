"""Streamlit UI for exploring AIDC verdicts and underlying events."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from aidc_signals.utils import OUT_DIR

st.set_page_config(page_title="AIDC Signals Dashboard", layout="wide")
st.title("📡 AIDC Signals — Verdict Explorer")

article_path = OUT_DIR / "article_verdicts.csv"
company_path = OUT_DIR / "company_verdicts.csv"
events_path = OUT_DIR / "events.csv"

missing_paths = [path for path in (article_path, company_path, events_path) if not path.exists()]
if missing_paths:
    st.warning(
        "Les fichiers suivants sont manquants. Veuillez exécuter `python run_all.py pipeline` puis `python run_all.py verdicts` :\n"
        + "\n".join(f"- {path}" for path in missing_paths)
    )
    st.stop()

articles = pd.read_csv(article_path, parse_dates=["published_at"], infer_datetime_format=True)
companies = pd.read_csv(company_path)
events = pd.read_csv(events_path, parse_dates=["published_at"], infer_datetime_format=True)

companies["as_of"] = pd.to_datetime(companies["as_of"], errors="coerce", utc=True)
companies["last_updated"] = companies["as_of"]

st.sidebar.header("Filtres")
verdict_options = ["UP", "DOWN", "NEUTRAL"]
selected_verdicts = st.sidebar.multiselect(
    "Verdicts", verdict_options, default=verdict_options
)
min_events = st.sidebar.slider("Nombre minimum d'événements", min_value=0, max_value=15, value=0)
min_domains = st.sidebar.slider("Sources uniques minimum", min_value=1, max_value=5, value=1)
selected_ticker = st.sidebar.selectbox("Ticker", options=["ALL", *sorted(companies["ticker"].unique())])
show_only_gated = st.sidebar.checkbox("Afficher uniquement les signaux validés (gate)", value=False)

filtered = companies.copy()
filtered = filtered[filtered["verdict"].isin(selected_verdicts)]
filtered = filtered[filtered["events"] >= min_events]
filtered = filtered[filtered["domains"] >= min_domains]
if show_only_gated:
    filtered = filtered[filtered["gate_passed"]]
if selected_ticker != "ALL":
    filtered = filtered[filtered["ticker"] == selected_ticker]

st.subheader("Synthèse des verdicts")
col1, col2, col3 = st.columns(3)
col1.metric("📈 Ups", int((filtered["verdict"] == "UP").sum()))
col2.metric("📉 Downs", int((filtered["verdict"] == "DOWN").sum()))
col3.metric("➖ Neutres", int((filtered["verdict"] == "NEUTRAL").sum()))

summary_columns = [
    "ticker",
    "verdict",
    "sig",
    "events",
    "domains",
    "gate_passed",
    "hold_days",
    "demand_total",
    "supply_total",
]
display_df = filtered[summary_columns].copy()
display_df["sig"] = display_df["sig"].round(3)
display_df["demand_total"] = display_df["demand_total"].round(3)
display_df["supply_total"] = display_df["supply_total"].round(3)
display_df.rename(
    columns={
        "sig": "SIG",
        "events": "#Events",
        "domains": "#Sources",
        "gate_passed": "Gate",
        "hold_days": "Hold (j)",
        "demand_total": "Demande",
        "supply_total": "Offre",
    },
    inplace=True,
)
st.dataframe(display_df, use_container_width=True)

st.caption(
    "Les signaux sont bornés via tanh(SIG_ALPHA × Scarcity). Les durées de hold sont indicatives et ne constituent pas un conseil financier."
)

st.subheader("Répartition des événements (global)")
if not events.empty:
    event_totals = (
        events.groupby("event_type")["scarcity_component"].sum().sort_values(ascending=False)
    )
    st.bar_chart(event_totals)
else:
    st.info("Aucun événement disponible pour le moment.")

if selected_ticker != "ALL":
    st.subheader(f"Détails pour {selected_ticker}")
    ticker_articles = (
        articles[articles["ticker"] == selected_ticker]
        .copy()
        .sort_values("published_at", ascending=False)
    )
    ticker_articles["published_at"] = pd.to_datetime(ticker_articles["published_at"], utc=True)
    article_view_cols = [
        "published_at",
        "title",
        "scarcity",
        "sig",
        "breadth",
        "domain_diversity",
        "supply_total",
        "demand_total",
        "verdict",
    ]
    st.dataframe(ticker_articles[article_view_cols], use_container_width=True)

    ticker_events = events[events["ticker"] == selected_ticker].copy()
    if not ticker_events.empty:
        event_breakdown = (
            ticker_events.groupby("event_type")[["scarcity_component", "supply_effect", "demand_effect"]]
            .sum()
            .sort_values("scarcity_component", ascending=False)
        )
        st.write("Contribution des événements")
        st.dataframe(event_breakdown, use_container_width=True)

        st.write("Événements récents")
        for article_id, subset in ticker_events.groupby("article_id"):
            meta_row = ticker_articles[ticker_articles["article_id"] == article_id]
            if not meta_row.empty:
                title = meta_row.iloc[0]["title"]
                published = meta_row.iloc[0]["published_at"]
                url = meta_row.iloc[0].get("url", "")
                header = f"📰 {title} — {published}" if pd.notnull(published) else f"📰 {title}"
            else:
                header = f"📰 Article {article_id}"
                url = ""
            with st.expander(header, expanded=False):
                if url:
                    st.markdown(f"[Ouvrir l'article]({url})")
                st.dataframe(
                    subset[[
                        "event_type",
                        "role",
                        "direction",
                        "confidence",
                        "relevance",
                        "sentiment",
                        "domain",
                        "scarcity_component",
                    ]],
                    use_container_width=True,
                )
    else:
        st.info("Aucun événement pour ce ticker dans la fenêtre sélectionnée.")

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ Les signaux sont expérimentaux et ne constituent pas une recommandation d'investissement."
)
