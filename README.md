# AIDC Signals

AIDC Signals construit des signaux long/short sur les actions IA & data centers en agrégeant des flux d'actualité temps réel,
extractions LLM structurées et un moteur de scoring Offre/Demande.

## 🎯 Fonctionnalités clés
- **Ingestion multi-sources** : Google News (EN/FR), flux IR/PR (configurables) avec déduplication par URL finale.
- **Extraction structurée** : appels `openai.ChatCompletion` (JSON schema strict) + cache `cache/events_cache.jsonl` pour éviter les relances.
- **Scoring Offre/Demande** : pondération par confiance, pertinence, sentiment, qualité de la source, décroissance temporelle et taxonomie explicite.
- **Agrégation Scarcity/SIG** : calcul `Scarcity = D_total - β * S_total`, normalisation `tanh(α × Scarcity)`, gating (≥ événements, ≥ sources).
- **Exports analytiques** : `events.csv`, `article_verdicts.csv`, `signals.csv`, `company_verdicts.csv`, graphiques event study.
- **UI Streamlit** : tableau de bord interactif avec filtres, détails par ticker et ventilation des événements.
- **Logs JSON & observabilité** : `out/ingest.log`, `out/pipeline.log` + caches `out/articles_full.jsonl`, `cache/events_cache.jsonl`.

## 🚀 Installation rapide
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python run_all.py pipeline --days 2 --use-google --use-rss
python run_all.py verdicts --lookback 14
streamlit run ui_verdicts.py
```

## ⚙️ Configuration
Paramétrez le comportement via des variables d'environnement :

| Variable | Rôle | Défaut |
| --- | --- | --- |
| `OPENAI_API_KEY` | Clé OpenAI (obligatoire) | – |
| `OPENAI_MODEL` | Modèle chat completions | `gpt-4o-mini` |
| `AIDC_GOOGLE_LANGS` | Langues Google News (`hl` pairs) | `en-US,fr-FR` |
| `AIDC_FETCH_CONCURRENCY` | Concurrence httpx pour le full-text | `6` |
| `AIDC_DECAY_TAU_DAYS` | Tau de décroissance temporelle | `7` |
| `AIDC_SIG_ALPHA` | Gain de la tanh du SIG | `1.2` |
| `AIDC_SCARCITY_BETA` | Pondération Offre vs Demande | `1.0` |
| `AIDC_GATE_MIN_EVENTS` | Événements minimum pour lever le gate | `3` |
| `AIDC_GATE_MIN_DOMAINS` | Sources distinctes minimum | `2` |
| `AIDC_GATE_STRONG_SIG` | Seuil SIG fort pour override gate | `0.4` |
| `AIDC_DEFAULT_LOOKBACK` | Lookback (jours) par défaut du pipeline | `14` |
| `AIDC_TIER{1,2,3}_WEIGHT` | Poids qualité des domaines par tier | `1.2 / 1.0 / 0.8` |

Les listes de sociétés (67 tickers IA/DC), flux RSS et tiers de domaines sont dans `config/`.

## 🧰 Utilisation
### CLI
```bash
python run_all.py ingest --days 1 --use-google --use-rss
python run_all.py pipeline --days 2 --use-google --use-rss
python run_all.py verdicts --lookback 14
python run_all.py eventstudy
```

### Tableau de bord Streamlit
```bash
streamlit run ui_verdicts.py
```
Fonctionnalités : filtres verdict/événements/sources, vue ticker détaillée (articles, contributions Offre/Demande, liens sources), histogramme des types d'événements.

## 📂 Fichiers générés (`out/`)
- `articles.jsonl` / `articles_full.jsonl` : ingestion normalisée + full-text extrait.
- `events.csv` : événements scorés (direction, confiance, sentiment, supply/demand effects, decay, domaine...).
- `article_verdicts.csv` : scores par article (Scarcity, SIG, offre/demande agrégés, métadonnées).
- `signals.csv` : signaux agrégés par ticker (D_total, S_total, Scarcity, SIG, gating, verdict, durée).
- `company_verdicts.csv` : copie des signaux agrégés (override possible via `run_all.py verdicts`).
- `eventstudy_signal_contribution.png` : graphique contributions par type d'événement.

Caches & logs :
- `cache/events_cache.jsonl` : mémorise les sorties OpenAI par `article_id`.
- `out/ingest.log` & `out/pipeline.log` : logs JSON pour ingestion / pipeline.

## 🔍 Taxonomie & scoring
- Taxonomie centralisée dans `aidc_signals/taxonomy.py` (rôle S/D, impact directionnel, poids relatif).
- Score événement = `weight × impact × confidence × relevance × sentiment_factor × domain_quality × decay`.
- `scarcity_component` additionné côté Demande, retranché côté Offre.
- Gating : `(événements ≥ MIN_EVENTS && sources ≥ MIN_DOMAINS)` ou `|SIG| ≥ AIDC_GATE_STRONG_SIG` avec ≥ MIN_DOMAINS.

## 🧪 Event study & visualisations
```bash
python run_all.py eventstudy
```
Produit un résumé `count/mean/sum` par `event_type` (colonne `scarcity_component`) et une barre de contribution cumulée.

## 🛠️ Dépannage
- **429 OpenAI** : réduire `AIDC_OPENAI_CONCURRENCY`, augmenter le backoff (`AIDC_OPENAI_MIN_DELAY_MS`, `AIDC_OPENAI_BACKOFF_MS`).
- **Timeout HTTP** : ajuster `AIDC_FETCH_TIMEOUT`, vérifier la connectivité, la disponibilité des flux RSS.
- **Peu d'événements** : étendre `--days`, enrichir `config/rss_feeds.txt`, vérifier les alias dans `config/companies.csv`.
- **UI vide** : relancer `python run_all.py pipeline` puis `python run_all.py verdicts`; vérifier les chemins `out/*.csv`.

## ⚠️ Disclaimer
Ce projet est fourni **à titre informatif uniquement**. Les signaux générés ne constituent **pas** un conseil financier. Vérifiez toujours les sources originales et respectez les conditions d'utilisation des fournisseurs d'actualité.
