# Yellow-Ball-Bot

A web-scraping and machine learning toolkit for tennis analytics.

## Current US Open and upcoming predictions

From `tennis-predictor`, run `python us_open_predict.py --no-model` or
`streamlit run streamlit_app.py` and click **Actualiser les prochains matchs**.
The app fetches confirmed ATP/WTA singles from ESPN with match IDs, UTC dates,
status, source URL and retrieval time. Unknown opponents, qualifying, doubles,
finished and in-progress matches are excluded. An unpublished match is never
replaced by a hypothetical draw. A provider failure is an error; a successful
empty schedule is reported explicitly. `--year` selects an edition, not a replay
of that edition's past matches.

Predictions include both players' probabilities and an explicit method/status.
The fallback updates the project's legacy manual Elo estimates with completed
main-draw matches of the requested edition (K=32; unseen players start at 1500
when first observed in a result). These are **uncalibrated estimates with subjective
initial ratings**, not validated current rankings. Each player's tournament sample
count is exported. Players without a rating or results receive no probability. The optional trained
model uses ATP history only; it is never applied to WTA. Scheduled times marked
`time_confirmed=false` indicate a known date with an unannounced time.
Refresh before use: schedules change, and downloaded reports are snapshots.

ESPN's public scoreboard is an external, unversioned source; changes to its schema
may require adapter updates. No API key is required. Tests run with
`python -m unittest discover -s tests -v`. The US Open Actions workflow generates
downloadable reports every six hours during August/September, and on manual runs.
Source failures fail the run instead of publishing invented matches.

## Projects

### 🎾 Tennis Match Outcome Predictor

An end-to-end ML pipeline that:
- Downloads all ATP historical match data (1968–2025) from the Jeff Sackmann dataset
- Engineers temporally-safe features (ELO, H2H, surface win rates, recent form, …)
- Trains and compares Logistic Regression, Random Forest, XGBoost, and LightGBM
- Simulates and evaluates the full **Roland Garros 2025** bracket

➡️ See [`tennis-predictor/README.md`](tennis-predictor/README.md) for full documentation and usage.
