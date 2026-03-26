# Yellow-Ball-Bot

A web-scraping and machine learning toolkit for tennis analytics.

## Projects

### 🎾 Tennis Match Outcome Predictor

An end-to-end ML pipeline that:
- Downloads all ATP historical match data (1968–2025) from the Jeff Sackmann dataset
- Engineers temporally-safe features (ELO, H2H, surface win rates, recent form, …)
- Trains and compares Logistic Regression, Random Forest, XGBoost, and LightGBM
- Simulates and evaluates the full **Roland Garros 2025** bracket

➡️ See [`tennis-predictor/README.md`](tennis-predictor/README.md) for full documentation and usage.
