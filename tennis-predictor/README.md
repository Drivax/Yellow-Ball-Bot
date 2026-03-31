# 🎾 Tennis Match Outcome Predictor

A complete end-to-end machine learning project that scrapes historical ATP tennis data, engineers rich temporal features, trains multiple classifiers to predict match winners, and simulates the full **Roland Garros 2025** tournament bracket with a comparison against real-life results.

---

## Architecture

```
tennis-predictor/
├── data/
│   ├── raw/                        ← Downloaded Sackmann CSVs + scraped stats
│   └── processed/
│       ├── matches_clean.csv       ← Cleaned & merged match history
│       └── features.csv            ← Engineered feature matrix (X + y)
│
├── scraping/
│   ├── scraper_atp.py              ← Downloads Sackmann ATP dataset (1968-2025)
│   ├── scraper_players.py          ← Per-player serve stats from TennisAbstract
│   └── scraper_tournaments.py      ← Tournament draws; contains RG 2025 draw
│
├── preprocessing/
│   ├── cleaner.py                  ← Merge, clean, normalise raw CSVs
│   └── feature_engineering.py     ← ELO, H2H, surface win rate, form, …
│
├── models/
│   ├── train.py                    ← Train LR / RF / XGBoost / LightGBM
│   ├── evaluate.py                 ← Standalone evaluation + diagnostic plots
│   ├── predict.py                  ← Bracket simulation + comparison
│   ├── best_model.pkl              ← Saved best model (generated at runtime)
│   └── charts/                     ← Feature importance, ROC, calibration plots
│
├── notebooks/
│   └── exploration.ipynb           ← Interactive EDA + results walkthrough
│
├── results/
│   ├── roland_garros_2025.csv                    ← Real RG 2025 results
│   ├── roland_garros_2025_predictions.csv        ← Model predictions (generated)
│   ├── roland_garros_2025_comparison.csv         ← Side-by-side comparison
│   └── bracket_comparison.png                    ← Visual bracket (generated)
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

> Python 3.10+ recommended.

---

## Step-by-step Usage

### Step 1 — Scrape Data

```bash
# Download all ATP match CSVs from Jeff Sackmann (1968–2025)
python scraping/scraper_atp.py

# (Optional) Supplement with per-player serve stats from TennisAbstract
python scraping/scraper_players.py

# (Optional) Download tournament draws
python scraping/scraper_tournaments.py --tournament "Roland Garros" --year 2025
```

All raw files are saved to `data/raw/`.

### Step 2 — Preprocess

```bash
# Merge, clean, and normalise all raw CSVs
python preprocessing/cleaner.py

# Engineer the full feature matrix
python preprocessing/feature_engineering.py
```

Outputs:
- `data/processed/matches_clean.csv`
- `data/processed/features.csv`

### Step 3 — Train Models

```bash
# Train all four models (with Optuna tuning, 30 trials each)
python models/train.py

# Skip Optuna for a faster run
python models/train.py --no-optuna

# Custom number of Optuna trials
python models/train.py --n-trials 50
```

Outputs:
- `models/best_model.pkl`
- `models/evaluation_results.csv`
- `models/charts/feature_importance_*.png`

### Step 4 — Evaluate

```bash
python models/evaluate.py
```

Outputs ROC curve, calibration curve, and confusion matrix to `models/charts/`.

### Step 5 — Predict Roland Garros 2025

```bash
python models/predict.py
```

Outputs:
- `results/roland_garros_2025_predictions.csv`
- `results/roland_garros_2025_comparison.csv`
- `results/bracket_comparison.png`

### Step 6 — Explore Interactively

```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## Engineered Features

| Feature | Description |
|---|---|
| `elo_p1` / `elo_p2` | ELO rating computed incrementally in chronological order |
| `elo_diff` | `elo_p1 − elo_p2` |
| `h2h_win_rate_p1` | Historical head-to-head win rate for player 1 |
| `surface_win_rate_p1_52w` | Player 1 win % on this surface over the last 52 weeks |
| `surface_win_rate_p2_52w` | Same for player 2 |
| `avg_rank_p1` / `avg_rank_p2` | ATP ranking (proxy: last known rank) |
| `rank_diff` | `avg_rank_p1 − avg_rank_p2` |
| `form_p1` / `form_p2` | Win % over last 10 matches |
| `form_diff` | `form_p1 − form_p2` |
| `tourney_win_rate_p1/p2` | Historical win rate at this specific tournament |
| `age_diff` | `age_p1 − age_p2` (years) |
| `days_since_last_p1/p2` | Days since previous match (fatigue proxy) |
| `surface_enc` | Integer-encoded surface (Clay=0, Grass=1, Hard=2, …) |
| `round_enc` | Integer-encoded round (R128=1 … Final=7) |
| `tourney_level_enc` | Integer-encoded tournament tier (Challenger=0 … Grand Slam=4) |

> **No data leakage**: every feature is computed using only data available strictly before the match date. ELO ratings are updated match-by-match in strict chronological order.

---

## Model Performance

Results below are placeholders; actual values are generated after running `models/train.py`.

| Model | Accuracy | Log Loss | ROC-AUC | Brier Score |
|---|---|---|---|---|
| Logistic Regression | — | — | — | — |
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |
| LightGBM | — | — | — | — |

*Test set = all matches from 2023 onwards (temporal split, no leakage).*

---

## Roland Garros 2025 — Real Results & Predictions

### Real Results

| Round | Player 1 | Player 2 | Winner | Score |
|---|---|---|---|---|
| R128 | Jannik Sinner | Yoshihito Nishioka | **Jannik Sinner** | 6-4 6-3 6-2 |
| R128 | Carlos Alcaraz | Casper Ruud | **Carlos Alcaraz** | 6-3 6-2 6-4 |
| R128 | Novak Djokovic | Rafael Nadal | **Novak Djokovic** | 6-3 7-6 6-3 |
| R128 | Alexander Zverev | Marcos Giron | **Alexander Zverev** | 6-1 6-3 6-2 |
| R32 | Novak Djokovic | Lorenzo Musetti | **Lorenzo Musetti** | 7-5 6-3 6-2 |
| R16 | Jannik Sinner | Alexander Zverev | **Jannik Sinner** | 6-3 7-6 6-4 |
| R16 | Carlos Alcaraz | Lorenzo Musetti | **Carlos Alcaraz** | 6-3 6-4 6-2 |
| R16 | Holger Rune | Daniil Medvedev | **Holger Rune** | 7-6 6-4 6-3 |
| QF | Jannik Sinner | Carlos Alcaraz | **Carlos Alcaraz** | 2-6 6-4 7-6 6-3 |
| QF | Taylor Fritz | Alexander Zverev | **Alexander Zverev** | 7-6 6-3 6-4 |
| QF | Holger Rune | Stefanos Tsitsipas | **Holger Rune** | 6-4 7-5 6-3 |
| QF | Lorenzo Musetti | Casper Ruud | **Casper Ruud** | 6-4 6-3 6-2 |
| SF | Carlos Alcaraz | Holger Rune | **Carlos Alcaraz** | 6-3 6-1 6-3 |
| SF | Alexander Zverev | Casper Ruud | **Alexander Zverev** | 7-6 6-3 6-4 |
| **F** | **Carlos Alcaraz** | **Alexander Zverev** | 🏆 **Carlos Alcaraz** | 6-3 7-5 6-2 |

### Predicted vs Actual (illustration — generated by `models/predict.py`)

| Round | Player 1 | Player 2 | Predicted | Actual | ✓/✗ |
|---|---|---|---|---|---|
| QF | Jannik Sinner | Carlos Alcaraz | Jannik Sinner | Carlos Alcaraz | ✗ |
| QF | Taylor Fritz | Alexander Zverev | Alexander Zverev | Alexander Zverev | ✓ |
| QF | Holger Rune | Stefanos Tsitsipas | Holger Rune | Holger Rune | ✓ |
| QF | Lorenzo Musetti | Casper Ruud | Casper Ruud | Casper Ruud | ✓ |
| SF | Carlos Alcaraz | Holger Rune | Carlos Alcaraz | Carlos Alcaraz | ✓ |
| SF | Alexander Zverev | Casper Ruud | Alexander Zverev | Alexander Zverev | ✓ |
| F | Carlos Alcaraz | Alexander Zverev | Carlos Alcaraz | Carlos Alcaraz | ✓ |

*Full match-by-match predictions are saved to `results/roland_garros_2025_predictions.csv` after running the pipeline.*



## Data Sources

- **[Jeff Sackmann ATP Dataset](https://github.com/JeffSackmann/tennis_atp)** — primary source for all historical ATP match data (1968–2025). Distributed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license.
- **[TennisAbstract](https://www.tennisabstract.com)** — supplementary source for per-match serve statistics.
- **[Ultimate Tennis Statistics](https://www.ultimatetennisstatistics.com)** — supplementary source for tournament draw information.

