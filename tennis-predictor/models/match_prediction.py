"""
predict_match.py
----------------
Edit the MATCHES list below, then run:

    python predict_match.py

No arguments needed.
"""

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT  = BASE_DIR.parent
MODEL_PATH    = BASE_DIR / "best_model.pkl"
FEATURES_CSV  = PROJECT_ROOT / "data" / "processed" / "features.csv"

sys.path.insert(0, str(PROJECT_ROOT))
from models.predict import build_match_feature_vector, _build_player_stats_snapshot

# ---------------------------------------------------------------------------
# Match definition helper
# ---------------------------------------------------------------------------

@dataclass
class Match:
    p1:      str
    p2:      str
    surface: str           = "Clay"
    round:   str           = "QF"
    level:   str           = "Grand Slam"
    date:    Optional[str] = None   # "YYYY-MM-DD" or None → today

def match(p1: str, p2: str, surface: str = "Clay", round: str = "QF",
          level: str = "Grand Slam", date: Optional[str] = None) -> Match:
    return Match(p1=p1, p2=p2, surface=surface, round=round, level=level, date=date)

# ===========================================================================
# ✏️  EDIT HERE — add as many matches as you want
# ===========================================================================

MATCHES = [
    match("Jannik Sinner",    "Carlos Alcaraz",  surface="Clay", round="QF",  level="Grand Slam", date="2026-03-29"),
    match("Alexander Zverev", "Novak Djokovic",   surface="Clay", round="SF",  level="Grand Slam", date="2026-03-30"),
]

# ===========================================================================
# Engine — do not edit below this line
# ===========================================================================

def _run():
    if not MODEL_PATH.exists():
        print(f"[ERROR] No model found at {MODEL_PATH}. Run models/train.py first.")
        return

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model        = bundle["model"]
    feature_cols = bundle["feature_cols"]
    model_name   = bundle["name"]

    features_df = pd.DataFrame()
    if FEATURES_CSV.exists():
        features_df = pd.read_csv(FEATURES_CSV, parse_dates=["date"])

    print()
    print("=" * 60)
    print("  🎾  Tennis Match Predictor")
    print(f"  Model: {model_name}")
    print("=" * 60)

    for i, m in enumerate(MATCHES, 1):
        cutoff       = pd.Timestamp(m.date) if m.date else pd.Timestamp.today().normalize()
        player_stats = _build_player_stats_snapshot(features_df, cutoff) if not features_df.empty else {}

        X = build_match_feature_vector(
            p1_name=m.p1, p2_name=m.p2,
            surface=m.surface, round_name=m.round,
            tourney_level=m.level, cutoff_date=cutoff,
            player_stats=player_stats, features_df=features_df,
            feature_cols=feature_cols,
        )

        proba       = model.predict_proba(X)[0]
        p1_win_prob = float(proba[1])
        winner      = m.p1 if p1_win_prob >= 0.5 else m.p2
        confidence  = p1_win_prob if p1_win_prob >= 0.5 else (1 - p1_win_prob)

        print()
        print(f"  [{i}] {m.p1}  vs  {m.p2}")
        print(f"       Surface : {m.surface}  |  Round : {m.round}  |  Level : {m.level}  |  Date : {cutoff.date()}")
        print(f"       ▶ Predicted winner : {winner}  ({confidence:.1%} confidence)")

    print()
    print("=" * 60)
    print()

if __name__ == "__main__":
    _run()
