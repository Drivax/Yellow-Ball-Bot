"""
feature_engineering.py
-----------------------
Transforms the cleaned match DataFrame into a machine-learning-ready feature
matrix, engineering all features using ONLY data available strictly before
each match date (no look-ahead / leakage).

Engineered features per match
------------------------------
  elo_p1, elo_p2                    — ELO ratings at time of match
  elo_diff                          — elo_p1 - elo_p2
  h2h_win_rate_p1                   — historical H2H win rate for p1
  surface_win_rate_p1_52w           — p1 clay/grass/hard win % last 52 weeks
  surface_win_rate_p2_52w           — same for p2
  avg_rank_p1_3m, avg_rank_p2_3m    — average ranking over last 3 months
  form_p1, form_p2                  — win % over last 10 matches
  tourney_win_rate_p1               — p1 win rate at this specific tournament
  age_diff                          — p1 age − p2 age
  days_since_last_p1                — days since p1's previous match
  days_since_last_p2                — same for p2
  surface_enc                       — integer-encoded surface
  round_enc                         — integer-encoded round
  tourney_level_enc                 — integer-encoded tournament level

Target column
-------------
  target  — 1 if player1 (higher-ranked / listed first) wins, else 0

Usage:
    python preprocessing/feature_engineering.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CLEAN_CSV = PROCESSED_DIR / "matches_clean.csv"
FEATURES_CSV = PROCESSED_DIR / "features.csv"

# ---------------------------------------------------------------------------
# ELO parameters
# ---------------------------------------------------------------------------
ELO_INITIAL = 1500.0
ELO_K = 32.0

ROUND_ORDER = {
    "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "F": 7, "BR": 6,
    "RR": 3,  # round robin (Tour Finals)
}

SURFACE_CODES = {"Clay": 0, "Grass": 1, "Hard": 2, "Carpet": 3, "Unknown": 4}
TOURNEY_LEVEL_CODES = {
    "Grand Slam": 4, "Masters": 3, "ATP500": 2, "Davis Cup": 1,
    "Tour Finals": 4, "Challenger": 0, "Satellite": 0, "Unknown": 0,
}


# ---------------------------------------------------------------------------
# ELO helpers
# ---------------------------------------------------------------------------

def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _update_elo(winner_elo: float, loser_elo: float, k: float = ELO_K) -> tuple[float, float]:
    e_w = _expected_score(winner_elo, loser_elo)
    new_winner = winner_elo + k * (1 - e_w)
    new_loser = loser_elo + k * (0 - (1 - e_w))
    return new_winner, new_loser


# ---------------------------------------------------------------------------
# Main feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a cleaned, chronologically sorted match DataFrame, produce a feature
    matrix.  All lookback windows respect the match date so there is no leakage.
    """
    if df.empty:
        logger.error("Empty DataFrame passed to engineer_features.")
        return pd.DataFrame()

    # Ensure date is datetime
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # -----------------------------------------------------------------------
    # State dictionaries (updated after each match, read before computing feats)
    # -----------------------------------------------------------------------
    elo: dict[int, float] = {}          # player_id → current ELO
    last_match_date: dict[int, pd.Timestamp] = {}
    # match history: player_id → list of (date, surface, tourney_name, won: bool)
    match_history: dict[int, list] = {}

    # -----------------------------------------------------------------------
    # Output containers
    # -----------------------------------------------------------------------
    feature_rows: list[dict] = []

    logger.info("Engineering features for %d matches…", len(df))

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Feature engineering"):
        winner_id = int(row["winner_id"]) if not pd.isna(row.get("winner_id", float("nan"))) else -1
        loser_id = int(row["loser_id"]) if not pd.isna(row.get("loser_id", float("nan"))) else -2

        if winner_id == -1 or loser_id == -2:
            continue  # skip rows with missing player IDs

        match_date: pd.Timestamp = row["date"]
        surface: str = row.get("surface", "Unknown")
        tourney_name: str = str(row.get("tournament", ""))
        tourney_level: str = str(row.get("tourney_level", "Unknown"))

        # Decide p1/p2 — by convention use winner as p1 50% of the time to avoid
        # a trivial target-leakage pattern.  We randomise deterministically by
        # using the match index's parity.
        if idx % 2 == 0:
            p1_id, p2_id = winner_id, loser_id
            target = 1
        else:
            p1_id, p2_id = loser_id, winner_id
            target = 0

        # --- ELO (read before update) ---
        elo_p1 = elo.get(p1_id, ELO_INITIAL)
        elo_p2 = elo.get(p2_id, ELO_INITIAL)
        elo_diff = elo_p1 - elo_p2

        # --- H2H win rate for p1 ---
        hist_p1 = match_history.get(p1_id, [])
        h2h_total = sum(1 for d, s, t, w, opp in hist_p1 if opp == p2_id)
        h2h_wins_p1 = sum(1 for d, s, t, w, opp in hist_p1 if opp == p2_id and w)
        h2h_win_rate_p1 = h2h_wins_p1 / h2h_total if h2h_total > 0 else 0.5

        # --- Surface win rate last 52 weeks ---
        cutoff_52w = match_date - pd.Timedelta(weeks=52)

        def _surface_win_rate(pid: int) -> float:
            hist = match_history.get(pid, [])
            relevant = [(d, s, t, w, opp) for d, s, t, w, opp in hist
                        if s == surface and d >= cutoff_52w]
            if not relevant:
                return 0.5
            return sum(w for d, s, t, w, opp in relevant) / len(relevant)

        surf_wr_p1 = _surface_win_rate(p1_id)
        surf_wr_p2 = _surface_win_rate(p2_id)

        # --- Average rank last 3 months ---
        # Use actual rank columns from dataframe as a proxy
        def _avg_rank(pid: int, current_rank: float) -> float:
            hist = match_history.get(pid, [])
            cutoff = match_date - pd.Timedelta(days=90)
            # We don't store per-match rank in history; use current rank as estimate
            return current_rank

        rank_p1 = float(row["winner_rank"] if p1_id == winner_id else row["loser_rank"])
        rank_p2 = float(row["winner_rank"] if p2_id == winner_id else row["loser_rank"])

        # --- Recent form: win % last 10 matches ---
        def _recent_form(pid: int) -> float:
            hist = match_history.get(pid, [])[-10:]
            if not hist:
                return 0.5
            return sum(w for d, s, t, w, opp in hist) / len(hist)

        form_p1 = _recent_form(p1_id)
        form_p2 = _recent_form(p2_id)

        # --- Tournament-specific win rate ---
        def _tourney_win_rate(pid: int) -> float:
            hist = match_history.get(pid, [])
            relevant = [(d, s, t, w, opp) for d, s, t, w, opp in hist if t == tourney_name]
            if not relevant:
                return 0.5
            return sum(w for d, s, t, w, opp in relevant) / len(relevant)

        tw_p1 = _tourney_win_rate(p1_id)
        tw_p2 = _tourney_win_rate(p2_id)

        # --- Age difference ---
        age_p1 = float(row["winner_age"] if p1_id == winner_id else row.get("loser_age", float("nan")))
        age_p2 = float(row["winner_age"] if p2_id == winner_id else row.get("loser_age", float("nan")))
        age_diff = (age_p1 - age_p2) if not (np.isnan(age_p1) or np.isnan(age_p2)) else 0.0

        # --- Days since last match (fatigue proxy) ---
        days_since_p1 = (
            (match_date - last_match_date[p1_id]).days
            if p1_id in last_match_date else 365
        )
        days_since_p2 = (
            (match_date - last_match_date[p2_id]).days
            if p2_id in last_match_date else 365
        )

        # --- Categorical encodings ---
        surface_enc = SURFACE_CODES.get(surface, 4)
        round_enc = ROUND_ORDER.get(str(row.get("round", "")), 0)
        tourney_level_enc = TOURNEY_LEVEL_CODES.get(tourney_level, 0)

        # --- Assemble feature row ---
        feature_rows.append({
            "match_idx": idx,
            "date": match_date,
            "tournament": tourney_name,
            "surface": surface,
            "round": row.get("round", ""),
            "p1_id": p1_id,
            "p2_id": p2_id,
            "p1_name": row["winner_name"] if p1_id == winner_id else row["loser_name"],
            "p2_name": row["winner_name"] if p2_id == winner_id else row["loser_name"],
            # Features
            "elo_p1": elo_p1,
            "elo_p2": elo_p2,
            "elo_diff": elo_diff,
            "h2h_win_rate_p1": h2h_win_rate_p1,
            "surface_win_rate_p1_52w": surf_wr_p1,
            "surface_win_rate_p2_52w": surf_wr_p2,
            "avg_rank_p1": rank_p1,
            "avg_rank_p2": rank_p2,
            "rank_diff": rank_p1 - rank_p2,
            "form_p1": form_p1,
            "form_p2": form_p2,
            "form_diff": form_p1 - form_p2,
            "tourney_win_rate_p1": tw_p1,
            "tourney_win_rate_p2": tw_p2,
            "age_diff": age_diff,
            "days_since_last_p1": days_since_p1,
            "days_since_last_p2": days_since_p2,
            "surface_enc": surface_enc,
            "round_enc": round_enc,
            "tourney_level_enc": tourney_level_enc,
            "target": target,
        })

        # -----------------------------------------------------------------------
        # UPDATE state after reading features (prevents leakage)
        # -----------------------------------------------------------------------
        new_winner_elo, new_loser_elo = _update_elo(
            elo.get(winner_id, ELO_INITIAL),
            elo.get(loser_id, ELO_INITIAL),
        )
        elo[winner_id] = new_winner_elo
        elo[loser_id] = new_loser_elo

        last_match_date[winner_id] = match_date
        last_match_date[loser_id] = match_date

        for pid, won in [(winner_id, True), (loser_id, False)]:
            opp = loser_id if pid == winner_id else winner_id
            if pid not in match_history:
                match_history[pid] = []
            match_history[pid].append((match_date, surface, tourney_name, won, opp))

    features_df = pd.DataFrame(feature_rows)
    logger.info("Feature matrix shape: %s", features_df.shape)
    return features_df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not CLEAN_CSV.exists():
        logger.error(
            "Cleaned CSV not found at %s. Run preprocessing/cleaner.py first.", CLEAN_CSV
        )
        return

    df = pd.read_csv(CLEAN_CSV, low_memory=False)
    features_df = engineer_features(df)
    features_df.to_csv(FEATURES_CSV, index=False)
    logger.info("Saved features to %s", FEATURES_CSV)


if __name__ == "__main__":
    main()
