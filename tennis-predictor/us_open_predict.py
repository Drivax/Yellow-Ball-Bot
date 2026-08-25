"""
us_open_predict.py
------------------
Scrapes upcoming US Open (ATP & WTA) matches, generates win-probability
predictions for every match in each draw, and writes a formatted report to:

    results/us_open_<year>_predictions.txt

Prediction engine (in preference order):
  1. Trained model loaded from models/best_model.pkl
  2. ELO-based fallback using the player ratings in scraper_us_open.py

Usage:
    python us_open_predict.py
    python us_open_predict.py --year 2026
    python us_open_predict.py --no-model     # force ELO fallback
"""

import argparse
import logging
import math
import pickle
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
RESULTS_DIR = BASE_DIR / "results"
FEATURES_CSV = BASE_DIR / "data" / "processed" / "features.csv"

sys.path.insert(0, str(BASE_DIR))

# ---------------------------------------------------------------------------
# Round ordering
# ---------------------------------------------------------------------------
ROUND_ORDER = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]

ROUND_LABELS = {
    "R128": "Round of 128",
    "R64": "Round of 64",
    "R32": "Round of 32",
    "R16": "Round of 16",
    "QF": "Quarterfinal",
    "SF": "Semifinal",
    "F": "Final",
}

# ---------------------------------------------------------------------------
# ELO-based predictor (fallback when no trained model is available)
# ---------------------------------------------------------------------------


class EloPredictor:
    """Simple Elo-based win-probability predictor."""

    def __init__(self, tour: str) -> None:
        from scraping.scraper_us_open import ATP_PLAYER_ELO, WTA_PLAYER_ELO

        self.elo_table: dict[str, float] = (
            ATP_PLAYER_ELO if tour == "ATP" else WTA_PLAYER_ELO
        )
        self.name = "ELO (fallback)"

    def _elo(self, player: str) -> float:
        return self.elo_table.get(player, 1600)

    def predict_proba(self, p1: str, p2: str) -> float:
        """Return P(p1 beats p2) using the standard Elo formula."""
        elo_diff = self._elo(p1) - self._elo(p2)
        return 1.0 / (1.0 + math.pow(10.0, -elo_diff / 400.0))


# ---------------------------------------------------------------------------
# Model-based predictor wrapper
# ---------------------------------------------------------------------------


class ModelPredictor:
    """Wraps the trained sklearn-compatible model from best_model.pkl."""

    def __init__(self, bundle: dict, features_df: pd.DataFrame) -> None:
        self.model = bundle["model"]
        self.feature_cols: list[str] = bundle["feature_cols"]
        self.name: str = bundle["name"]
        self.features_df = features_df

    def predict_proba(
        self,
        p1: str,
        p2: str,
        surface: str = "Hard",
        round_name: str = "R32",
        tourney_level: str = "Grand Slam",
        cutoff_date: pd.Timestamp | None = None,
    ) -> float:
        """Return P(p1 beats p2)."""
        from models.predict import (
            build_match_feature_vector,
            _build_player_stats_snapshot,
        )

        if cutoff_date is None:
            cutoff_date = pd.Timestamp.today().normalize()

        player_stats = (
            _build_player_stats_snapshot(self.features_df, cutoff_date)
            if not self.features_df.empty
            else {}
        )

        X = build_match_feature_vector(
            p1_name=p1,
            p2_name=p2,
            surface=surface,
            round_name=round_name,
            tourney_level=tourney_level,
            cutoff_date=cutoff_date,
            player_stats=player_stats,
            features_df=self.features_df,
            feature_cols=self.feature_cols,
        )
        proba = self.model.predict_proba(X)[0]
        return float(proba[1])


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------


def predict_matches(
    matches_df: pd.DataFrame,
    predictor,
    tour: str,
    surface: str = "Hard",
    tourney_level: str = "Grand Slam",
) -> pd.DataFrame:
    """
    Add prediction columns to a matches DataFrame.
    Works with both EloPredictor and ModelPredictor.
    """
    rows = []
    for _, row in matches_df.iterrows():
        p1 = str(row["player1"])
        p2 = str(row["player2"])
        rnd = str(row.get("round", "R32"))

        if isinstance(predictor, ModelPredictor):
            p1_win_prob = predictor.predict_proba(
                p1=p1,
                p2=p2,
                surface=surface,
                round_name=rnd,
                tourney_level=tourney_level,
            )
        else:
            p1_win_prob = predictor.predict_proba(p1, p2)

        predicted_winner = p1 if p1_win_prob >= 0.5 else p2
        confidence = p1_win_prob if p1_win_prob >= 0.5 else (1.0 - p1_win_prob)

        rows.append(
            {
                "round": rnd,
                "player1": p1,
                "player2": p2,
                "predicted_winner": predicted_winner,
                "p1_win_prob": round(p1_win_prob, 4),
                "confidence": round(confidence, 4),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Text-file report writer
# ---------------------------------------------------------------------------

_WIDTH = 72


def _hr(char: str = "=") -> str:
    return char * _WIDTH


def _centre(text: str) -> str:
    return text.center(_WIDTH)


def _format_section(
    predictions_df: pd.DataFrame,
    tournament_name: str,
    surface: str,
    predictor_name: str,
) -> list[str]:
    lines: list[str] = []
    lines.append(_hr())
    lines.append(_centre(f"  {tournament_name.upper()}  "))
    lines.append(_centre(f"Surface: {surface}   |   Model: {predictor_name}"))
    lines.append(_hr())

    final_rows = predictions_df[predictions_df["round"] == "F"]
    champion = final_rows["predicted_winner"].iloc[0] if not final_rows.empty else "TBD"

    lines.append(f"  Predicted Champion: {champion}")
    lines.append("")

    for rnd in ROUND_ORDER:
        sub = predictions_df[predictions_df["round"] == rnd]
        if sub.empty:
            continue
        label = ROUND_LABELS.get(rnd, rnd)
        lines.append(f"  [ {label} ]")
        lines.append("  " + "-" * (_WIDTH - 4))
        for _, row in sub.iterrows():
            p1 = row["player1"]
            p2 = row["player2"]
            winner = row["predicted_winner"]
            conf = row["confidence"]
            loser = p2 if winner == p1 else p1
            lines.append(f"  {p1:<30} vs  {p2}")
            lines.append(f"      ▶ {winner}  wins  ({conf:.1%} confidence)")
            lines.append(f"        (over {loser})")
            lines.append("")

    return lines


def write_predictions_txt(
    atp_df: pd.DataFrame,
    wta_df: pd.DataFrame,
    atp_predictor_name: str,
    wta_predictor_name: str,
    output_path: Path,
    year: int = 2026,
) -> None:
    """Write the full prediction report to a .txt file."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    lines.append(_hr("*"))
    lines.append(_centre(""))
    lines.append(_centre("TENNIS PREDICTIONS — US OPEN"))
    lines.append(_centre(f"Year: {year}   |   Generated: {now}"))
    lines.append(_centre(""))
    lines.append(_hr("*"))
    lines.append("")
    lines.append("  Surface: Hard (outdoor)  |  Venue: USTA Billie Jean King National Tennis Center")
    lines.append("  Location: Flushing Meadows, New York")
    lines.append("")

    atp_section = _format_section(
        atp_df,
        f"ATP US Open {year}  (Hard, Grand Slam)",
        "Hard",
        atp_predictor_name,
    )
    lines.extend(atp_section)
    lines.append("")

    wta_section = _format_section(
        wta_df,
        f"WTA US Open {year}  (Hard, Grand Slam)",
        "Hard",
        wta_predictor_name,
    )
    lines.extend(wta_section)
    lines.append("")

    lines.append(_hr("="))
    lines.append(_centre("END OF REPORT"))
    lines.append(_hr("="))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Predictions written to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict US Open ATP & WTA matches."
    )
    parser.add_argument(
        "--year", type=int, default=date.today().year,
        help="Tournament year (default: current year).",
    )
    parser.add_argument(
        "--no-model", action="store_true",
        help="Skip model loading and use the ELO fallback predictor.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .txt file path.",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = RESULTS_DIR / f"us_open_{args.year}_predictions.txt"

    # -----------------------------------------------------------------------
    # Scrape upcoming matches
    # -----------------------------------------------------------------------
    from scraping.scraper_us_open import (
        get_atp_us_open_matches,
        get_wta_us_open_matches,
    )

    logger.info("Fetching ATP US Open %d matches…", args.year)
    atp_matches_df = get_atp_us_open_matches(args.year)
    logger.info("ATP: %d match-ups loaded.", len(atp_matches_df))

    logger.info("Fetching WTA US Open %d matches…", args.year)
    wta_matches_df = get_wta_us_open_matches(args.year)
    logger.info("WTA: %d match-ups loaded.", len(wta_matches_df))

    # -----------------------------------------------------------------------
    # Load model or fall back to ELO predictor
    # -----------------------------------------------------------------------
    features_df = pd.DataFrame()
    if FEATURES_CSV.exists():
        features_df = pd.read_csv(FEATURES_CSV, parse_dates=["date"])

    use_model = not args.no_model and BEST_MODEL_PATH.exists()

    if use_model:
        try:
            with open(BEST_MODEL_PATH, "rb") as f:
                bundle = pickle.load(f)
            atp_predictor: ModelPredictor | EloPredictor = ModelPredictor(bundle, features_df)
            wta_predictor: ModelPredictor | EloPredictor = ModelPredictor(bundle, features_df)
            logger.info("Loaded trained model: %s", bundle["name"])
        except Exception as exc:
            logger.warning("Could not load model (%s). Falling back to ELO.", exc)
            use_model = False

    if not use_model:
        logger.info("Using ELO fallback predictor.")
        atp_predictor = EloPredictor("ATP")
        wta_predictor = EloPredictor("WTA")

    # -----------------------------------------------------------------------
    # Generate predictions
    # -----------------------------------------------------------------------
    logger.info("Predicting ATP US Open matches…")
    atp_predictions = predict_matches(
        atp_matches_df,
        atp_predictor,
        tour="ATP",
        surface="Hard",
        tourney_level="Grand Slam",
    )

    logger.info("Predicting WTA US Open matches…")
    wta_predictions = predict_matches(
        wta_matches_df,
        wta_predictor,
        tour="WTA",
        surface="Hard",
        tourney_level="Grand Slam",
    )

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  🎾  US OPEN PREDICTIONS — ATP & WTA")
    print("=" * 72)

    for tour, preds in [("ATP US Open", atp_predictions), ("WTA US Open", wta_predictions)]:
        print(f"\n  {tour}")
        print("  " + "-" * 68)
        for rnd in ROUND_ORDER:
            sub = preds[preds["round"] == rnd]
            if sub.empty:
                continue
            label = ROUND_LABELS.get(rnd, rnd)
            print(f"\n  [{label}]")
            for _, row in sub.iterrows():
                print(
                    f"    {row['player1']:<28} vs  {row['player2']:<28}"
                    f"  →  {row['predicted_winner']}  ({row['confidence']:.1%})"
                )

    print("\n" + "=" * 72)

    # -----------------------------------------------------------------------
    # Save CSV predictions
    # -----------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    atp_csv = RESULTS_DIR / f"us_open_{args.year}_atp_predictions.csv"
    wta_csv = RESULTS_DIR / f"us_open_{args.year}_wta_predictions.csv"
    atp_predictions.to_csv(atp_csv, index=False)
    wta_predictions.to_csv(wta_csv, index=False)
    logger.info("Saved ATP CSV predictions to %s", atp_csv)
    logger.info("Saved WTA CSV predictions to %s", wta_csv)

    # -----------------------------------------------------------------------
    # Write predictions .txt report
    # -----------------------------------------------------------------------
    write_predictions_txt(
        atp_df=atp_predictions,
        wta_df=wta_predictions,
        atp_predictor_name=atp_predictor.name,
        wta_predictor_name=wta_predictor.name,
        output_path=args.output,
        year=args.year,
    )

    print(f"\n  📄  Full predictions report saved to: {args.output}\n")


if __name__ == "__main__":
    main()
