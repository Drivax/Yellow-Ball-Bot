"""
predict.py
----------
Loads the best trained model and simulates the full Roland Garros 2025
tournament bracket, predicting the winner and win probability for each match.

The script also compares predictions against the real results (embedded in
scraping/scraper_tournaments.py) and produces a bracket comparison table
and visual.

Usage:
    python models/predict.py
    python models/predict.py --draw-csv path/to/draw.csv
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(__file__).resolve().parent
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
RESULTS_DIR = BASE_DIR / "results"
PREDICTIONS_CSV = RESULTS_DIR / "roland_garros_2025_predictions.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FEATURES_CSV = PROCESSED_DIR / "features.csv"

# Add project root to path so we can import from scraping/
sys.path.insert(0, str(BASE_DIR))

ROUND_ORDER = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]

# ---------------------------------------------------------------------------
# Feature helpers — build a feature row from per-player stats snapshot
# ---------------------------------------------------------------------------

def _build_player_stats_snapshot(features_df: pd.DataFrame, cutoff_date: pd.Timestamp) -> dict:
    """
    Compute per-player stats frozen as of *cutoff_date* from the feature history.
    Returns a dict: player_name -> { elo, avg_rank, form, surface_win_rates, ... }
    """
    hist = features_df[features_df["date"] < cutoff_date].copy()
    player_stats: dict[str, dict] = {}

    # For each player we want the most recent row they appeared in
    for col_player, col_elo, col_rank, col_form in [
        ("p1_name", "elo_p1", "avg_rank_p1", "form_p1"),
        ("p2_name", "elo_p2", "avg_rank_p2", "form_p2"),
    ]:
        if col_player not in hist.columns:
            continue
        latest = hist.sort_values("date").groupby(col_player).last().reset_index()
        for _, row in latest.iterrows():
            name = row[col_player]
            if name not in player_stats:
                player_stats[name] = {
                    "elo": row.get(col_elo, 1500),
                    "rank": row.get(col_rank, 500),
                    "form": row.get(col_form, 0.5),
                }

    return player_stats


def _get_surface_win_rate(features_df: pd.DataFrame, player_name: str,
                           surface: str, cutoff_date: pd.Timestamp) -> float:
    cutoff_52w = cutoff_date - pd.Timedelta(weeks=52)
    mask_p1 = (
        (features_df["p1_name"] == player_name)
        & (features_df["surface"] == surface)
        & (features_df["date"] >= cutoff_52w)
        & (features_df["date"] < cutoff_date)
    )
    mask_p2 = (
        (features_df["p2_name"] == player_name)
        & (features_df["surface"] == surface)
        & (features_df["date"] >= cutoff_52w)
        & (features_df["date"] < cutoff_date)
    )
    vals_p1 = features_df.loc[mask_p1, "surface_win_rate_p1_52w"]
    vals_p2 = features_df.loc[mask_p2, "surface_win_rate_p2_52w"]
    all_vals = pd.concat([vals_p1, vals_p2])
    return float(all_vals.mean()) if not all_vals.empty else 0.5


def _get_h2h_win_rate(features_df: pd.DataFrame, p1_name: str, p2_name: str,
                       cutoff_date: pd.Timestamp) -> float:
    mask = (
        (features_df["p1_name"] == p1_name)
        & (features_df["p2_name"] == p2_name)
        & (features_df["date"] < cutoff_date)
    )
    relevant = features_df[mask]
    if relevant.empty:
        return 0.5
    return float(relevant["h2h_win_rate_p1"].iloc[-1])


def build_match_feature_vector(
    p1_name: str,
    p2_name: str,
    surface: str,
    round_name: str,
    tourney_level: str,
    cutoff_date: pd.Timestamp,
    player_stats: dict,
    features_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Assemble a single feature row for a prediction."""

    SURFACE_CODES = {"Clay": 0, "Grass": 1, "Hard": 2, "Carpet": 3, "Unknown": 4}
    ROUND_ORDER_MAP = {
        "R128": 1, "R64": 2, "R32": 3, "R16": 4,
        "QF": 5, "SF": 6, "F": 7,
    }
    TOURNEY_LEVEL_CODES = {
        "Grand Slam": 4, "Masters": 3, "ATP500": 2, "Davis Cup": 1,
        "Tour Finals": 4, "Challenger": 0,
    }

    p1 = player_stats.get(p1_name, {"elo": 1500, "rank": 500, "form": 0.5})
    p2 = player_stats.get(p2_name, {"elo": 1500, "rank": 500, "form": 0.5})

    elo_p1 = p1["elo"]
    elo_p2 = p2["elo"]
    rank_p1 = p1["rank"]
    rank_p2 = p2["rank"]
    form_p1 = p1["form"]
    form_p2 = p2["form"]

    h2h = _get_h2h_win_rate(features_df, p1_name, p2_name, cutoff_date)
    surf_p1 = _get_surface_win_rate(features_df, p1_name, surface, cutoff_date)
    surf_p2 = _get_surface_win_rate(features_df, p2_name, surface, cutoff_date)

    row = {
        "elo_p1": elo_p1,
        "elo_p2": elo_p2,
        "elo_diff": elo_p1 - elo_p2,
        "h2h_win_rate_p1": h2h,
        "surface_win_rate_p1_52w": surf_p1,
        "surface_win_rate_p2_52w": surf_p2,
        "avg_rank_p1": rank_p1,
        "avg_rank_p2": rank_p2,
        "rank_diff": rank_p1 - rank_p2,
        "form_p1": form_p1,
        "form_p2": form_p2,
        "form_diff": form_p1 - form_p2,
        "tourney_win_rate_p1": 0.5,
        "tourney_win_rate_p2": 0.5,
        "age_diff": 0.0,
        "days_since_last_p1": 7,
        "days_since_last_p2": 7,
        "surface_enc": SURFACE_CODES.get(surface, 4),
        "round_enc": ROUND_ORDER_MAP.get(round_name, 0),
        "tourney_level_enc": TOURNEY_LEVEL_CODES.get(tourney_level, 0),
    }
    return pd.DataFrame([{col: row.get(col, 0) for col in feature_cols}])


# ---------------------------------------------------------------------------
# Bracket simulation
# ---------------------------------------------------------------------------

def simulate_bracket(draw_df: pd.DataFrame, model, feature_cols: list[str],
                     player_stats: dict, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate the tournament bracket round by round.
    *draw_df* must have columns: round, player1, player2
    """
    prediction_rows = []
    # Work round by round so predicted winners carry forward
    active_players_by_round: dict[str, list[str]] = {}
    for rnd in ROUND_ORDER:
        round_matches = draw_df[draw_df["round"] == rnd]
        if round_matches.empty:
            continue

        logger.info("Simulating round: %s (%d matches)", rnd, len(round_matches))

        # Match date proxy — use a fixed date per round (actual schedule)
        ROUND_DATES = {
            "R128": pd.Timestamp("2025-05-25"),
            "R64":  pd.Timestamp("2025-05-27"),
            "R32":  pd.Timestamp("2025-05-29"),
            "R16":  pd.Timestamp("2025-05-31"),
            "QF":   pd.Timestamp("2025-06-03"),
            "SF":   pd.Timestamp("2025-06-06"),
            "F":    pd.Timestamp("2025-06-08"),
        }
        cutoff = ROUND_DATES.get(rnd, pd.Timestamp("2025-05-25"))

        for _, match in round_matches.iterrows():
            p1 = str(match["player1"])
            p2 = str(match["player2"])

            X = build_match_feature_vector(
                p1_name=p1, p2_name=p2, surface="Clay",
                round_name=rnd, tourney_level="Grand Slam",
                cutoff_date=cutoff, player_stats=player_stats,
                features_df=features_df, feature_cols=feature_cols,
            )
            proba = model.predict_proba(X)[0]
            p1_win_prob = float(proba[1])
            predicted_winner = p1 if p1_win_prob >= 0.5 else p2
            win_prob = p1_win_prob if p1_win_prob >= 0.5 else (1 - p1_win_prob)

            prediction_rows.append({
                "round": rnd,
                "player1": p1,
                "player2": p2,
                "predicted_winner": predicted_winner,
                "win_probability": round(win_prob, 4),
            })
            logger.debug("  %s vs %s → %s (%.2f%%)", p1, p2, predicted_winner, win_prob * 100)

    return pd.DataFrame(prediction_rows)


# ---------------------------------------------------------------------------
# Comparison with real results
# ---------------------------------------------------------------------------

def compare_predictions(predictions_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    """Merge predictions with real results and compute accuracy."""
    merged = predictions_df.merge(
        real_df[["round", "player1", "player2", "winner"]].rename(
            columns={"winner": "actual_winner"}
        ),
        on=["round", "player1", "player2"],
        how="left",
    )
    # Also try reversed player1/player2 order
    reversed_real = real_df[["round", "player1", "player2", "winner"]].copy()
    reversed_real = reversed_real.rename(
        columns={"player1": "player2", "player2": "player1", "winner": "actual_winner"}
    )
    merged = merged.combine_first(
        predictions_df.merge(reversed_real, on=["round", "player1", "player2"], how="left")
    )

    merged["correct"] = merged["predicted_winner"] == merged["actual_winner"]
    return merged


def print_accuracy_report(comparison_df: pd.DataFrame) -> None:
    total = comparison_df["actual_winner"].notna().sum()
    correct = comparison_df["correct"].sum()
    print("\n" + "=" * 60)
    print("  Roland Garros 2025 — Prediction Accuracy Report")
    print("=" * 60)
    if total > 0:
        print(f"  Overall: {correct}/{total} correct ({100*correct/total:.1f}%)")
    print("\n  Per round breakdown:")
    for rnd in ROUND_ORDER:
        sub = comparison_df[comparison_df["round"] == rnd]
        n_total = sub["actual_winner"].notna().sum()
        n_correct = sub["correct"].sum()
        if n_total > 0:
            print(f"    {rnd:5s}: {n_correct}/{n_total} ({100*n_correct/n_total:.0f}%)")
    print("=" * 60)
    print("\n  Match-by-match comparison:")
    print(f"  {'Round':5}  {'Player 1':25}  {'Player 2':25}  {'Predicted':25}  {'Actual':25}  {'✓'}")
    print("  " + "-" * 115)
    for _, row in comparison_df.iterrows():
        ok = "✓" if row["correct"] else "✗"
        actual = str(row.get("actual_winner", "?"))
        print(
            f"  {row['round']:5}  {row['player1']:25}  {row['player2']:25}"
            f"  {row['predicted_winner']:25}  {actual:25}  {ok}"
        )
    print()


def plot_bracket_comparison(comparison_df: pd.DataFrame, output_path: Path) -> None:
    """Generate a matplotlib visual of predicted vs actual winners per round."""
    fig, axes = plt.subplots(1, len(ROUND_ORDER), figsize=(22, 10))
    fig.suptitle("Roland Garros 2025 — Predicted vs Actual", fontsize=14, fontweight="bold")

    for ax, rnd in zip(axes, ROUND_ORDER):
        sub = comparison_df[comparison_df["round"] == rnd].reset_index(drop=True)
        ax.set_title(rnd, fontweight="bold")
        ax.axis("off")
        if sub.empty:
            continue
        for i, row in sub.iterrows():
            colour = "green" if row.get("correct") else "red"
            if pd.isna(row.get("actual_winner")):
                colour = "blue"
            text = f"{row['predicted_winner']}\n({row['win_probability']:.0%})"
            ax.text(
                0.5, 1 - (i + 0.5) / max(len(sub), 1),
                text, ha="center", va="center",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colour, alpha=0.3),
                transform=ax.transAxes,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved bracket comparison chart to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Roland Garros 2025 bracket.")
    parser.add_argument("--draw-csv", type=Path, default=None,
                        help="Path to draw CSV (optional; uses embedded data by default).")
    args = parser.parse_args()

    if not BEST_MODEL_PATH.exists():
        logger.error("No model at %s. Run models/train.py first.", BEST_MODEL_PATH)
        return

    # Load model
    with open(BEST_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    model_name = bundle["name"]
    logger.info("Loaded model: %s", model_name)

    # Load Roland Garros 2025 draw
    if args.draw_csv and args.draw_csv.exists():
        draw_df = pd.read_csv(args.draw_csv)
    else:
        from scraping.scraper_tournaments import get_roland_garros_2025_draw
        draw_df = get_roland_garros_2025_draw()
    logger.info("Draw has %d matches.", len(draw_df))

    # Load historical features for player stat snapshots
    if FEATURES_CSV.exists():
        features_df = pd.read_csv(FEATURES_CSV)
        features_df["date"] = pd.to_datetime(features_df["date"])
        cutoff = pd.Timestamp("2025-05-24")
        player_stats = _build_player_stats_snapshot(features_df, cutoff)
        logger.info("Loaded player stats snapshot for %d players.", len(player_stats))
    else:
        logger.warning("Features CSV not found. Using default ELO=1500 for all players.")
        features_df = pd.DataFrame()
        player_stats = {}

    # Simulate bracket
    predictions_df = simulate_bracket(draw_df, model, feature_cols, player_stats, features_df)

    # Save predictions
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(PREDICTIONS_CSV, index=False)
    logger.info("Saved predictions to %s", PREDICTIONS_CSV)

    # Compare with real results
    real_df = draw_df.copy()
    if "winner" in real_df.columns:
        comparison_df = compare_predictions(predictions_df, real_df)
        print_accuracy_report(comparison_df)
        comparison_csv = RESULTS_DIR / "roland_garros_2025_comparison.csv"
        comparison_df.to_csv(comparison_csv, index=False)
        logger.info("Saved comparison to %s", comparison_csv)

        # Plot
        bracket_img = RESULTS_DIR / "bracket_comparison.png"
        plot_bracket_comparison(comparison_df, bracket_img)


if __name__ == "__main__":
    main()
