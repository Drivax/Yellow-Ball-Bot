"""
cleaner.py
----------
Loads all raw Jeff Sackmann ATP match CSV files, merges them into a single
chronologically sorted DataFrame, cleans missing values, and normalises player
identifiers.

Usage:
    python preprocessing/cleaner.py
"""

import logging
from pathlib import Path

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
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# ---------------------------------------------------------------------------
# Column renaming map (Sackmann column names → project canonical names)
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "tourney_id": "tourney_id",
    "tourney_name": "tournament",
    "surface": "surface",
    "draw_size": "draw_size",
    "tourney_level": "tourney_level",
    "tourney_date": "date",
    "match_num": "match_num",
    "winner_id": "winner_id",
    "winner_seed": "winner_seed",
    "winner_entry": "winner_entry",
    "winner_name": "winner_name",
    "winner_hand": "winner_hand",
    "winner_ht": "winner_ht",
    "winner_ioc": "winner_ioc",
    "winner_age": "winner_age",
    "winner_rank": "winner_rank",
    "winner_rank_points": "winner_rank_points",
    "loser_id": "loser_id",
    "loser_seed": "loser_seed",
    "loser_entry": "loser_entry",
    "loser_name": "loser_name",
    "loser_hand": "loser_hand",
    "loser_ht": "loser_ht",
    "loser_ioc": "loser_ioc",
    "loser_age": "loser_age",
    "loser_rank": "loser_rank",
    "loser_rank_points": "loser_rank_points",
    "score": "score",
    "best_of": "best_of",
    "round": "round",
    "minutes": "minutes",
    # Serve statistics
    "w_ace": "winner_aces",
    "w_df": "winner_double_faults",
    "w_svpt": "winner_serve_points",
    "w_1stIn": "winner_first_serves_in",
    "w_1stWon": "winner_first_serve_won",
    "w_2ndWon": "winner_second_serve_won",
    "w_SvGms": "winner_serve_games",
    "w_bpSaved": "winner_bp_saved",
    "w_bpFaced": "winner_bp_faced",
    "l_ace": "loser_aces",
    "l_df": "loser_double_faults",
    "l_svpt": "loser_serve_points",
    "l_1stIn": "loser_first_serves_in",
    "l_1stWon": "loser_first_serve_won",
    "l_2ndWon": "loser_second_serve_won",
    "l_SvGms": "loser_serve_games",
    "l_bpSaved": "loser_bp_saved",
    "l_bpFaced": "loser_bp_faced",
}

SURFACE_NORMALISATION = {
    "clay": "Clay",
    "grass": "Grass",
    "hard": "Hard",
    "carpet": "Carpet",
}

TOURNEY_LEVEL_MAP = {
    "G": "Grand Slam",
    "M": "Masters",
    "A": "ATP500",
    "D": "Davis Cup",
    "F": "Tour Finals",
    "C": "Challenger",
    "S": "Satellite",
    "U": "Unknown",
}

REQUIRED_COLUMNS = [
    "date", "tournament", "surface", "round",
    "winner_id", "winner_name", "winner_rank", "winner_age",
    "loser_id", "loser_name", "loser_rank", "loser_age",
]


def _load_single_year(csv_path: Path) -> pd.DataFrame:
    """Load and lightly validate one year's match CSV."""
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:
        logger.warning("Could not read %s: %s", csv_path, exc)
        return pd.DataFrame()

    # Keep only columns we recognise
    keep = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df[list(keep.keys())].rename(columns=keep)
    return df


def load_all_matches(raw_dir: Path = RAW_DIR, include_challengers: bool = False) -> pd.DataFrame:
    """Load every ATP main-tour match CSV and return a single merged DataFrame."""
    patterns = ["atp_matches_[12][0-9][0-9][0-9].csv"]
    if include_challengers:
        patterns += [
            "atp_matches_qual_chall_[12][0-9][0-9][0-9].csv",
            "atp_matches_futures_[12][0-9][0-9][0-9].csv",
        ]

    all_files: list[Path] = []
    for pattern in patterns:
        all_files.extend(sorted(raw_dir.glob(pattern)))

    if not all_files:
        logger.error(
            "No CSV files found in %s. Run scraping/scraper_atp.py first.", raw_dir
        )
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for path in tqdm(all_files, desc="Loading CSV files"):
        df = _load_single_year(path)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d total match records.", len(combined))
    return combined


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning transformations to the merged DataFrame."""
    if df.empty:
        return df

    # --- Parse date ---
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])

    # --- Normalise surface ---
    if "surface" in df.columns:
        df["surface"] = (
            df["surface"].str.strip().str.lower().map(SURFACE_NORMALISATION).fillna("Unknown")
        )

    # --- Normalise tournament level ---
    if "tourney_level" in df.columns:
        df["tourney_level"] = df["tourney_level"].map(TOURNEY_LEVEL_MAP).fillna("Unknown")

    # --- Ensure required IDs are integers ---
    for col in ("winner_id", "loser_id"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # --- Ranking imputation: fill missing rank with a large sentinel ---
    for col in ("winner_rank", "loser_rank"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(2000).astype(int)

    # --- Age: numeric ---
    for col in ("winner_age", "loser_age"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Serve stats: numeric ---
    serve_cols = [
        "winner_aces", "winner_double_faults", "winner_serve_points",
        "winner_first_serves_in", "winner_first_serve_won", "winner_second_serve_won",
        "winner_bp_saved", "winner_bp_faced",
        "loser_aces", "loser_double_faults", "loser_serve_points",
        "loser_first_serves_in", "loser_first_serve_won", "loser_second_serve_won",
        "loser_bp_saved", "loser_bp_faced",
    ]
    for col in serve_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Drop duplicates ---
    df = df.drop_duplicates()

    # --- Sort chronologically ---
    df = df.sort_values("date").reset_index(drop=True)

    logger.info("After cleaning: %d rows remain.", len(df))
    return df


def save(df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """Persist the cleaned DataFrame."""
    if output_path is None:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PROCESSED_DIR / "matches_clean.csv"
    df.to_csv(output_path, index=False)
    logger.info("Saved cleaned data to %s", output_path)
    return output_path


def main() -> None:
    raw_df = load_all_matches(include_challengers=False)
    clean_df = clean(raw_df)
    save(clean_df)


if __name__ == "__main__":
    main()
