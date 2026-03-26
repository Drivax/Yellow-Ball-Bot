"""
scraper_players.py
------------------
Scrapes per-match player statistics (serve %, aces, double faults, break
points, etc.) from TennisAbstract.com to supplement the Jeff Sackmann dataset
which does not always include those granular stats.

Usage:
    python scraping/scraper_players.py [--player-id <player_id>]

The scraper respects rate limits and backs off politely to avoid overloading
the source server.
"""

import argparse
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
BASE_URL = "https://www.tennisabstract.com"
PLAYER_STATS_URL = f"{BASE_URL}/cgi-bin/player-classic.cgi"
MAX_RETRIES = 4
RETRY_BACKOFF = 3  # seconds
REQUEST_DELAY = 1.5  # polite delay between requests

STAT_COLUMNS = [
    "match_id",
    "player_name",
    "opponent_name",
    "tournament",
    "surface",
    "date",
    "round",
    "result",
    "sets_won",
    "sets_lost",
    "aces",
    "double_faults",
    "first_serve_pct",
    "first_serve_won_pct",
    "second_serve_won_pct",
    "break_points_saved",
    "break_points_faced",
    "return_points_won_pct",
]


def _get_html(url: str, params: dict | None = None, retries: int = MAX_RETRIES) -> str | None:
    """Fetch raw HTML, retrying on transient errors. Returns None on failure."""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.text
            if response.status_code in {403, 404}:
                logger.warning("HTTP %s for %s — stopping retries.", response.status_code, url)
                return None
            logger.warning("HTTP %s (attempt %d/%d): %s", response.status_code, attempt, retries, url)
        except requests.RequestException as exc:
            logger.warning("Request error (attempt %d/%d): %s — %s", attempt, retries, url, exc)
        if attempt < retries:
            time.sleep(RETRY_BACKOFF * attempt)
    return None


def parse_player_match_stats(html: str, player_name: str) -> pd.DataFrame:
    """Parse TennisAbstract match-stats table into a DataFrame."""
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    # TennisAbstract uses a table with id="matchstats" or class patterns
    table = soup.find("table", {"id": "matchstats"}) or soup.find(
        "table", class_=re.compile(r"match", re.I)
    )
    if table is None:
        logger.debug("No match stats table found for %s", player_name)
        return pd.DataFrame(columns=STAT_COLUMNS)

    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    for tr in table.find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) < 5:
            continue
        row = dict(zip(headers, cells))
        row["player_name"] = player_name
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def scrape_player_stats(player_id: str, player_name: str) -> pd.DataFrame:
    """Fetch all match stats for a single player from TennisAbstract."""
    params = {"p": player_id}
    html = _get_html(PLAYER_STATS_URL, params=params)
    if html is None:
        logger.error("Could not fetch stats for player %s (%s)", player_name, player_id)
        return pd.DataFrame(columns=STAT_COLUMNS)
    df = parse_player_match_stats(html, player_name)
    logger.info("Parsed %d match rows for %s", len(df), player_name)
    return df


def load_player_list(atp_players_csv: Path) -> list[tuple[str, str]]:
    """Load the Jeff Sackmann player index and return (player_id, full_name) tuples."""
    if not atp_players_csv.exists():
        logger.warning("Player index not found at %s — run scraper_atp.py first.", atp_players_csv)
        return []
    df = pd.read_csv(atp_players_csv, header=None,
                     names=["player_id", "first_name", "last_name", "hand", "dob", "ioc"])
    df["full_name"] = df["first_name"].fillna("") + " " + df["last_name"].fillna("")
    df["full_name"] = df["full_name"].str.strip()
    return list(zip(df["player_id"].astype(str), df["full_name"]))


def scrape_all_players(output_dir: Path | None = None) -> None:
    """Iterate over all known ATP players and scrape their match stats."""
    output_dir = output_dir or RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    players_csv = RAW_DATA_DIR / "atp_players.csv"
    players = load_player_list(players_csv)
    if not players:
        logger.warning("No players to scrape. Exiting.")
        return

    for player_id, player_name in tqdm(players, desc="Scraping player stats"):
        output_file = output_dir / f"player_stats_{player_id}.csv"
        if output_file.exists():
            continue  # already cached
        df = scrape_player_stats(player_id, player_name)
        if not df.empty:
            df.to_csv(output_file, index=False)
        time.sleep(REQUEST_DELAY)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape per-player match statistics.")
    parser.add_argument(
        "--player-id",
        help="Scrape a single player by TennisAbstract player ID (optional).",
        default=None,
    )
    parser.add_argument(
        "--player-name",
        help="Player name (used with --player-id).",
        default="Unknown",
    )
    args = parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.player_id:
        logger.info("Scraping single player: %s (%s)", args.player_name, args.player_id)
        df = scrape_player_stats(args.player_id, args.player_name)
        if not df.empty:
            out = RAW_DATA_DIR / f"player_stats_{args.player_id}.csv"
            df.to_csv(out, index=False)
            logger.info("Saved to %s", out)
    else:
        logger.info("Scraping all players from player index…")
        scrape_all_players()


if __name__ == "__main__":
    main()
