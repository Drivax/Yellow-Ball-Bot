"""
scraper_atp.py
--------------
Downloads all ATP match CSV files from the Jeff Sackmann GitHub repository
(https://github.com/JeffSackmann/tennis_atp) covering 1968-2025 and saves
them as-is to data/raw/.

Usage:
    python scraping/scraper_atp.py
"""

import logging
import time
from pathlib import Path

import requests
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
GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
)
FIRST_YEAR = 1968
LAST_YEAR = 2025
MAX_RETRIES = 5
RETRY_BACKOFF = 2  # seconds


def _download_file(url: str, dest: Path, retries: int = MAX_RETRIES) -> bool:
    """Download *url* to *dest*, retrying on transient failures.

    Returns True on success, False if all retries were exhausted.
    """
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                dest.write_bytes(response.content)
                return True
            if response.status_code == 404:
                logger.debug("Not found (404): %s", url)
                return False
            logger.warning(
                "HTTP %s for %s (attempt %d/%d)",
                response.status_code,
                url,
                attempt,
                retries,
            )
        except requests.RequestException as exc:
            logger.warning(
                "Request error for %s (attempt %d/%d): %s",
                url,
                attempt,
                retries,
                exc,
            )
        if attempt < retries:
            sleep_time = RETRY_BACKOFF * attempt
            logger.debug("Sleeping %ss before retry…", sleep_time)
            time.sleep(sleep_time)
    logger.error("Failed to download after %d attempts: %s", retries, url)
    return False


def download_main_tour_matches() -> None:
    """Download atp_matches_YYYY.csv for every year 1968-2025."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    years = list(range(FIRST_YEAR, LAST_YEAR + 1))
    success, skipped, failed = 0, 0, 0

    for year in tqdm(years, desc="Downloading ATP match files"):
        filename = f"atp_matches_{year}.csv"
        dest = RAW_DATA_DIR / filename
        if dest.exists():
            logger.debug("Already exists, skipping: %s", filename)
            skipped += 1
            continue
        url = f"{GITHUB_RAW_BASE}/{filename}"
        if _download_file(url, dest):
            logger.info("Downloaded: %s", filename)
            success += 1
        else:
            failed += 1
        # Polite rate limit – one request per ~0.3 s
        time.sleep(0.3)

    logger.info(
        "Main tour matches — downloaded: %d, skipped (cached): %d, failed: %d",
        success,
        skipped,
        failed,
    )


def download_futures_and_challengers() -> None:
    """Download atp_matches_futures_YYYY.csv and atp_matches_qual_chall_YYYY.csv."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    prefixes = ["atp_matches_futures", "atp_matches_qual_chall"]
    years = list(range(1991, LAST_YEAR + 1))  # futures data starts ~1991

    for prefix in prefixes:
        for year in tqdm(years, desc=f"Downloading {prefix}"):
            filename = f"{prefix}_{year}.csv"
            dest = RAW_DATA_DIR / filename
            if dest.exists():
                continue
            url = f"{GITHUB_RAW_BASE}/{filename}"
            _download_file(url, dest)
            time.sleep(0.3)


def download_player_index() -> None:
    """Download the master player index (atp_players.csv)."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = "atp_players.csv"
    dest = RAW_DATA_DIR / filename
    if dest.exists():
        logger.info("Player index already cached.")
        return
    url = f"{GITHUB_RAW_BASE}/{filename}"
    if _download_file(url, dest):
        logger.info("Downloaded player index.")
    else:
        logger.error("Could not download player index.")


def download_rankings() -> None:
    """Download ATP ranking snapshots."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Sackmann ships rankings in decade-banded files plus a current file
    ranking_files = (
        [f"atp_rankings_{decade}s.csv" for decade in range(197, 203)]
        + ["atp_rankings_current.csv"]
    )
    for filename in tqdm(ranking_files, desc="Downloading ranking files"):
        dest = RAW_DATA_DIR / filename
        if dest.exists():
            continue
        url = f"{GITHUB_RAW_BASE}/{filename}"
        _download_file(url, dest)
        time.sleep(0.3)


def main() -> None:
    logger.info("=== ATP Scraper Started ===")
    download_player_index()
    download_main_tour_matches()
    download_futures_and_challengers()
    download_rankings()
    logger.info("=== ATP Scraper Finished ===")


if __name__ == "__main__":
    main()
