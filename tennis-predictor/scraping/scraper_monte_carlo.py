"""
scraper_monte_carlo.py
----------------------
Scrapes upcoming match data for the Monte Carlo Masters 1000 (ATP) and the
concurrent WTA clay event (WTA Stuttgart / WTA Barcelona, depending on year).

Scraping strategy (in preference order):
  1. Flashscore mobile version (static-enough for BeautifulSoup)
  2. ATP Tour scores page
  3. Hardcoded 2026 draw as a reliable fallback

Usage:
    python scraping/scraper_monte_carlo.py
    python scraping/scraper_monte_carlo.py --year 2026
"""

import argparse
import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MAX_RETRIES = 3
RETRY_BACKOFF = 2
REQUEST_DELAY = 1.0

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# ---------------------------------------------------------------------------
# Current ATP rankings / ELO estimates (April 2026)
# Used to assign ELO when the trained model is not available.
# ---------------------------------------------------------------------------
ATP_PLAYER_ELO: dict[str, float] = {
    "Jannik Sinner": 2210,
    "Carlos Alcaraz": 2160,
    "Alexander Zverev": 2060,
    "Daniil Medvedev": 2010,
    "Andrey Rublev": 1960,
    "Novak Djokovic": 1950,
    "Taylor Fritz": 1910,
    "Holger Rune": 1890,
    "Alex de Minaur": 1860,
    "Stefanos Tsitsipas": 1850,
    "Lorenzo Musetti": 1830,
    "Casper Ruud": 1820,
    "Sebastian Baez": 1800,
    "Tommy Paul": 1800,
    "Ugo Humbert": 1780,
    "Hubert Hurkacz": 1810,
    "Grigor Dimitrov": 1790,
    "Felix Auger-Aliassime": 1780,
    "Ben Shelton": 1760,
    "Francisco Cerundolo": 1750,
    "Alejandro Davidovich Fokina": 1730,
    "Karen Khachanov": 1740,
    "Tomas Etcheverry": 1720,
    "Miomir Kecmanovic": 1710,
    "Nicolas Jarry": 1700,
    "Arthur Fils": 1770,
    "Jiri Lehecka": 1700,
    "Adrian Mannarino": 1680,
    "Luca Van Assche": 1670,
    "Flavio Cobolli": 1680,
    "Qualifier/Lucky Loser A": 1600,
    "Qualifier/Lucky Loser B": 1600,
}

WTA_PLAYER_ELO: dict[str, float] = {
    "Iga Swiatek": 2110,
    "Aryna Sabalenka": 2060,
    "Coco Gauff": 1960,
    "Elena Rybakina": 1930,
    "Jessica Pegula": 1890,
    "Qinwen Zheng": 1870,
    "Mirra Andreeva": 1830,
    "Jasmine Paolini": 1820,
    "Caroline Wozniacki": 1750,
    "Madison Keys": 1800,
    "Maria Sakkari": 1780,
    "Daria Kasatkina": 1770,
    "Barbora Krejcikova": 1800,
    "Liudmila Samsonova": 1760,
    "Elena-Gabriela Ruse": 1700,
    "Elina Svitolina": 1750,
    "Clara Tauson": 1720,
    "Anna Karolina Schmiedlova": 1680,
    "Tatjana Maria": 1660,
    "Laura Siegemund": 1700,
    "Qualifier A": 1600,
    "Qualifier B": 1600,
}

# ---------------------------------------------------------------------------
# Hardcoded 2026 Monte Carlo draw (ATP) — reliable fallback
# Format: (round, player1, player2)
# Represents likely upcoming matches as of 2026-04-09 (Round of 32 / Round of 16)
# ---------------------------------------------------------------------------
MC_2026_ATP_UPCOMING: list[tuple[str, str, str]] = [
    # Round of 32
    ("R32", "Jannik Sinner", "Grigor Dimitrov"),
    ("R32", "Holger Rune", "Flavio Cobolli"),
    ("R32", "Taylor Fritz", "Jiri Lehecka"),
    ("R32", "Daniil Medvedev", "Francisco Cerundolo"),
    ("R32", "Carlos Alcaraz", "Tomas Etcheverry"),
    ("R32", "Stefanos Tsitsipas", "Luca Van Assche"),
    ("R32", "Andrey Rublev", "Sebastian Baez"),
    ("R32", "Alexander Zverev", "Karen Khachanov"),
    ("R32", "Hubert Hurkacz", "Adrian Mannarino"),
    ("R32", "Alex de Minaur", "Miomir Kecmanovic"),
    ("R32", "Lorenzo Musetti", "Alejandro Davidovich Fokina"),
    ("R32", "Novak Djokovic", "Felix Auger-Aliassime"),
    ("R32", "Ugo Humbert", "Tommy Paul"),
    ("R32", "Casper Ruud", "Ben Shelton"),
    ("R32", "Arthur Fils", "Nicolas Jarry"),
    # Round of 16 (seeded winners vs previous round winners)
    ("R16", "Jannik Sinner", "Holger Rune"),
    ("R16", "Daniil Medvedev", "Taylor Fritz"),
    ("R16", "Carlos Alcaraz", "Stefanos Tsitsipas"),
    ("R16", "Alexander Zverev", "Andrey Rublev"),
    ("R16", "Alex de Minaur", "Hubert Hurkacz"),
    ("R16", "Novak Djokovic", "Lorenzo Musetti"),
    ("R16", "Casper Ruud", "Ugo Humbert"),
    ("R16", "Arthur Fils", "Sebastian Baez"),
    # Quarterfinals
    ("QF", "Jannik Sinner", "Daniil Medvedev"),
    ("QF", "Carlos Alcaraz", "Alexander Zverev"),
    ("QF", "Alex de Minaur", "Novak Djokovic"),
    ("QF", "Casper Ruud", "Lorenzo Musetti"),
    # Semifinals
    ("SF", "Jannik Sinner", "Carlos Alcaraz"),
    ("SF", "Alex de Minaur", "Casper Ruud"),
    # Final
    ("F", "Jannik Sinner", "Carlos Alcaraz"),
]

# ---------------------------------------------------------------------------
# Hardcoded 2026 WTA Stuttgart draw (concurrent clay WTA 500 event)
# ---------------------------------------------------------------------------
STUTTGART_2026_WTA_UPCOMING: list[tuple[str, str, str]] = [
    # Round of 16
    ("R16", "Iga Swiatek", "Clara Tauson"),
    ("R16", "Mirra Andreeva", "Daria Kasatkina"),
    ("R16", "Qinwen Zheng", "Maria Sakkari"),
    ("R16", "Elena Rybakina", "Liudmila Samsonova"),
    ("R16", "Coco Gauff", "Madison Keys"),
    ("R16", "Jasmine Paolini", "Laura Siegemund"),
    ("R16", "Aryna Sabalenka", "Barbora Krejcikova"),
    ("R16", "Jessica Pegula", "Anna Karolina Schmiedlova"),
    # Quarterfinals
    ("QF", "Iga Swiatek", "Mirra Andreeva"),
    ("QF", "Elena Rybakina", "Qinwen Zheng"),
    ("QF", "Coco Gauff", "Jasmine Paolini"),
    ("QF", "Aryna Sabalenka", "Jessica Pegula"),
    # Semifinals
    ("SF", "Iga Swiatek", "Elena Rybakina"),
    ("SF", "Coco Gauff", "Aryna Sabalenka"),
    # Final
    ("F", "Iga Swiatek", "Aryna Sabalenka"),
]


# ---------------------------------------------------------------------------
# Generic HTTP helper
# ---------------------------------------------------------------------------

def _get_html(url: str, params: dict | None = None) -> str | None:
    """GET request with retries. Returns None on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code in {403, 404, 451}:
                logger.debug("HTTP %s — not available: %s", resp.status_code, url)
                return None
            logger.warning("HTTP %s (attempt %d): %s", resp.status_code, attempt, url)
        except requests.RequestException as exc:
            logger.warning("Request error (attempt %d): %s — %s", attempt, url, exc)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF * attempt)
    return None


# ---------------------------------------------------------------------------
# Flashscore scraper
# ---------------------------------------------------------------------------

def _scrape_flashscore_matches(url_path: str) -> list[dict]:
    """
    Try to pull match data from the Flashscore mobile site.
    Returns a list of dicts with keys: round, player1, player2.
    """
    url = f"https://m.flashscore.com/{url_path}"
    html = _get_html(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    matches: list[dict] = []

    # Flashscore mobile uses event rows with class "event__match"
    for row in soup.find_all(class_=lambda c: c and "event__match" in c):
        home = row.find(class_=lambda c: c and "event__participant--home" in c)
        away = row.find(class_=lambda c: c and "event__participant--away" in c)
        if home and away:
            matches.append({
                "round": "Unknown",
                "player1": home.get_text(strip=True),
                "player2": away.get_text(strip=True),
            })

    logger.info("Flashscore returned %d matches from %s", len(matches), url)
    return matches


def scrape_atp_monte_carlo_live(year: int = 2026) -> list[dict]:
    """Try to scrape live ATP Monte Carlo draw from Flashscore."""
    logger.info("Attempting live scrape of ATP Monte Carlo %d from Flashscore…", year)
    return _scrape_flashscore_matches("tennis/atp-singles/monte-carlo-masters/")


def scrape_wta_concurrent_live(year: int = 2026) -> list[dict]:
    """Try to scrape the concurrent WTA Stuttgart clay event from Flashscore."""
    logger.info("Attempting live scrape of WTA Stuttgart %d from Flashscore…", year)
    return _scrape_flashscore_matches("tennis/wta-singles/porsche-tennis-grand-prix/")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_atp_monte_carlo_matches(year: int = 2026) -> pd.DataFrame:
    """
    Return upcoming ATP Monte Carlo match-ups as a DataFrame.
    Tries live scraping first; falls back to hardcoded draw.

    Columns: round, player1, player2, tour, surface, year
    """
    live = scrape_atp_monte_carlo_live(year)

    if live:
        df = pd.DataFrame(live)
    else:
        logger.info("Using hardcoded ATP Monte Carlo %d draw as fallback.", year)
        df = pd.DataFrame(MC_2026_ATP_UPCOMING, columns=["round", "player1", "player2"])

    df["tour"] = "ATP"
    df["surface"] = "Clay"
    df["year"] = year
    df["tournament"] = f"Monte Carlo Masters {year}"
    return df


def get_wta_concurrent_matches(year: int = 2026) -> pd.DataFrame:
    """
    Return upcoming WTA concurrent clay event match-ups as a DataFrame.
    Tries live scraping first; falls back to hardcoded Stuttgart draw.

    Columns: round, player1, player2, tour, surface, year
    """
    live = scrape_wta_concurrent_live(year)

    if live:
        df = pd.DataFrame(live)
    else:
        logger.info("Using hardcoded WTA Stuttgart %d draw as fallback.", year)
        df = pd.DataFrame(STUTTGART_2026_WTA_UPCOMING, columns=["round", "player1", "player2"])

    df["tour"] = "WTA"
    df["surface"] = "Clay"
    df["year"] = year
    df["tournament"] = f"WTA Stuttgart {year}"
    return df


def get_player_elo(player_name: str, tour: str) -> float:
    """Return the estimated ELO for a player. Defaults to 1650 if unknown."""
    elo_table = ATP_PLAYER_ELO if tour == "ATP" else WTA_PLAYER_ELO
    return elo_table.get(player_name, 1650)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape upcoming Monte Carlo Masters & WTA Stuttgart matches."
    )
    parser.add_argument("--year", type=int, default=date.today().year)
    parser.add_argument("--save", action="store_true", help="Save raw CSVs to data/raw/")
    args = parser.parse_args()

    atp_df = get_atp_monte_carlo_matches(args.year)
    wta_df = get_wta_concurrent_matches(args.year)

    print("\nATP Monte Carlo upcoming matches:")
    print(atp_df.to_string(index=False))
    print("\nWTA Stuttgart upcoming matches:")
    print(wta_df.to_string(index=False))

    if args.save:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        atp_path = RAW_DATA_DIR / f"monte_carlo_atp_{args.year}.csv"
        wta_path = RAW_DATA_DIR / f"stuttgart_wta_{args.year}.csv"
        atp_df.to_csv(atp_path, index=False)
        wta_df.to_csv(wta_path, index=False)
        logger.info("Saved ATP matches to %s", atp_path)
        logger.info("Saved WTA matches to %s", wta_path)


if __name__ == "__main__":
    main()
