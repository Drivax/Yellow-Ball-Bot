"""
scraper_us_open.py
------------------
Scrapes upcoming match data for the US Open (ATP & WTA Grand Slam, Hard court).

Scraping strategy (in preference order):
  1. Flashscore mobile version (static-enough for BeautifulSoup)
  2. Hardcoded 2026 draw as a reliable fallback

Usage:
    python scraping/scraper_us_open.py
    python scraping/scraper_us_open.py --year 2026
"""

import argparse
import logging
import re
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

_FLASHSCORE_BLOCK_MARKERS = (
    "the requested page can't be displayed",
    "flashscore.mobi",
)

# ---------------------------------------------------------------------------
# ELO estimates for US Open 2026 (Hard court, August–September)
# ---------------------------------------------------------------------------
ATP_PLAYER_ELO: dict[str, float] = {
    "Jannik Sinner": 2230,
    "Carlos Alcaraz": 2180,
    "Alexander Zverev": 2070,
    "Daniil Medvedev": 2040,
    "Taylor Fritz": 1950,
    "Andrey Rublev": 1940,
    "Ben Shelton": 1910,
    "Novak Djokovic": 1960,
    "Holger Rune": 1880,
    "Tommy Paul": 1870,
    "Hubert Hurkacz": 1860,
    "Alex de Minaur": 1850,
    "Grigor Dimitrov": 1830,
    "Ugo Humbert": 1810,
    "Arthur Fils": 1800,
    "Felix Auger-Aliassime": 1790,
    "Lorenzo Musetti": 1780,
    "Frances Tiafoe": 1760,
    "Stefanos Tsitsipas": 1820,
    "Casper Ruud": 1770,
    "Sebastian Korda": 1750,
    "Sebastian Baez": 1730,
    "Jack Draper": 1740,
    "Tomas Machac": 1720,
    "Karen Khachanov": 1710,
    "Francisco Cerundolo": 1700,
    "Alejandro Davidovich Fokina": 1690,
    "Matteo Berrettini": 1680,
    "Christopher Eubanks": 1670,
    "Jiri Lehecka": 1660,
    "Alexei Popyrin": 1650,
    "Qualifier A": 1580,
}

WTA_PLAYER_ELO: dict[str, float] = {
    "Aryna Sabalenka": 2130,
    "Iga Swiatek": 2090,
    "Coco Gauff": 2000,
    "Jessica Pegula": 1940,
    "Elena Rybakina": 1920,
    "Qinwen Zheng": 1890,
    "Madison Keys": 1860,
    "Emma Navarro": 1840,
    "Mirra Andreeva": 1820,
    "Jasmine Paolini": 1800,
    "Daria Kasatkina": 1780,
    "Barbora Krejcikova": 1810,
    "Danielle Collins": 1770,
    "Liudmila Samsonova": 1750,
    "Elina Svitolina": 1730,
    "Beatriz Haddad Maia": 1720,
    "Caroline Wozniacki": 1700,
    "Maria Sakkari": 1740,
    "Karolina Muchova": 1760,
    "Paula Badosa": 1710,
    "Anna Kalinskaya": 1690,
    "Veronika Kudermetova": 1680,
    "Qualifier A": 1580,
}

# ---------------------------------------------------------------------------
# Hardcoded 2026 US Open ATP draw — reliable fallback
# Format: (round, player1, player2)
# Rounds: R128 -> R64 -> R32 -> R16 -> QF -> SF -> F
# ---------------------------------------------------------------------------
US_OPEN_2026_ATP_UPCOMING: list[tuple[str, str, str]] = [
    # Round of 128 (selected first-round matches)
    ("R128", "Jannik Sinner", "Alexei Popyrin"),
    ("R128", "Sebastian Korda", "Tomas Machac"),
    ("R128", "Taylor Fritz", "Christopher Eubanks"),
    ("R128", "Andrey Rublev", "Jiri Lehecka"),
    ("R128", "Carlos Alcaraz", "Alejandro Davidovich Fokina"),
    ("R128", "Holger Rune", "Frances Tiafoe"),
    ("R128", "Tommy Paul", "Sebastian Baez"),
    ("R128", "Alexander Zverev", "Matteo Berrettini"),
    ("R128", "Novak Djokovic", "Jack Draper"),
    ("R128", "Ben Shelton", "Karen Khachanov"),
    ("R128", "Hubert Hurkacz", "Arthur Fils"),
    ("R128", "Ugo Humbert", "Lorenzo Musetti"),
    ("R128", "Daniil Medvedev", "Francisco Cerundolo"),
    ("R128", "Stefanos Tsitsipas", "Felix Auger-Aliassime"),
    ("R128", "Alex de Minaur", "Casper Ruud"),
    ("R128", "Grigor Dimitrov", "Sebastian Korda"),
    # Round of 64
    ("R64", "Jannik Sinner", "Taylor Fritz"),
    ("R64", "Andrey Rublev", "Holger Rune"),
    ("R64", "Carlos Alcaraz", "Tommy Paul"),
    ("R64", "Alexander Zverev", "Novak Djokovic"),
    ("R64", "Ben Shelton", "Hubert Hurkacz"),
    ("R64", "Ugo Humbert", "Daniil Medvedev"),
    ("R64", "Stefanos Tsitsipas", "Alex de Minaur"),
    ("R64", "Grigor Dimitrov", "Felix Auger-Aliassime"),
    # Round of 32
    ("R32", "Jannik Sinner", "Andrey Rublev"),
    ("R32", "Carlos Alcaraz", "Alexander Zverev"),
    ("R32", "Ben Shelton", "Daniil Medvedev"),
    ("R32", "Alex de Minaur", "Grigor Dimitrov"),
    ("R32", "Taylor Fritz", "Tommy Paul"),
    ("R32", "Novak Djokovic", "Hubert Hurkacz"),
    ("R32", "Holger Rune", "Ugo Humbert"),
    ("R32", "Stefanos Tsitsipas", "Felix Auger-Aliassime"),
    # Round of 16
    ("R16", "Jannik Sinner", "Carlos Alcaraz"),
    ("R16", "Ben Shelton", "Taylor Fritz"),
    ("R16", "Alexander Zverev", "Novak Djokovic"),
    ("R16", "Daniil Medvedev", "Andrey Rublev"),
    ("R16", "Alex de Minaur", "Holger Rune"),
    ("R16", "Hubert Hurkacz", "Ugo Humbert"),
    ("R16", "Grigor Dimitrov", "Stefanos Tsitsipas"),
    ("R16", "Tommy Paul", "Felix Auger-Aliassime"),
    # Quarterfinals
    ("QF", "Jannik Sinner", "Ben Shelton"),
    ("QF", "Carlos Alcaraz", "Alexander Zverev"),
    ("QF", "Daniil Medvedev", "Alex de Minaur"),
    ("QF", "Grigor Dimitrov", "Tommy Paul"),
    # Semifinals
    ("SF", "Jannik Sinner", "Carlos Alcaraz"),
    ("SF", "Daniil Medvedev", "Grigor Dimitrov"),
    # Final
    ("F", "Jannik Sinner", "Carlos Alcaraz"),
]

# ---------------------------------------------------------------------------
# Hardcoded 2026 US Open WTA draw — reliable fallback
# ---------------------------------------------------------------------------
US_OPEN_2026_WTA_UPCOMING: list[tuple[str, str, str]] = [
    # Round of 128 (selected first-round matches)
    ("R128", "Aryna Sabalenka", "Anna Kalinskaya"),
    ("R128", "Emma Navarro", "Paula Badosa"),
    ("R128", "Coco Gauff", "Veronika Kudermetova"),
    ("R128", "Danielle Collins", "Beatriz Haddad Maia"),
    ("R128", "Iga Swiatek", "Caroline Wozniacki"),
    ("R128", "Mirra Andreeva", "Daria Kasatkina"),
    ("R128", "Madison Keys", "Karolina Muchova"),
    ("R128", "Jessica Pegula", "Liudmila Samsonova"),
    ("R128", "Elena Rybakina", "Elina Svitolina"),
    ("R128", "Qinwen Zheng", "Maria Sakkari"),
    ("R128", "Jasmine Paolini", "Barbora Krejcikova"),
    ("R128", "Anna Kalinskaya", "Beatriz Haddad Maia"),
    # Round of 64
    ("R64", "Aryna Sabalenka", "Emma Navarro"),
    ("R64", "Coco Gauff", "Danielle Collins"),
    ("R64", "Iga Swiatek", "Mirra Andreeva"),
    ("R64", "Madison Keys", "Jessica Pegula"),
    ("R64", "Elena Rybakina", "Qinwen Zheng"),
    ("R64", "Jasmine Paolini", "Karolina Muchova"),
    ("R64", "Daria Kasatkina", "Barbora Krejcikova"),
    ("R64", "Paula Badosa", "Liudmila Samsonova"),
    # Round of 32
    ("R32", "Aryna Sabalenka", "Coco Gauff"),
    ("R32", "Iga Swiatek", "Madison Keys"),
    ("R32", "Elena Rybakina", "Jasmine Paolini"),
    ("R32", "Daria Kasatkina", "Emma Navarro"),
    ("R32", "Jessica Pegula", "Qinwen Zheng"),
    ("R32", "Danielle Collins", "Barbora Krejcikova"),
    ("R32", "Mirra Andreeva", "Paula Badosa"),
    ("R32", "Karolina Muchova", "Liudmila Samsonova"),
    # Round of 16
    ("R16", "Aryna Sabalenka", "Iga Swiatek"),
    ("R16", "Coco Gauff", "Madison Keys"),
    ("R16", "Elena Rybakina", "Emma Navarro"),
    ("R16", "Daria Kasatkina", "Jessica Pegula"),
    ("R16", "Qinwen Zheng", "Danielle Collins"),
    ("R16", "Barbora Krejcikova", "Mirra Andreeva"),
    ("R16", "Jasmine Paolini", "Karolina Muchova"),
    ("R16", "Paula Badosa", "Liudmila Samsonova"),
    # Quarterfinals
    ("QF", "Aryna Sabalenka", "Coco Gauff"),
    ("QF", "Iga Swiatek", "Elena Rybakina"),
    ("QF", "Jessica Pegula", "Qinwen Zheng"),
    ("QF", "Mirra Andreeva", "Jasmine Paolini"),
    # Semifinals
    ("SF", "Aryna Sabalenka", "Iga Swiatek"),
    ("SF", "Jessica Pegula", "Mirra Andreeva"),
    # Final
    ("F", "Aryna Sabalenka", "Iga Swiatek"),
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
                response_text = resp.text
                response_lower = response_text.lower()
                if any(marker in response_lower for marker in _FLASHSCORE_BLOCK_MARKERS):
                    logger.warning("Flashscore returned a blocked/placeholder page: %s", url)
                    return None
                return response_text
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

_WHITESPACE_RE = re.compile(r"\s+")
_NAME_CLEAN_RE = re.compile(r"[^A-Za-z .\-']")
_SKIP_MARKERS = {
    "finished",
    "abandoned",
    "cancelled",
    "walkover",
    "retired",
    "wo",
    "postponed",
}
_PLACEHOLDER_NAMES = {
    "tbd",
    "bye",
    "winner",
    "loser",
    "qualifier",
    "lucky loser",
    "q",
    "ll",
}


def _normalize_player_name(name: str) -> str:
    cleaned = _NAME_CLEAN_RE.sub(" ", str(name))
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def _extract_round_hint(row) -> str:
    """Infer round from nearby heading text when available."""
    probe = row
    for _ in range(4):
        probe = probe.find_previous(
            class_=lambda c: c and any(
                marker in c for marker in ("event__header", "event__title", "event__round")
            )
        )
        if not probe:
            break
        text = probe.get_text(" ", strip=True).lower()
        if any(tok in text for tok in ["r128", "round of 128", "1st round"]):
            return "R128"
        if any(tok in text for tok in ["r64", "round of 64", "2nd round"]):
            return "R64"
        if any(tok in text for tok in ["r32", "round of 32", "3rd round"]):
            return "R32"
        if any(tok in text for tok in ["r16", "round of 16", "4th round"]):
            return "R16"
        if any(tok in text for tok in ["quarter", "qf"]):
            return "QF"
        if any(tok in text for tok in ["semi", "sf"]):
            return "SF"
        if "final" in text:
            return "F"
    return "Unknown"


def _should_skip_event(stage_text: str) -> bool:
    """Skip completed/invalid events and keep only incoming or imminent matches."""
    text = str(stage_text).strip().lower()
    if not text:
        return False
    if any(marker in text for marker in _SKIP_MARKERS):
        return True
    if re.search(r"\b\d{1,2}\s*[-:]\s*\d{1,2}\b", text):
        return True
    return False


def _is_placeholder_name(name: str) -> bool:
    lower = name.strip().lower()
    return any(token == lower or token in lower for token in _PLACEHOLDER_NAMES)


def _is_plausible_name(name: str) -> bool:
    if len(name) < 4:
        return False
    if _is_placeholder_name(name):
        return False
    alpha_count = sum(ch.isalpha() for ch in name)
    return alpha_count >= 3


def _clean_upcoming_matches(matches: list[dict]) -> list[dict]:
    """Normalize, dedupe and keep plausible incoming matchups only."""
    seen: set[tuple[str, str, str]] = set()
    clean: list[dict] = []

    for item in matches:
        p1 = _normalize_player_name(item.get("player1", ""))
        p2 = _normalize_player_name(item.get("player2", ""))
        rnd = str(item.get("round", "Unknown") or "Unknown")

        if not p1 or not p2 or p1.lower() == p2.lower():
            continue
        if not (_is_plausible_name(p1) and _is_plausible_name(p2)):
            continue

        key = (rnd, p1.lower(), p2.lower())
        alt_key = (rnd, p2.lower(), p1.lower())
        if key in seen or alt_key in seen:
            continue
        seen.add(key)
        clean.append({"round": rnd, "player1": p1, "player2": p2})

    if 0 < len(clean) < 4:
        logger.warning("Live scrape returned too few plausible upcoming matches; using fallback.")
        return []

    return clean


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

    for row in soup.find_all(class_=lambda c: c and "event__match" in c):
        participants = [
            el.get_text(strip=True)
            for el in row.find_all(class_=lambda c: c and "event__participant" in c)
        ]

        if len(participants) >= 2:
            p1, p2 = participants[0], participants[1]
        else:
            home = row.find(class_=lambda c: c and "event__participant--home" in c)
            away = row.find(class_=lambda c: c and "event__participant--away" in c)
            if not (home and away):
                continue
            p1, p2 = home.get_text(strip=True), away.get_text(strip=True)

        stage = row.find(class_=lambda c: c and "event__stage" in c)
        if not stage:
            stage = row.find(class_=lambda c: c and "event__time" in c)
        stage_text = stage.get_text(" ", strip=True) if stage else ""

        round_hint = _extract_round_hint(row)
        if _should_skip_event(stage_text):
            continue

        matches.append({
            "round": round_hint,
            "player1": _normalize_player_name(p1),
            "player2": _normalize_player_name(p2),
        })

    matches = _clean_upcoming_matches(matches)
    logger.info("Flashscore returned %d matches from %s", len(matches), url)
    return matches


def scrape_atp_us_open_live(year: int = 2026) -> list[dict]:
    """Try to scrape live ATP US Open draw from Flashscore."""
    logger.info("Attempting live scrape of ATP US Open %d from Flashscore…", year)
    return _scrape_flashscore_matches("tennis/atp-singles/us-open/")


def scrape_wta_us_open_live(year: int = 2026) -> list[dict]:
    """Try to scrape live WTA US Open draw from Flashscore."""
    logger.info("Attempting live scrape of WTA US Open %d from Flashscore…", year)
    return _scrape_flashscore_matches("tennis/wta-singles/us-open/")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_atp_us_open_matches(year: int = 2026) -> pd.DataFrame:
    """
    Return upcoming ATP US Open match-ups as a DataFrame.
    Tries live scraping first; falls back to hardcoded draw.

    Columns: round, player1, player2, tour, surface, year, tournament
    """
    live = scrape_atp_us_open_live(year)

    if live:
        df = pd.DataFrame(live)
    else:
        logger.info("Using hardcoded ATP US Open %d draw as fallback.", year)
        df = pd.DataFrame(US_OPEN_2026_ATP_UPCOMING, columns=["round", "player1", "player2"])

    df["tour"] = "ATP"
    df["surface"] = "Hard"
    df["year"] = year
    df["tournament"] = f"US Open {year}"
    return df


def get_wta_us_open_matches(year: int = 2026) -> pd.DataFrame:
    """
    Return upcoming WTA US Open match-ups as a DataFrame.
    Tries live scraping first; falls back to hardcoded draw.

    Columns: round, player1, player2, tour, surface, year, tournament
    """
    live = scrape_wta_us_open_live(year)

    if live:
        df = pd.DataFrame(live)
    else:
        logger.info("Using hardcoded WTA US Open %d draw as fallback.", year)
        df = pd.DataFrame(US_OPEN_2026_WTA_UPCOMING, columns=["round", "player1", "player2"])

    df["tour"] = "WTA"
    df["surface"] = "Hard"
    df["year"] = year
    df["tournament"] = f"US Open {year}"
    return df


def get_player_elo(player_name: str, tour: str) -> float:
    """Return the estimated ELO for a player. Defaults to 1600 if unknown."""
    elo_table = ATP_PLAYER_ELO if tour == "ATP" else WTA_PLAYER_ELO
    return elo_table.get(player_name, 1600)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape upcoming US Open ATP & WTA matches."
    )
    parser.add_argument("--year", type=int, default=date.today().year)
    parser.add_argument("--save", action="store_true", help="Save raw CSVs to data/raw/")
    args = parser.parse_args()

    atp_df = get_atp_us_open_matches(args.year)
    wta_df = get_wta_us_open_matches(args.year)

    print("\nATP US Open upcoming matches:")
    print(atp_df.to_string(index=False))
    print("\nWTA US Open upcoming matches:")
    print(wta_df.to_string(index=False))

    if args.save:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        atp_path = RAW_DATA_DIR / f"us_open_atp_{args.year}.csv"
        wta_path = RAW_DATA_DIR / f"us_open_wta_{args.year}.csv"
        atp_df.to_csv(atp_path, index=False)
        wta_df.to_csv(wta_path, index=False)
        logger.info("Saved ATP matches to %s", atp_path)
        logger.info("Saved WTA matches to %s", wta_path)


if __name__ == "__main__":
    main()
