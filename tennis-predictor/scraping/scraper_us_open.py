"""Upcoming US Open singles from ESPN's public scoreboard.

Only confirmed, unstarted matches in the requested edition are returned.
An unavailable source raises ScheduleUnavailable; an empty schedule is valid.
No invented draw or stale cache is substituted for live data.
"""
import argparse
from datetime import date, datetime, timezone
import logging
import re
import unicodedata

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/tennis/{tour}/scoreboard"
COLUMNS = ["match_id", "round", "player1", "player2", "tour", "surface", "year",
           "tournament", "scheduled_at", "time_confirmed", "status", "source", "fetched_at"]
ROUNDS = {"1st Round": "R128", "2nd Round": "R64", "3rd Round": "R32",
          "Round 1": "R128", "Round 2": "R64", "Round 3": "R32", "Round 4": "R16",
          "4th Round": "R16", "Quarterfinal": "QF", "Semifinal": "SF", "Final": "F"}


class ScheduleUnavailable(RuntimeError):
    """The provider could not supply a valid schedule."""


def name_key(name):
    normalized = " ".join(unicodedata.normalize("NFKD", name).encode("ascii", "ignore")
                          .decode().casefold().replace("-", " ").split())
    return {"zheng qinwen": "qinwen zheng"}.get(normalized, normalized)


def _is_plausible_name(name):
    # Match complete placeholder words, never the letters q/ll within real names.
    return (isinstance(name, str) and sum(c.isalpha() for c in name) >= 3
            and not re.search(r"\b(tbd|bye|winner|loser|qualifier|unknown|q|ll)\b", name, re.I))


def parse_scoreboard(payload, tour, year, now=None, source="ESPN"):
    """Parse the observed tournament -> groupings -> competitions schema."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must include a timezone")
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise ScheduleUnavailable("Unexpected ESPN scoreboard schema")
    slug = {"ATP": "mens-singles", "WTA": "womens-singles"}[tour]
    rows, seen = [], set()
    completed, completed_ids = [], set()
    for event in payload["events"]:
        if event.get("name", "").casefold() not in {"us open", "u.s. open"}:
            continue
        if str(event.get("season", {}).get("year")) != str(year):
            continue
        groups = event.get("groupings")
        if not isinstance(groups, list):
            raise ScheduleUnavailable("Missing US Open groupings")
        for group in groups:
            if group.get("grouping", {}).get("slug") != slug:
                continue
            competitions = group.get("competitions")
            if not isinstance(competitions, list):
                raise ScheduleUnavailable("Missing US Open competitions")
            for match in competitions:
                state = match.get("status", {}).get("type", {})
                # Preserve completed main-draw results for a transparent tournament-only Elo.
                if state.get("name") == "STATUS_FINAL" and state.get("completed"):
                    players = match.get("competitors", [])
                    try:
                        played = datetime.fromisoformat(match["date"].replace("Z", "+00:00"))
                        valid_date = played.tzinfo is not None and played.year == year and played < now
                    except (KeyError, ValueError, TypeError):
                        valid_date = False
                    winner = [p for p in players if p.get("winner") is True]
                    loser = [p for p in players if p.get("winner") is False]
                    notes = " ".join(n.get("text", "") for n in match.get("notes", []))
                    mid = str(match.get("id", ""))
                    if (valid_date and len(players) == 2 and len(winner) == len(loser) == 1
                            and match.get("round", {}).get("displayName") in ROUNDS
                            and mid and mid not in completed_ids
                            and not re.search(r"retired|walkover|default|abandon", notes, re.I)):
                        wn = winner[0].get("athlete", {}).get("displayName", "")
                        ln = loser[0].get("athlete", {}).get("displayName", "")
                        if _is_plausible_name(wn) and _is_plausible_name(ln) and name_key(wn) != name_key(ln):
                            completed.append({"winner": wn, "loser": ln, "date": played.isoformat(), "id": mid})
                            completed_ids.add(mid)
                if (state.get("state") != "pre" or state.get("completed")
                        or state.get("name") != "STATUS_SCHEDULED"):
                    continue
                rnd = ROUNDS.get(match.get("round", {}).get("displayName"))
                if rnd is None:  # Exclude qualifying and unknown formats.
                    continue
                players = match.get("competitors", [])
                if len(players) != 2 or any(p.get("type") != "athlete" for p in players):
                    continue
                names = [p.get("athlete", {}).get("displayName", "").strip() for p in players]
                if not all(_is_plausible_name(n) for n in names) or name_key(names[0]) == name_key(names[1]):
                    continue
                try:
                    scheduled = datetime.fromisoformat(match["date"].replace("Z", "+00:00"))
                    if scheduled.tzinfo is None or scheduled.year != year:
                        continue
                except (KeyError, ValueError, TypeError):
                    continue
                time_confirmed = match.get("timeValid") is True
                if time_confirmed and scheduled <= now:
                    continue
                if not time_confirmed and scheduled.date() < now.date():
                    continue
                match_id = str(match.get("id", ""))
                if not match_id or match_id in seen:
                    continue
                seen.add(match_id)
                rows.append(dict(match_id=match_id, round=rnd, player1=names[0], player2=names[1],
                                 tour=tour, surface="Hard", year=year, tournament=f"US Open {year}",
                                 scheduled_at=scheduled.isoformat(), time_confirmed=time_confirmed,
                                 status="scheduled", source=source, fetched_at=now.isoformat()))
    frame = pd.DataFrame(rows, columns=COLUMNS).sort_values("scheduled_at").reset_index(drop=True)
    frame.attrs["completed_results"] = sorted(completed, key=lambda r: (r["date"], r["id"]))
    return frame


def get_us_open_matches(tour, year=None):
    tour = tour.upper()
    if tour not in {"ATP", "WTA"}:
        raise ValueError("tour must be ATP or WTA")
    now = datetime.now(timezone.utc)
    year = now.year if year is None else year
    # During the event query today; otherwise query its final day to obtain the edition.
    query_date = now.date() if now.year == year and now.month in {8, 9} else date(year, 9, 13)
    url = SCOREBOARD_URL.format(tour=tour.lower())
    retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    try:
        with requests.Session() as session:
            session.mount("https://", HTTPAdapter(max_retries=retry))
            response = session.get(url, params={"dates": query_date.strftime("%Y%m%d")}, timeout=(5, 20))
            response.raise_for_status()
            return parse_scoreboard(response.json(), tour, year, now, response.url)
    except (requests.RequestException, ValueError) as exc:
        raise ScheduleUnavailable(f"ESPN schedule unavailable for {tour}: {exc}") from exc


def get_atp_us_open_matches(year=None):
    return get_us_open_matches("ATP", year)


def get_wta_us_open_matches(year=None):
    return get_us_open_matches("WTA", year)


# Legacy hand-entered estimates: not measured 2026 ratings or calibrated probabilities.
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


def get_player_elo(player_name, tour):
    table = ATP_PLAYER_ELO if tour == "ATP" else WTA_PLAYER_ELO
    return {name_key(k): v for k, v in table.items()}.get(name_key(player_name), 1600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=date.today().year)
    args = parser.parse_args()
    for tour in ("ATP", "WTA"):
        print(get_us_open_matches(tour, args.year).to_string(index=False))
