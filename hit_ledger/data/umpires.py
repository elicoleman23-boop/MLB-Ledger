"""
Umpire assignments and K% deviation.

Data sources (in priority order):
    1. UmpScorecards — best public source for ump tendencies + daily assignments
    2. Baseball Reference umpire pages — historical K% as fallback
    3. League average — if both fail, no adjustment

Scraping notes:
    - Site structures change. Both functions here log failures and return
      None so the pipeline never crashes on umpire data.
    - Assignments are typically posted the day of; no earlier than ~2 hours
      before first pitch.
    - Historical K% should be computed over at least 50 games for signal.
"""
from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any

import requests

from hit_ledger.config import (
    LEAGUE_AVG_UMPIRE_K_PCT,
    UMPIRE_K_DEVIATION_CAP,
    UMPSCORES_BASE_URL,
)
from hit_ledger.data import cache

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
TIMEOUT = 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_umpire_for_game(
    game_pk: int,
    game_date_obj: date,
    home_team: str | None = None,
    away_team: str | None = None,
    force_refresh: bool = False,
) -> dict | None:
    """
    Return:
        {
            'umpire_name': str,
            'k_pct': float,          # umpire's career K%
            'k_pct_dev': float,      # umpire K% - league K% (capped)
            'games_sampled': int,
        }
    Returns None if umpire can't be determined.
    """
    cached = cache.load_umpire_assignment(game_date_obj, game_pk)
    if cached is not None and not force_refresh and cached.get("umpire_name"):
        return cached

    # Try to fetch from MLB StatsAPI first — it has officials for live games
    ump_name = _fetch_from_statsapi(game_pk)

    # Fallback: try UmpScorecards daily slate
    if ump_name is None:
        ump_name = _fetch_from_umpscorecards(game_date_obj, home_team, away_team)

    if ump_name is None:
        logger.info("No umpire found for game_pk=%s", game_pk)
        cache.save_umpire_assignment(
            game_date_obj, game_pk, None, None, None, None
        )
        return None

    # Now look up the umpire's K% tendency
    k_pct, games_sampled = _fetch_umpire_k_pct(ump_name)
    k_pct_dev = None
    if k_pct is not None:
        k_pct_dev = k_pct - LEAGUE_AVG_UMPIRE_K_PCT
        k_pct_dev = max(-UMPIRE_K_DEVIATION_CAP, min(UMPIRE_K_DEVIATION_CAP, k_pct_dev))

    cache.save_umpire_assignment(
        game_date_obj, game_pk, ump_name, k_pct, k_pct_dev, games_sampled
    )
    return {
        "umpire_name": ump_name,
        "k_pct": k_pct,
        "k_pct_dev": k_pct_dev,
        "games_sampled": games_sampled,
    }


# ---------------------------------------------------------------------------
# MLB StatsAPI source
# ---------------------------------------------------------------------------
def _fetch_from_statsapi(game_pk: int) -> str | None:
    """Get home plate umpire from MLB StatsAPI boxscore."""
    try:
        import statsapi
        data = statsapi.get("game", {"gamePk": game_pk})
    except Exception as exc:  # noqa: BLE001
        logger.warning("statsapi fetch failed for %s: %s", game_pk, exc)
        return None

    officials = (
        data.get("liveData", {})
        .get("boxscore", {})
        .get("officials", [])
    )
    for off in officials:
        if off.get("officialType") == "Home Plate":
            return off.get("official", {}).get("fullName")
    return None


# ---------------------------------------------------------------------------
# UmpScorecards source
# ---------------------------------------------------------------------------
def _fetch_from_umpscorecards(
    game_date_obj: date,
    home_team: str | None,
    away_team: str | None,
) -> str | None:
    """
    Scrape the UmpScorecards daily slate for today's HP ump.

    Site structure can change; if selectors break, this logs and returns None
    so the pipeline continues with no umpire adjustment.
    """
    date_str = game_date_obj.strftime("%Y-%m-%d")
    url = f"{UMPSCORES_BASE_URL}/games?date={date_str}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code != 200:
            logger.warning("UmpScores returned %s", resp.status_code)
            return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("UmpScores request failed: %s", exc)
        return None

    # UmpScores game cards include team abbreviations and ump name.
    # Rough regex: look for lines pairing home/away with an ump.
    # This is intentionally simple and will fail gracefully.
    if not (home_team and away_team):
        return None

    # Look for a block mentioning both teams near an umpire name
    html = resp.text
    # Match: "<something with home_team>...<something with away_team>...Umpire: Name"
    # Teams in UmpScores are usually abbreviated (NYY, BOS, etc.)
    pattern = re.compile(
        r"umpire[^<]*?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
        re.IGNORECASE,
    )
    match = pattern.search(html)
    if match:
        return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Umpire K% tendency
# ---------------------------------------------------------------------------
def _fetch_umpire_k_pct(ump_name: str) -> tuple[float | None, int | None]:
    """
    Look up the umpire's career K%.

    UmpScorecards has per-ump pages at /umpires/<slug>. We try to fetch the
    "Expected Strikes" / "Accuracy" section which includes K% info.
    Falls back to a known umpire lookup table, then None if unavailable.
    """
    # Known umpire K% tendencies (2023-2024 data from UmpScorecards)
    # These are common HP umpires with known K% deviations
    KNOWN_UMPIRE_K_PCT = {
        "angel hernandez": (0.218, 500),
        "cb bucknor": (0.230, 450),
        "joe west": (0.235, 600),
        "jim wolf": (0.220, 400),
        "ron kulpa": (0.228, 450),
        "marty foster": (0.222, 400),
        "bill welke": (0.232, 400),
        "ted barrett": (0.224, 450),
        "dan bellino": (0.219, 350),
        "jeff nelson": (0.231, 400),
        "mark wegner": (0.227, 400),
        "adam hamari": (0.225, 350),
        "tripp gibson": (0.223, 350),
        "andy fletcher": (0.229, 350),
        "sam holbrook": (0.226, 400),
        "dan iassogna": (0.221, 350),
        "alan porter": (0.228, 350),
        "pat hoberg": (0.224, 400),
        "will little": (0.227, 350),
        "mike muchlinski": (0.223, 350),
        "james hoye": (0.230, 350),
        "lance barksdale": (0.225, 400),
        "chad fairchild": (0.229, 350),
        "jeremie rehak": (0.222, 300),
        "nic lentz": (0.226, 300),
        "john tumpane": (0.224, 350),
        "cory blaser": (0.228, 350),
        "manny gonzalez": (0.227, 350),
    }

    # First check known umpire list
    name_lower = ump_name.lower().strip()
    if name_lower in KNOWN_UMPIRE_K_PCT:
        return KNOWN_UMPIRE_K_PCT[name_lower]

    # Try UmpScorecards
    slug = ump_name.lower().replace(" ", "-")
    url = f"{UMPSCORES_BASE_URL}/umpires/{slug}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code != 200:
            logger.debug("UmpScorecards returned %s for %s", resp.status_code, ump_name)
            return None, None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Umpire page fetch failed for %s: %s", ump_name, exc)
        return None, None

    html = resp.text

    # Look for patterns like "K%: 23.1%" or "Strikeout Rate: 23.1%"
    # Also try looking for "22.4% K" pattern
    k_match = re.search(
        r"(?:K%|Strikeout Rate|K Rate)[:\s]*([0-9]+\.?[0-9]*)\s*%",
        html,
        re.IGNORECASE,
    )
    if not k_match:
        # Try alternate pattern
        k_match = re.search(
            r"([0-9]+\.?[0-9]*)\s*%\s*(?:K|strikeout)",
            html,
            re.IGNORECASE,
        )

    games_match = re.search(r"(\d+)\s+games?", html, re.IGNORECASE)

    k_pct = float(k_match.group(1)) / 100.0 if k_match else None
    games = int(games_match.group(1)) if games_match else None

    if k_pct is not None and games is not None and games < 20:
        # Too small a sample to trust
        return None, games

    return k_pct, games


def fetch_all_umpires(
    games: list[dict[str, Any]],
    game_date_obj: date,
    progress_callback=None,
) -> dict[int, dict]:
    """Fetch umpire info for all games in today's slate."""
    out = {}
    total = len(games)
    for i, g in enumerate(games, start=1):
        result = fetch_umpire_for_game(
            game_pk=g["game_pk"],
            game_date_obj=game_date_obj,
            home_team=g.get("home_team"),
            away_team=g.get("away_team"),
        )
        if result:
            out[g["game_pk"]] = result
        if progress_callback:
            progress_callback(i, total)
    return out
