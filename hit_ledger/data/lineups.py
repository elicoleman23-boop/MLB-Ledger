"""
Daily schedule and lineup fetching via MLB StatsAPI.

MLB-StatsAPI is an unofficial but widely used wrapper. It gives us:
    - Today's schedule with venue + probable pitchers
    - Confirmed lineups (typically 1-3 hours before first pitch)
    - Batter handedness from roster endpoints

Lineups may be empty for games that haven't posted yet. The caller
should handle missing lineups gracefully (show "TBD" in the UI).
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

import statsapi

from hit_ledger.data import cache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
def fetch_schedule(game_date: date) -> list[dict[str, Any]]:
    """
    Return a list of game dicts for `game_date`:
        game_pk, home_team, away_team, venue, game_time,
        home_pitcher_id, away_pitcher_id
    """
    raw = statsapi.schedule(date=game_date.strftime("%m/%d/%Y"))
    games: list[dict[str, Any]] = []
    for g in raw:
        if g.get("status") in {"Postponed", "Cancelled", "Suspended"}:
            continue

        home_pitcher_id = _parse_pitcher_id(g.get("home_probable_pitcher_id"))
        away_pitcher_id = _parse_pitcher_id(g.get("away_probable_pitcher_id"))

        # For past games, probable pitcher ID may be None but we can get
        # the actual starter from the boxscore
        if home_pitcher_id is None or away_pitcher_id is None:
            try:
                box = statsapi.boxscore_data(g["game_id"])
                if home_pitcher_id is None:
                    home_pitchers = box.get("home", {}).get("pitchers", [])
                    if home_pitchers:
                        home_pitcher_id = int(home_pitchers[0])
                if away_pitcher_id is None:
                    away_pitchers = box.get("away", {}).get("pitchers", [])
                    if away_pitchers:
                        away_pitcher_id = int(away_pitchers[0])
            except Exception as exc:
                logger.warning("Could not fetch boxscore pitchers for game %s: %s", g["game_id"], exc)

        games.append(
            {
                "game_pk": g["game_id"],
                "home_team": g["home_name"],
                "away_team": g["away_name"],
                "venue": g.get("venue_name"),
                "game_time": g.get("game_datetime"),
                "home_pitcher_id": home_pitcher_id,
                "away_pitcher_id": away_pitcher_id,
                "home_pitcher_name": g.get("home_probable_pitcher"),
                "away_pitcher_name": g.get("away_probable_pitcher"),
            }
        )
    return games


def _parse_pitcher_id(raw_id: Any) -> int | None:
    """MLB StatsAPI sometimes returns empty strings for TBD pitchers."""
    if raw_id is None or raw_id == "":
        return None
    try:
        return int(raw_id)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Lineups
# ---------------------------------------------------------------------------
def fetch_lineups_for_game(game_pk: int) -> list[dict[str, Any]]:
    """
    Return a list of lineup entries for both teams:
        game_pk, team, batter_id, batter_name, lineup_slot, bats
    Returns [] if lineups are not yet posted.
    """
    try:
        box = statsapi.boxscore_data(game_pk)
    except Exception as exc:  # noqa: BLE001
        logger.warning("boxscore_data failed for %s: %s", game_pk, exc)
        return []

    entries: list[dict[str, Any]] = []
    team_info = box.get("teamInfo", {})

    for side in ("home", "away"):
        team_data = box.get(side, {})
        # Team name can be in teamInfo or in team.name
        team_name = (
            team_data.get("team", {}).get("name")
            or team_info.get(side, {}).get("teamName")
            or team_info.get(side, {}).get("shortName")
        )
        batting_order = team_data.get("battingOrder", [])
        players_dict = team_data.get("players", {})

        if not batting_order:
            # Lineup not yet posted
            continue

        if not team_name:
            # Team name missing from boxscore data - skip this team
            logger.warning("Missing team name in boxscore for game %s side %s", game_pk, side)
            continue

        for slot_idx, player_id in enumerate(batting_order, start=1):
            player_key = f"ID{player_id}"
            player = players_dict.get(player_key, {})
            person = player.get("person", {})
            bat_side = player.get("batSide", {}).get("code")  # L/R/S
            entries.append(
                {
                    "game_pk": game_pk,
                    "team": team_name,
                    "batter_id": int(player_id),
                    "batter_name": person.get("fullName"),
                    "lineup_slot": slot_idx,
                    "bats": bat_side,
                }
            )
    return entries


def fetch_pitcher_handedness(pitcher_id: int) -> str | None:
    """Return 'L', 'R', or None."""
    try:
        data = statsapi.lookup_player(pitcher_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("lookup_player failed for %s: %s", pitcher_id, exc)
        return None
    if not data:
        return None
    hand = data[0].get("pitchHand", {}).get("code")
    return hand


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def refresh_daily_schedule(game_date: date) -> dict[str, Any]:
    """
    Fetch schedule + all lineups for a given date, persist to cache,
    and return a summary dict.

    Returns:
        {
            'n_games': int,
            'n_lineups_posted': int,
            'n_batters': int,
        }
    """
    games = fetch_schedule(game_date)
    cache.save_games(game_date, games)

    all_lineups: list[dict[str, Any]] = []
    lineups_posted = 0
    for g in games:
        lineup = fetch_lineups_for_game(g["game_pk"])
        if lineup:
            lineups_posted += 1
        all_lineups.extend(lineup)

    cache.save_lineups(game_date, all_lineups)

    return {
        "n_games": len(games),
        "n_lineups_posted": lineups_posted,
        "n_batters": len(all_lineups),
    }
