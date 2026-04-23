"""
Team bullpen xBA-against by handedness.

Approach: pull all Statcast pitches for the season filtered to relief
appearances (non-starter rows). Group by team + batter handedness,
compute xBA from PA-ending events.

This is a heavy query but runs once per team per day and caches.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
from pybaseball import statcast

from hit_ledger.config import (
    BULLPEN_REGRESSION_K,
    LEAGUE_AVG_BULLPEN_XBA_VS_L,
    LEAGUE_AVG_BULLPEN_XBA_VS_R,
)
from hit_ledger.data import cache

logger = logging.getLogger(__name__)


# MLB team abbreviations used by Statcast in the `home_team` / `away_team` cols.
# These map to the full team names used by MLB StatsAPI (so we can join).
TEAM_ABBR_TO_NAME = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles", "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs", "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies", "DET": "Detroit Tigers",
    "HOU": "Houston Astros", "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins", "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates",
    "SD": "San Diego Padres", "SEA": "Seattle Mariners",
    "SF": "San Francisco Giants", "STL": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays", "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays", "WSH": "Washington Nationals",
}
TEAM_NAME_TO_ABBR = {v: k for k, v in TEAM_ABBR_TO_NAME.items()}

# Also support short team names (teamName from MLB API boxscore)
TEAM_SHORT_TO_ABBR = {
    "D-backs": "ARI", "Diamondbacks": "ARI", "Braves": "ATL",
    "Orioles": "BAL", "Red Sox": "BOS", "Cubs": "CHC",
    "White Sox": "CWS", "Reds": "CIN", "Guardians": "CLE",
    "Rockies": "COL", "Tigers": "DET", "Astros": "HOU",
    "Royals": "KC", "Angels": "LAA", "Dodgers": "LAD",
    "Marlins": "MIA", "Brewers": "MIL", "Twins": "MIN",
    "Mets": "NYM", "Yankees": "NYY", "Athletics": "OAK",
    "Phillies": "PHI", "Pirates": "PIT", "Padres": "SD",
    "Mariners": "SEA", "Giants": "SF", "Cardinals": "STL",
    "Rays": "TB", "Rangers": "TEX", "Blue Jays": "TOR",
    "Nationals": "WSH",
}


def _regress_bullpen(raw_xba: float, n_pa: int, league_avg: float) -> float:
    """Bayesian shrinkage to league average."""
    return (n_pa * raw_xba + BULLPEN_REGRESSION_K * league_avg) / (
        n_pa + BULLPEN_REGRESSION_K
    )


def fetch_team_bullpen_profile(
    team_name: str,
    as_of: date,
    force_refresh: bool = False,
) -> dict:
    """
    Return bullpen xBA-against for `team_name` by batter handedness.

    Returns:
        {
            'xba_vs_r': float,
            'xba_vs_l': float,
            'pa_vs_r': int,
            'pa_vs_l': int,
        }
    """
    cached = cache.load_bullpen_profile(as_of, team_name)
    if cached is not None and not force_refresh:
        return cached

    abbr = TEAM_NAME_TO_ABBR.get(team_name) or TEAM_SHORT_TO_ABBR.get(team_name)
    if abbr is None:
        logger.warning("Unknown team name: %s", team_name)
        return _league_avg_profile()

    # Pull season-to-date Statcast for this team (as pitching team)
    season_start = date(as_of.year, 3, 1)
    try:
        df = statcast(
            start_dt=season_start.isoformat(),
            end_dt=(as_of - timedelta(days=1)).isoformat(),
            team=abbr,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("statcast fetch failed for %s: %s", team_name, exc)
        return _league_avg_profile()

    if df.empty:
        return _league_avg_profile()

    # Filter to only pitches where this team was pitching
    # statcast() returns both halves of innings; keep where inning_topbot indicates
    # the opposing team is at bat, meaning `abbr` is fielding.
    # The `pitcher` column with a lookup of starter vs reliever is the cleanest
    # filter. For now we use a simpler heuristic: exclude the top-of-rotation
    # pitcher from each game via `pitcher` rank within game.
    df_pitching = df[df["pitching_team"] == abbr] if "pitching_team" in df.columns else df

    # Filter to relief pitches: for each game, first pitcher = starter, exclude.
    if "game_pk" in df_pitching.columns and "pitcher" in df_pitching.columns:
        # Identify the starter per game (first unique pitcher in chronological order)
        df_sorted = df_pitching.sort_values(["game_pk", "at_bat_number", "pitch_number"])
        starters = df_sorted.groupby("game_pk")["pitcher"].first()
        df_pitching = df_pitching.merge(
            starters.rename("starter_id"), left_on="game_pk", right_index=True
        )
        relief = df_pitching[df_pitching["pitcher"] != df_pitching["starter_id"]]
    else:
        relief = df_pitching

    if relief.empty:
        return _league_avg_profile()

    # Compute xBA by batter stand
    result = {}
    for stand, key, league in [
        ("R", "vs_r", LEAGUE_AVG_BULLPEN_XBA_VS_R),
        ("L", "vs_l", LEAGUE_AVG_BULLPEN_XBA_VS_L),
    ]:
        side = relief[relief["stand"] == stand]
        pa_ending = side[side["events"].notna() & (side["events"] != "")]
        n_pa = len(pa_ending)
        if n_pa == 0:
            result[f"xba_{key}"] = league
            result[f"pa_{key}"] = 0
            continue

        xba_series = pa_ending["estimated_ba_using_speedangle"].copy()
        hit_events = {"single", "double", "triple", "home_run"}
        # Reset index to avoid duplicate index issues
        pa_ending_reset = pa_ending.reset_index(drop=True)
        xba_series = xba_series.reset_index(drop=True)
        for i, idx in enumerate(xba_series[xba_series.isna()].index):
            ev = pa_ending_reset.loc[idx, "events"]
            # Handle case where ev might still be a Series
            if isinstance(ev, pd.Series):
                ev = ev.iloc[0] if len(ev) > 0 else ""
            xba_series.loc[idx] = 1.0 if ev in hit_events else 0.0
        raw = float(xba_series.mean())
        result[f"xba_{key}"] = _regress_bullpen(raw, n_pa, league)
        result[f"pa_{key}"] = n_pa

    cache.save_bullpen_profile(
        as_of,
        team_name,
        result["xba_vs_r"],
        result["xba_vs_l"],
        result["pa_vs_r"],
        result["pa_vs_l"],
    )
    return result


def _league_avg_profile() -> dict:
    return {
        "xba_vs_r": LEAGUE_AVG_BULLPEN_XBA_VS_R,
        "xba_vs_l": LEAGUE_AVG_BULLPEN_XBA_VS_L,
        "pa_vs_r": 0,
        "pa_vs_l": 0,
    }


def fetch_all_bullpens(
    team_names: list[str],
    as_of: date,
    progress_callback=None,
) -> dict[str, dict]:
    out = {}
    total = len(team_names)
    for i, team in enumerate(team_names, start=1):
        out[team] = fetch_team_bullpen_profile(team, as_of)
        if progress_callback:
            progress_callback(i, total)
    return out
