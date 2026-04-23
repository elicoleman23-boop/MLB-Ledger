"""
Pitcher pitch-level Statcast profile acquisition.

Mirrors data.statcast.fetch_batter_profile but from the pitcher's POV.
The returned DataFrame is used to compute per-pitch-type xBA-against and
contact rate against batters of a given handedness — i.e. the pitcher
side of the log-5 matchup blend.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
from pybaseball import statcast_pitcher

from hit_ledger.data import cache

logger = logging.getLogger(__name__)

# If current-season sample is below this, extend into prior season for depth.
_MIN_CURRENT_SEASON_PITCHES = 500


def fetch_pitcher_profile(
    pitcher_id: int,
    as_of: date,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Pull pitch-level Statcast data for the pitcher for the current season
    (fallback to prior season if fewer than ~500 pitches), ending at
    `as_of` (exclusive).

    Returns a slim DataFrame with columns:
        game_date, pitch_type, p_throws, stand, events, description,
        estimated_ba_using_speedangle, launch_speed, launch_angle, batter
    """
    cached = cache.load_pitcher_profile(as_of, pitcher_id)
    if cached is not None and not force_refresh:
        return cached

    season_start = date(as_of.year, 3, 1)
    try:
        raw = statcast_pitcher(
            start_dt=season_start.isoformat(),
            end_dt=(as_of - timedelta(days=1)).isoformat(),
            player_id=pitcher_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("statcast_pitcher failed for %s: %s", pitcher_id, exc)
        raw = pd.DataFrame()

    if len(raw) < _MIN_CURRENT_SEASON_PITCHES:
        prior_start = date(as_of.year - 1, 3, 1)
        prior_end = date(as_of.year - 1, 11, 1)
        try:
            prior = statcast_pitcher(
                start_dt=prior_start.isoformat(),
                end_dt=prior_end.isoformat(),
                player_id=pitcher_id,
            )
            raw = pd.concat([raw, prior], ignore_index=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("fallback statcast_pitcher failed for %s: %s", pitcher_id, exc)

    slim = _slim_pitcher_frame(raw)
    cache.save_pitcher_profile(as_of, pitcher_id, slim)
    return slim


def _slim_pitcher_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns we use; drop pitches with no pitch_type."""
    keep = [
        "game_date", "pitch_type", "p_throws", "stand",
        "events", "description", "estimated_ba_using_speedangle",
        "launch_speed", "launch_angle",
        "batter",  # analog of `pitcher` on the batter profile
    ]
    if df.empty:
        return pd.DataFrame(columns=keep)

    present = [c for c in keep if c in df.columns]
    out = df[present].copy()
    out = out[out["pitch_type"].notna()]
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    return out.reset_index(drop=True)


def fetch_all_pitcher_profiles(
    pitcher_ids: list[int],
    as_of: date,
    progress_callback=None,
) -> dict[int, pd.DataFrame]:
    """Fetch pitch-level profiles for a list of pitchers with progress reporting."""
    out: dict[int, pd.DataFrame] = {}
    total = len(pitcher_ids)
    for i, pid in enumerate(pitcher_ids, start=1):
        out[pid] = fetch_pitcher_profile(pid, as_of)
        if progress_callback:
            progress_callback(i, total)
    return out
