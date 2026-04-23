"""
Statcast data acquisition via pybaseball.

Two primary pulls per day:
    1. Batter profiles: per-pitch Statcast data for the last N seasons.
    2. Pitcher arsenals: the probable pitcher's pitch-type distribution.

Both are cached aggressively in SQLite (via data.cache) because
pybaseball scrapes are slow and rate-limited.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd
from pybaseball import statcast_batter, statcast_pitcher

from hit_ledger.config import SEASONS_LOOKBACK
from hit_ledger.data import cache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batter profile
# ---------------------------------------------------------------------------
def fetch_batter_profile(
    batter_id: int,
    as_of: date,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Pull the last `SEASONS_LOOKBACK` seasons of pitch-level Statcast data
    for `batter_id`, ending at `as_of` (exclusive).

    Returns a slim DataFrame with only the columns needed for modeling:
        game_date, pitch_type, p_throws, stand, events, description,
        estimated_ba_using_speedangle (xBA proxy), launch_speed, launch_angle
    """
    cached = cache.load_batter_profile(as_of, batter_id)
    if cached is not None and not force_refresh:
        return cached

    start = as_of.replace(year=as_of.year - SEASONS_LOOKBACK)
    try:
        raw = statcast_batter(
            start_dt=start.isoformat(),
            end_dt=(as_of - timedelta(days=1)).isoformat(),
            player_id=batter_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("statcast_batter failed for %s: %s", batter_id, exc)
        raw = pd.DataFrame()

    slim = _slim_statcast_frame(raw)
    cache.save_batter_profile(as_of, batter_id, slim)
    return slim


def _slim_statcast_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns we actually use, drop pitches with no pitch_type."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "game_date", "pitch_type", "p_throws", "stand",
                "events", "description", "estimated_ba_using_speedangle",
                "launch_speed", "launch_angle", "pitcher",
            ]
        )

    keep = [
        "game_date", "pitch_type", "p_throws", "stand",
        "events", "description", "estimated_ba_using_speedangle",
        "launch_speed", "launch_angle",
        "pitcher",  # needed for BvP lookup
    ]
    present = [c for c in keep if c in df.columns]
    out = df[present].copy()
    out = out[out["pitch_type"].notna()]
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pitcher arsenal
# ---------------------------------------------------------------------------
def fetch_pitcher_arsenal(
    pitcher_id: int,
    as_of: date,
    throws: str | None = None,
    force_refresh: bool = False,
) -> tuple[str, dict[str, float]]:
    """
    Return (throws, {pitch_type: share}) for a pitcher's recent arsenal.

    Uses the current season only; if fewer than ~200 pitches, falls back
    to the prior season.
    """
    cached = cache.load_pitcher_arsenal(as_of, pitcher_id)
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

    if len(raw) < 200:
        # Fall back to prior season
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
            logger.warning("fallback statcast_pitcher failed: %s", exc)

    if raw.empty or "pitch_type" not in raw.columns:
        arsenal: dict[str, float] = {}
        hand = throws or "R"
    else:
        pitches = raw[raw["pitch_type"].notna()]
        counts = pitches["pitch_type"].value_counts(normalize=True)
        arsenal = {pt: float(share) for pt, share in counts.items() if share >= 0.02}
        # Renormalize after dropping tiny shares
        total = sum(arsenal.values())
        if total > 0:
            arsenal = {k: v / total for k, v in arsenal.items()}
        # Prefer observed throws from data, fall back to provided
        hand = (
            pitches["p_throws"].mode().iat[0]
            if "p_throws" in pitches.columns and not pitches["p_throws"].mode().empty
            else (throws or "R")
        )

    cache.save_pitcher_arsenal(as_of, pitcher_id, hand, arsenal)
    return hand, arsenal


# ---------------------------------------------------------------------------
# Batch fetch orchestrator
# ---------------------------------------------------------------------------
def fetch_all_batters(
    batter_ids: list[int],
    as_of: date,
    progress_callback: Any = None,
) -> dict[int, pd.DataFrame]:
    """Fetch all batter profiles, reporting progress via callback(done, total)."""
    out: dict[int, pd.DataFrame] = {}
    total = len(batter_ids)
    for i, bid in enumerate(batter_ids, start=1):
        out[bid] = fetch_batter_profile(bid, as_of)
        if progress_callback:
            progress_callback(i, total)
    return out


def fetch_all_pitchers(
    pitcher_ids: list[int],
    as_of: date,
    progress_callback: Any = None,
) -> dict[int, tuple[str, dict[str, float]]]:
    out: dict[int, tuple[str, dict[str, float]]] = {}
    total = len(pitcher_ids)
    for i, pid in enumerate(pitcher_ids, start=1):
        out[pid] = fetch_pitcher_arsenal(pid, as_of)
        if progress_callback:
            progress_callback(i, total)
    return out
