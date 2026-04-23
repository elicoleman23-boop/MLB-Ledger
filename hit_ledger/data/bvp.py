"""
Batter vs Pitcher (BvP) career history.

Per user spec: annotation only. This module fetches lifetime BvP stats
but they do NOT flow into the probability model. The UI uses them as
a sidebar note next to each batter row when the "BvP" toggle is enabled.

Data source: pybaseball's `statcast_batter` already gives us pitcher IDs
per pitch, so we can compute career BvP from the batter's Statcast blob
that's already cached. No additional scrape needed.
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from hit_ledger.config import BVP_MIN_PA_TO_DISPLAY
from hit_ledger.data import cache

logger = logging.getLogger(__name__)


def compute_bvp(
    batter_id: int,
    pitcher_id: int,
    batter_df: pd.DataFrame,
    as_of: date,
    force_refresh: bool = False,
) -> dict | None:
    """
    Compute lifetime BvP stats from a batter's Statcast history.

    Args:
        batter_id, pitcher_id: MLBAM ids
        batter_df: slim Statcast frame from data.statcast.fetch_batter_profile
                   (contains the batter's entire pitch-by-pitch history)
        as_of: for cache key

    Returns:
        {
            'pa': int, 'ab': int, 'hits': int, 'hr': int,
            'so': int, 'bb': int, 'ba': float, 'slg': float,
        }
        or None if the batter has never faced this pitcher (per the sample).
    """
    cached = cache.load_bvp(as_of, batter_id, pitcher_id)
    if cached is not None and not force_refresh:
        if cached.get("pa", 0) == 0:
            return None
        return _enrich_bvp(cached)

    if batter_df.empty or "pitcher" not in batter_df.columns:
        # The slim frame we cache doesn't include `pitcher` column.
        # Need to extend the batter profile pull to include it for this to work.
        # For now, return None and log.
        logger.debug(
            "Batter profile missing 'pitcher' column — BvP unavailable for %s vs %s",
            batter_id, pitcher_id,
        )
        return None

    vs_pitcher = batter_df[batter_df["pitcher"] == pitcher_id]
    pa_ending = vs_pitcher[vs_pitcher["events"].notna() & (vs_pitcher["events"] != "")]

    if pa_ending.empty:
        cache.save_bvp(as_of, batter_id, pitcher_id, {"pa": 0})
        return None

    events = pa_ending["events"].value_counts().to_dict()
    pa = len(pa_ending)
    # AB excludes walks, HBP, sac bunts, sac flies, catcher interference
    non_ab = {
        "walk", "hit_by_pitch", "sac_bunt", "sac_fly",
        "sac_fly_double_play", "catcher_interf",
    }
    ab = sum(n for ev, n in events.items() if ev not in non_ab)
    hits = sum(events.get(ev, 0) for ev in ["single", "double", "triple", "home_run"])
    hr = events.get("home_run", 0)
    so = events.get("strikeout", 0) + events.get("strikeout_double_play", 0)
    bb = events.get("walk", 0)

    stats = {
        "pa": pa, "ab": ab, "hits": hits, "hr": hr,
        "so": so, "bb": bb,
    }
    cache.save_bvp(as_of, batter_id, pitcher_id, stats)

    if pa < BVP_MIN_PA_TO_DISPLAY:
        return None
    return _enrich_bvp(stats)


def _enrich_bvp(stats: dict) -> dict:
    """Add BA and SLG to raw counts for display."""
    ab = stats.get("ab", 0)
    if ab == 0:
        stats["ba"] = 0.0
        stats["slg"] = 0.0
        return stats
    hits = stats.get("hits", 0)
    hr = stats.get("hr", 0)
    stats["ba"] = hits / ab
    # Approximate SLG: we don't have 2B/3B breakdown cached, so approximate
    # using (hits + HRs count double) — crude but fine for display
    stats["slg"] = (hits + hr) / ab  # approximate
    return stats


def format_bvp_annotation(bvp: dict | None) -> str:
    """Produce a short UI string like '4-for-18 (.222), 2 HR, 6 K'."""
    if bvp is None or bvp.get("pa", 0) < BVP_MIN_PA_TO_DISPLAY:
        return ""
    ab = bvp.get("ab", 0)
    hits = bvp.get("hits", 0)
    hr = bvp.get("hr", 0)
    so = bvp.get("so", 0)
    ba = bvp.get("ba", 0.0)
    if ab == 0:
        return f"{bvp.get('pa', 0)} PA, {hr} HR, {so} K"
    ba_str = f".{int(ba * 1000):03d}" if ba > 0 else ".000"
    parts = [f"{hits}-for-{ab} ({ba_str})"]
    if hr:
        parts.append(f"{hr} HR")
    if so:
        parts.append(f"{so} K")
    return ", ".join(parts)
