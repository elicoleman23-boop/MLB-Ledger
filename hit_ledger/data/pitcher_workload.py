"""
Starter workload and Times-Through-Order splits.

Two analyses on the same starter Statcast data:

1. Average IP/start from last N starts → expected PAs vs starter per lineup slot
2. Pitcher-specific xBA by TTO (1st, 2nd, 3rd time through the order)

If the pitcher has fewer than STARTER_RECENT_STARTS starts, we use whatever
is available. If they have no starts in the data, we fall back to league
averages (defined in config).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from pybaseball import statcast_pitcher

from hit_ledger.config import (
    LEAGUE_AVG_PA_PER_INNING,
    LEAGUE_AVG_STARTER_IP,
    LEAGUE_AVG_XBA,
    STARTER_RECENT_STARTS,
    TTO_MIN_PA_2ND,
    TTO_MIN_PA_3RD,
)
from hit_ledger.data import cache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data pull helper (shared between workload and TTO)
# ---------------------------------------------------------------------------
def _pull_starter_pitches(pitcher_id: int, as_of: date) -> pd.DataFrame:
    """Season-to-date pitches thrown by `pitcher_id`."""
    season_start = date(as_of.year, 3, 1)
    try:
        df = statcast_pitcher(
            start_dt=season_start.isoformat(),
            end_dt=(as_of - timedelta(days=1)).isoformat(),
            player_id=pitcher_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("statcast_pitcher failed for %s: %s", pitcher_id, exc)
        return pd.DataFrame()
    return df


# ---------------------------------------------------------------------------
# Workload: average IP per start
# ---------------------------------------------------------------------------
def fetch_starter_workload(
    pitcher_id: int,
    as_of: date,
    force_refresh: bool = False,
) -> dict:
    """
    Return:
        {
            'avg_ip_per_start': float,
            'starts_sampled': int,
            'season_xba': float,
        }
    """
    cached = cache.load_starter_workload(as_of, pitcher_id)
    if cached is not None and not force_refresh:
        return cached

    df = _pull_starter_pitches(pitcher_id, as_of)
    if df.empty or "game_pk" not in df.columns:
        out = _league_avg_workload()
        cache.save_starter_workload(
            as_of, pitcher_id,
            out["avg_ip_per_start"],
            out["starts_sampled"],
            out["season_xba"],
        )
        return out

    # Identify games where this pitcher was the starter.
    # Heuristic: first pitcher in a game for a given team is the starter.
    # statcast_pitcher already filtered by pitcher, so we just need their games
    # where they threw pitch 1 of the game for their team.
    starter_games = _identify_starter_games(df)

    if not starter_games:
        out = _league_avg_workload()
        cache.save_starter_workload(
            as_of, pitcher_id,
            out["avg_ip_per_start"],
            out["starts_sampled"],
            out["season_xba"],
        )
        return out

    # Compute IP per start (use outs recorded / 3)
    ip_by_game = {}
    for game_pk in starter_games:
        game_df = df[df["game_pk"] == game_pk]
        outs = _count_outs(game_df)
        ip_by_game[game_pk] = outs / 3.0

    # Sort by date, take last N
    game_dates = (
        df[df["game_pk"].isin(starter_games)]
        .groupby("game_pk")["game_date"]
        .first()
        .sort_values(ascending=False)
    )
    recent_games = game_dates.index[:STARTER_RECENT_STARTS]
    recent_ips = [ip_by_game[g] for g in recent_games if g in ip_by_game]

    avg_ip = float(np.mean(recent_ips)) if recent_ips else LEAGUE_AVG_STARTER_IP

    # Season xBA-against (used for TTO quality scaling)
    season_xba = _compute_pitcher_xba(df)

    out = {
        "avg_ip_per_start": avg_ip,
        "starts_sampled": len(recent_ips),
        "season_xba": season_xba,
    }
    cache.save_starter_workload(
        as_of, pitcher_id,
        out["avg_ip_per_start"],
        out["starts_sampled"],
        out["season_xba"],
    )
    return out


def _identify_starter_games(df: pd.DataFrame) -> list[int]:
    """Games where this pitcher threw the first pitch of the game (= started)."""
    if "at_bat_number" not in df.columns or "pitch_number" not in df.columns:
        return []
    # Games where this pitcher has at_bat_number == 1 (first PA of game)
    return df[df["at_bat_number"] == 1]["game_pk"].unique().tolist()


def _count_outs(game_df: pd.DataFrame) -> int:
    """Count outs recorded while this pitcher was pitching in a game."""
    pa_ending = game_df[game_df["events"].notna() & (game_df["events"] != "")]
    # Events that record outs. Some events record multiple outs (double plays).
    out_events = {
        "field_out", "strikeout", "grounded_into_double_play", "force_out",
        "sac_fly", "sac_bunt", "sac_fly_double_play", "strikeout_double_play",
        "fielders_choice_out", "caught_stealing_2b", "caught_stealing_3b",
        "caught_stealing_home", "pickoff_caught_stealing_2b",
        "pickoff_caught_stealing_3b", "pickoff_caught_stealing_home",
        "pickoff_1b", "pickoff_2b", "pickoff_3b",
        "triple_play", "double_play",
    }
    # Standard approximation: 1 out per out-event, 2 for DPs, 3 for TPs
    outs = 0
    for ev in pa_ending["events"]:
        if ev in {"grounded_into_double_play", "double_play",
                  "strikeout_double_play", "sac_fly_double_play"}:
            outs += 2
        elif ev == "triple_play":
            outs += 3
        elif ev in out_events:
            outs += 1
    return outs


def _compute_pitcher_xba(df: pd.DataFrame) -> float:
    """Season xBA-against."""
    if df.empty:
        return LEAGUE_AVG_XBA
    pa_ending = df[df["events"].notna() & (df["events"] != "")]
    if pa_ending.empty:
        return LEAGUE_AVG_XBA
    xba_series = pa_ending["estimated_ba_using_speedangle"].copy()
    hit_events = {"single", "double", "triple", "home_run"}
    for idx in xba_series[xba_series.isna()].index:
        ev = pa_ending.at[idx, "events"]
        xba_series.at[idx] = 1.0 if ev in hit_events else 0.0
    return float(xba_series.mean())


def _league_avg_workload() -> dict:
    return {
        "avg_ip_per_start": LEAGUE_AVG_STARTER_IP,
        "starts_sampled": 0,
        "season_xba": LEAGUE_AVG_XBA,
    }


def expected_pa_vs_starter(avg_ip: float, lineup_slot: int) -> float:
    """
    Given the starter's avg IP and a batter's lineup slot, estimate
    how many PAs the batter will get against the starter.

    Approach: each inning has 3-4.5 PAs distributed across 9 batters.
    After N innings, each batter gets approximately (N × PA_PER_INNING / 9)
    PAs. Lineup slots 1-3 are slightly more likely to get an extra PA
    when innings end mid-rotation.
    """
    innings = avg_ip
    total_pa_vs_starter = innings * LEAGUE_AVG_PA_PER_INNING  # total team PAs
    # Lineup slot adjustment: earlier hitters get marginally more PAs
    base = total_pa_vs_starter / 9
    slot_bonus = {
        1: 0.25, 2: 0.20, 3: 0.15, 4: 0.10, 5: 0.05,
        6: 0.00, 7: -0.05, 8: -0.10, 9: -0.15,
    }
    return max(0.0, base + slot_bonus.get(lineup_slot, 0.0))


# ---------------------------------------------------------------------------
# TTO splits
# ---------------------------------------------------------------------------
def fetch_tto_splits(
    pitcher_id: int,
    as_of: date,
    force_refresh: bool = False,
) -> dict:
    """
    Return:
        {
            'xba': {1: float, 2: float, 3: float},
            'pa':  {1: int,   2: int,   3: int},
        }
    """
    cached = cache.load_tto_splits(as_of, pitcher_id)
    if cached is not None and not force_refresh:
        return cached

    df = _pull_starter_pitches(pitcher_id, as_of)
    if df.empty:
        empty = {"xba": {1: None, 2: None, 3: None},
                 "pa": {1: 0, 2: 0, 3: 0}}
        cache.save_tto_splits(as_of, pitcher_id, empty["xba"], empty["pa"])
        return empty

    # Add TTO column per row. "TTO" = which time the batter is seeing
    # this pitcher in the current game. Count unique (game_pk, batter)
    # appearances up to that at_bat_number.
    df = df.copy()
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])

    # For each (game, batter), how many times has the batter faced this pitcher so far?
    # We compute this as the rank of the at_bat_number among unique PAs for that
    # batter in that game.
    pa_end_mask = df["events"].notna() & (df["events"] != "")

    # Keep only one row per PA (the last pitch)
    pa_rows = df[pa_end_mask].drop_duplicates(
        subset=["game_pk", "at_bat_number"], keep="last"
    )

    # Rank PAs within (game, batter) chronologically
    pa_rows = pa_rows.sort_values(["game_pk", "batter", "at_bat_number"])
    pa_rows["tto"] = pa_rows.groupby(["game_pk", "batter"]).cumcount() + 1

    # Aggregate xBA by TTO
    xbas = {}
    pas = {}
    for tto in [1, 2, 3]:
        tto_pas = pa_rows[pa_rows["tto"] == tto]
        n = len(tto_pas)
        pas[tto] = n
        if n == 0:
            xbas[tto] = None
            continue
        xba_series = tto_pas["estimated_ba_using_speedangle"].copy()
        hit_events = {"single", "double", "triple", "home_run"}
        for idx in xba_series[xba_series.isna()].index:
            ev = tto_pas.at[idx, "events"]
            xba_series.at[idx] = 1.0 if ev in hit_events else 0.0
        xbas[tto] = float(xba_series.mean())

    out = {"xba": xbas, "pa": pas}
    cache.save_tto_splits(as_of, pitcher_id, out["xba"], out["pa"])
    return out


def fetch_pitcher_stats(
    pitcher_id: int,
    as_of: date,
) -> dict:
    """
    Fetch traditional pitcher stats for display: ERA, WHIP, K%, HR/9.

    Returns:
        {
            'era': float,
            'whip': float,
            'k_pct': float,
            'hr_per_9': float,
            'xba': float,
            'throws': str,
        }
    """
    try:
        import statsapi
        # Get player stats from MLB API
        player_stats = statsapi.player_stat_data(
            personId=pitcher_id,
            group="pitching",
            type="season",
            sportId=1,
        )

        stats = {}
        if player_stats.get("stats"):
            for stat_entry in player_stats["stats"]:
                if stat_entry.get("stats"):
                    s = stat_entry["stats"]
                    stats["era"] = float(s.get("era", 0)) if s.get("era") and s.get("era") != "-" else None
                    stats["whip"] = float(s.get("whip", 0)) if s.get("whip") and s.get("whip") != "-" else None

                    # Calculate K%: strikeouts / batters faced
                    so = int(s.get("strikeOuts", 0) or 0)
                    bf = int(s.get("battersFaced", 0) or 0)
                    stats["k_pct"] = (so / bf) if bf > 0 else None

                    # HR/9: (HR / IP) * 9
                    hr = int(s.get("homeRuns", 0) or 0)
                    ip_str = s.get("inningsPitched", "0")
                    try:
                        # IP is stored as "X.Y" where Y is the partial inning (0, 1, or 2)
                        ip_parts = str(ip_str).split(".")
                        ip = float(ip_parts[0])
                        if len(ip_parts) > 1:
                            ip += float(ip_parts[1]) / 3.0
                    except (ValueError, IndexError):
                        ip = 0
                    stats["hr_per_9"] = (hr / ip * 9) if ip > 0 else None
                    break

        # Get handedness from player info
        player_info = statsapi.get("person", {"personId": pitcher_id})
        if player_info.get("people"):
            stats["throws"] = player_info["people"][0].get("pitchHand", {}).get("code", "R")
        else:
            stats["throws"] = "R"

        # Get xBA from workload data (already cached)
        workload = fetch_starter_workload(pitcher_id, as_of)
        stats["xba"] = workload.get("season_xba")

        return stats

    except Exception as exc:
        logger.warning("Failed to fetch pitcher stats for %s: %s", pitcher_id, exc)
        return {}


def tto_penalty_to_apply(
    tto_data: dict,
    season_xba: float,
    tto_num: int,
) -> float:
    """
    Return the xBA delta to add for a given TTO.

    Hybrid approach:
        - If pitcher has sufficient sample for this TTO (per config thresholds),
          use (pitcher_tto_xba - pitcher_1st_tto_xba) as the penalty.
        - Otherwise, use flat league penalty scaled by pitcher quality:
            worse pitchers (higher season_xba) get bigger penalties.
    """
    from hit_ledger.config import (
        LEAGUE_AVG_XBA,
        TTO_FLAT_PENALTY,
        TTO_QUALITY_MULT_CAP,
    )

    if tto_num == 1:
        return 0.0

    xbas = tto_data.get("xba", {})
    pas = tto_data.get("pa", {})

    min_pa = TTO_MIN_PA_2ND if tto_num == 2 else TTO_MIN_PA_3RD
    tto1_pa = pas.get(1, 0)
    this_pa = pas.get(tto_num, 0)

    # Use pitcher-specific if sample is good on BOTH the reference (TTO1) and this TTO
    if (
        tto1_pa >= TTO_MIN_PA_2ND
        and this_pa >= min_pa
        and xbas.get(1) is not None
        and xbas.get(tto_num) is not None
    ):
        return xbas[tto_num] - xbas[1]

    # Fallback: flat penalty scaled by quality
    flat = TTO_FLAT_PENALTY.get(tto_num, 0.020)
    # Quality multiplier: worse than league → larger penalty
    quality_mult = (season_xba - LEAGUE_AVG_XBA) / LEAGUE_AVG_XBA
    quality_mult = max(-TTO_QUALITY_MULT_CAP, min(TTO_QUALITY_MULT_CAP, quality_mult))
    return flat * (1.0 + quality_mult)
