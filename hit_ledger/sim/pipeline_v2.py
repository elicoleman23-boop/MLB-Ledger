"""
v2 top-level orchestrator.

Stages:
    1. schedule & lineups (unchanged from v1)
    2. batter profiles (unchanged)
    3. pitcher arsenals (unchanged)
    4. starter workload (IP/start, season xBA)  — NEW
    5. TTO splits                                — NEW
    6. team bullpen profiles                     — NEW
    7. umpire assignments                        — NEW
    8. build v2 matchups (per-PA probability sequences)
    9. simulate
   10. optionally compute BvP annotations for display
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

import pandas as pd

from hit_ledger.config import BVP_DEFAULT_ENABLED, PA_BY_LINEUP_SLOT, DEFAULT_PA
from hit_ledger.data import cache, lineups, statcast
from hit_ledger.data import bullpen as bullpen_data
from hit_ledger.data import pitcher_workload
from hit_ledger.data import umpires as umpire_data
from hit_ledger.data import bvp as bvp_data
from hit_ledger.sim.engine_v2 import results_to_df_v2, simulate_v2
from hit_ledger.sim.matchup_v2 import build_matchup_v2
from hit_ledger.utils.teams import TEAM_SHORT_TO_FULL

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]


def _normalize_team(name: str) -> str:
    """Convert short team name to full name for comparison."""
    return TEAM_SHORT_TO_FULL.get(name, name)


@dataclass
class PipelineResultV2:
    game_date: date
    games: pd.DataFrame
    lineups: pd.DataFrame
    predictions: pd.DataFrame
    matchup_details: dict  # batter_id -> MatchupV2
    bvp_annotations: dict  # batter_id -> str
    umpires: dict          # game_pk -> umpire info
    pitcher_stats: dict    # pitcher_id -> stats dict
    summary: dict


def run_daily_pipeline_v2(
    game_date: date,
    progress: ProgressCallback | None = None,
    force_refresh: bool = False,
    enable_bvp: bool = BVP_DEFAULT_ENABLED,
) -> PipelineResultV2:
    """Full v2 pipeline."""
    _tick(progress, "schedule", 0.0)

    # 1-2. Schedule & lineups
    summary = lineups.refresh_daily_schedule(game_date)
    games_df = cache.load_games(game_date)
    lineups_df = cache.load_lineups(game_date)
    _tick(progress, "schedule", 1.0)

    if lineups_df.empty:
        return PipelineResultV2(
            game_date=game_date,
            games=games_df, lineups=lineups_df,
            predictions=pd.DataFrame(), matchup_details={},
            bvp_annotations={}, umpires={}, pitcher_stats={}, summary=summary,
        )

    # 3. Batter profiles
    batter_ids = lineups_df["batter_id"].dropna().astype(int).unique().tolist()
    batter_profiles = statcast.fetch_all_batters(
        batter_ids, game_date,
        progress_callback=lambda d, t: _tick(progress, "batters", d / max(t, 1)),
    )

    # 4. Pitcher arsenals
    pitcher_ids = []
    for _, g in games_df.iterrows():
        if pd.notna(g["home_pitcher_id"]):
            pitcher_ids.append(int(g["home_pitcher_id"]))
        if pd.notna(g["away_pitcher_id"]):
            pitcher_ids.append(int(g["away_pitcher_id"]))
    pitcher_ids = list(set(pitcher_ids))

    pitcher_data = statcast.fetch_all_pitchers(
        pitcher_ids, game_date,
        progress_callback=lambda d, t: _tick(progress, "pitchers", d / max(t, 1)),
    )

    # 5. Starter workload (NEW)
    _tick(progress, "workload", 0.0)
    workloads = {}
    pitcher_stats_cache = {}
    for i, pid in enumerate(pitcher_ids, start=1):
        workloads[pid] = pitcher_workload.fetch_starter_workload(pid, game_date)
        pitcher_stats_cache[pid] = pitcher_workload.fetch_pitcher_stats(pid, game_date)
        _tick(progress, "workload", i / max(len(pitcher_ids), 1))

    # 6. TTO splits (NEW)
    _tick(progress, "tto", 0.0)
    tto_data = {}
    for i, pid in enumerate(pitcher_ids, start=1):
        tto_data[pid] = pitcher_workload.fetch_tto_splits(pid, game_date)
        _tick(progress, "tto", i / max(len(pitcher_ids), 1))

    # 7. Team bullpen profiles (NEW)
    team_names = list(set(lineups_df["team"].dropna().tolist()))
    bullpens = bullpen_data.fetch_all_bullpens(
        team_names, game_date,
        progress_callback=lambda d, t: _tick(progress, "bullpens", d / max(t, 1)),
    )

    # 8. Umpire assignments (NEW)
    game_dicts = games_df.to_dict(orient="records")
    umpires = umpire_data.fetch_all_umpires(
        game_dicts, game_date,
        progress_callback=lambda d, t: _tick(progress, "umpires", d / max(t, 1)),
    )

    # 9. Build v2 matchups
    _tick(progress, "matchups", 0.0)
    game_lookup = games_df.set_index("game_pk").to_dict("index")

    matchups = []
    batter_to_game = {}
    batter_to_slot = {}
    bvp_annotations = {}

    for _, row in lineups_df.iterrows():
        bid = int(row["batter_id"])
        game_pk = int(row["game_pk"])
        team = row["team"]
        team_full = _normalize_team(team)  # Convert short name to full name
        game_info = game_lookup.get(game_pk, {})

        if team_full == game_info.get("home_team"):
            opp_pitcher_id = game_info.get("away_pitcher_id")
            opp_team = game_info.get("away_team")
        else:
            opp_pitcher_id = game_info.get("home_pitcher_id")
            opp_team = game_info.get("home_team")

        if pd.isna(opp_pitcher_id) or opp_pitcher_id is None:
            continue
        opp_pitcher_id = int(opp_pitcher_id)

        pitcher_info = pitcher_data.get(opp_pitcher_id)
        if pitcher_info is None:
            continue
        starter_throws, arsenal = pitcher_info

        batter_df = batter_profiles.get(bid, pd.DataFrame())
        bats = row.get("bats") or "R"
        slot = int(row["lineup_slot"]) if pd.notna(row["lineup_slot"]) else 5
        total_pa = PA_BY_LINEUP_SLOT.get(slot, DEFAULT_PA)

        ump_info = umpires.get(game_pk)
        ump_k_dev = ump_info.get("k_pct_dev") if ump_info else None

        pitcher_hr9 = (pitcher_stats_cache.get(opp_pitcher_id) or {}).get("hr_per_9")

        matchup = build_matchup_v2(
            batter_id=bid,
            batter_df=batter_df,
            starter_id=opp_pitcher_id,
            starter_throws=starter_throws,
            starter_arsenal=arsenal,
            starter_workload=workloads.get(opp_pitcher_id, {}),
            tto_splits=tto_data.get(opp_pitcher_id, {"xba": {}, "pa": {}}),
            bullpen_profile=bullpens.get(opp_team, {}),
            batter_stands=bats,
            lineup_slot=slot,
            total_pa=total_pa,
            venue=game_info.get("venue"),
            umpire_k_dev=ump_k_dev,
            as_of=game_date,
            pitcher_hr9=pitcher_hr9,
        )
        matchups.append(matchup)
        batter_to_game[bid] = game_pk
        batter_to_slot[bid] = slot

        # 10. BvP annotations (display only)
        if enable_bvp:
            bvp = bvp_data.compute_bvp(bid, opp_pitcher_id, batter_df, game_date)
            annotation = bvp_data.format_bvp_annotation(bvp)
            if annotation:
                bvp_annotations[bid] = annotation

    _tick(progress, "matchups", 1.0)

    # Simulate
    _tick(progress, "simulate", 0.0)
    sim_results = simulate_v2(matchups, batter_to_slot)
    preds_df = results_to_df_v2(sim_results)

    if not preds_df.empty:
        preds_df["game_pk"] = preds_df["batter_id"].map(batter_to_game)
        cache.save_predictions(game_date, preds_df)
    _tick(progress, "simulate", 1.0)

    matchup_details = {m.batter_id: m for m in matchups}

    return PipelineResultV2(
        game_date=game_date,
        games=games_df,
        lineups=lineups_df,
        predictions=preds_df,
        matchup_details=matchup_details,
        bvp_annotations=bvp_annotations,
        umpires=umpires,
        pitcher_stats=pitcher_stats_cache,
        summary={**summary, "n_matchups": len(matchups)},
    )


def _tick(progress, stage, frac):
    if progress is not None:
        try:
            progress(stage, frac)
        except Exception:
            pass
