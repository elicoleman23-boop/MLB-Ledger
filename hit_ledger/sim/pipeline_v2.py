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

from hit_ledger.config import (
    BVP_DEFAULT_ENABLED,
    DEFAULT_PA,
    N_SIMULATIONS,
    PA_BY_LINEUP_SLOT,
    PARK_FACTORS_HITS,
    PARK_FACTORS_HR,
    PBP_DEFAULT_N_SIMS,
    USE_PITCH_BY_PITCH_SIM,
)
from hit_ledger.data import cache, lineups, statcast
from hit_ledger.data import bullpen as bullpen_data
from hit_ledger.data import pitcher_profile as pitcher_profile_data
from hit_ledger.data import pitcher_workload
from hit_ledger.data import umpires as umpire_data
from hit_ledger.data import bvp as bvp_data
from hit_ledger.sim.engine_v2 import results_to_df_v2, simulate_v2
from hit_ledger.sim.matchup_v2 import build_matchup_v2
from hit_ledger.sim.pitch_sim import (
    build_batter_pitch_profile,
    build_pitcher_pitch_profile,
)
from hit_ledger.sim.pitch_sim.bullpen_aggregator import build_team_bullpen_pitch_profile
from hit_ledger.sim.pitch_sim_engine import sample_one_trace, simulate_pbp
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
    # Fix E Phase 3: populated only when use_pbp=True. batter_id → one
    # full PA trace (list of dicts from pa_engine.simulate_pa with each PA's
    # pitch_sequence) for UI inspection.
    pbp_sample_traces: dict = None  # type: ignore[assignment]


def run_daily_pipeline_v2(
    game_date: date,
    progress: ProgressCallback | None = None,
    force_refresh: bool = False,
    enable_bvp: bool = BVP_DEFAULT_ENABLED,
    use_pbp: bool = USE_PITCH_BY_PITCH_SIM,
    n_sims: int | None = None,
) -> PipelineResultV2:
    """Full v2 pipeline.

    When `use_pbp=True`, the fast PA-level engine is replaced by the
    pitch-by-pitch simulator (hit_ledger.sim.pitch_sim_engine.simulate_pbp).
    Everything upstream (data fetch, matchup build) is unchanged so a
    single pipeline supports both modes.
    """
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

    # "pitchers" stage covers both arsenal + pitch-level profile fetches.
    # Split the stage progress 50/50 between the two so the bar still
    # advances smoothly without needing a new UI stage weight.
    def _tick_pitchers_arsenal(done, total):
        _tick(progress, "pitchers", 0.5 * (done / max(total, 1)))

    def _tick_pitchers_profile(done, total):
        _tick(progress, "pitchers", 0.5 + 0.5 * (done / max(total, 1)))

    pitcher_data = statcast.fetch_all_pitchers(
        pitcher_ids, game_date,
        progress_callback=_tick_pitchers_arsenal,
    )

    # Pitcher pitch-level profiles (for log-5 blend in matchup_v2)
    pitcher_profiles = pitcher_profile_data.fetch_all_pitcher_profiles(
        pitcher_ids, game_date,
        progress_callback=_tick_pitchers_profile,
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
    # Per-batter pbp routing: opposing team for bullpen lookup, venue for park
    batter_to_opp_team: dict[int, str] = {}
    batter_to_venue: dict[int, str | None] = {}
    # Per-batter pbp starter profile selection
    batter_to_starter_id: dict[int, int] = {}

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
        pitcher_df = pitcher_profiles.get(opp_pitcher_id, pd.DataFrame())

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
            pitcher_df=pitcher_df,
        )
        matchups.append(matchup)
        batter_to_game[bid] = game_pk
        batter_to_slot[bid] = slot
        batter_to_opp_team[bid] = opp_team
        batter_to_venue[bid] = game_info.get("venue")
        batter_to_starter_id[bid] = opp_pitcher_id

        # 10. BvP annotations (display only)
        if enable_bvp:
            bvp = bvp_data.compute_bvp(bid, opp_pitcher_id, batter_df, game_date)
            annotation = bvp_data.format_bvp_annotation(bvp)
            if annotation:
                bvp_annotations[bid] = annotation

    _tick(progress, "matchups", 1.0)

    # Simulate — fast PA-level or pitch-by-pitch depending on use_pbp
    _tick(progress, "simulate", 0.0)
    pbp_sample_traces: dict = {}
    if use_pbp:
        # Build pitch profiles. Batter profiles come from the Statcast
        # frames we already fetched; pitcher profiles likewise.
        batter_pitch_profiles = {
            bid: build_batter_pitch_profile(df)
            for bid, df in batter_profiles.items()
        }
        pitcher_pitch_profiles: dict[int, dict] = {}
        for pid in pitcher_ids:
            df = pitcher_profiles.get(pid, pd.DataFrame())
            arsenal_info = pitcher_data.get(pid)
            arsenal = arsenal_info[1] if arsenal_info else {}
            pitcher_pitch_profiles[pid] = build_pitcher_pitch_profile(df, arsenal)

        # Stopgap bullpen profile per team: no per-reliever data yet, so
        # every team's bullpen gets the same league-prior profile.
        # (Fix F replaces this with per-reliever profiles + usage weighting.)
        team_names = set(batter_to_opp_team.values())
        team_bullpen_profiles = {
            team: build_team_bullpen_pitch_profile([], {}, {})
            for team in team_names
        }
        batter_bullpen_profiles = {
            bid: team_bullpen_profiles.get(team, {})
            for bid, team in batter_to_opp_team.items()
        }

        # Park multipliers per batter
        batter_park_mults: dict[int, tuple[float, float]] = {}
        for bid, venue in batter_to_venue.items():
            batter_park_mults[bid] = (
                PARK_FACTORS_HITS.get(venue or "", PARK_FACTORS_HITS["_default"]),
                PARK_FACTORS_HR.get(venue or "", PARK_FACTORS_HR["_default"]),
            )

        effective_n_sims = n_sims if n_sims is not None else PBP_DEFAULT_N_SIMS
        sim_results = simulate_pbp(
            matchups,
            batter_pitch_profiles=batter_pitch_profiles,
            pitcher_pitch_profiles=pitcher_pitch_profiles,
            batter_bullpen_profiles=batter_bullpen_profiles,
            lineup_slots=batter_to_slot,
            n_sims=effective_n_sims,
            batter_park_mults=batter_park_mults,
        )

        # One sample trace per batter, for UI inspection. Cheap: a single
        # PA-per-matchup-slot walk per batter, not n_sims worth.
        import numpy as np  # noqa: PLC0415 — tight local scope
        trace_rng = np.random.default_rng(7)
        for m in matchups:
            bid = m.batter_id
            pbp_sample_traces[bid] = sample_one_trace(
                matchup=m,
                batter_pitch_profile=batter_pitch_profiles.get(bid, {}),
                pitcher_pitch_profile=pitcher_pitch_profiles.get(m.starter_id, {}),
                bullpen_pitch_profile=batter_bullpen_profiles.get(bid, {}),
                rng=trace_rng,
                park_hit_mult=batter_park_mults.get(bid, (1.0, 1.0))[0],
                park_hr_mult=batter_park_mults.get(bid, (1.0, 1.0))[1],
            )
    else:
        effective_n_sims = n_sims if n_sims is not None else N_SIMULATIONS
        sim_results = simulate_v2(matchups, batter_to_slot, n_sims=effective_n_sims)

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
        summary={**summary, "n_matchups": len(matchups), "mode": "pbp" if use_pbp else "fast"},
        pbp_sample_traces=pbp_sample_traces,
    )


def _tick(progress, stage, frac):
    if progress is not None:
        try:
            progress(stage, frac)
        except Exception:
            pass
