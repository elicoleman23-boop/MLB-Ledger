"""Replay engine — takes one historical game slate and produces model
predictions as if we'd run the model pregame.

`replay_game` is the per-game workhorse. It builds matchups for every
batter in both lineups, routes them through the same
`simulate_v2` the live fast engine uses, and returns a flat list of
per-batter prediction dicts. Skips (missing lineup, fetch failure) are
logged but not escalated — the runner aggregates skip reasons for the
final summary. Leakage, on the other hand, raises immediately: one
bad fetch poisons the entire slate.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from hit_ledger.config import DEFAULT_PA, PA_BY_LINEUP_SLOT
from hit_ledger.data import lineups as lineups_data
from hit_ledger.sim.engine_v2 import simulate_v2
from hit_ledger.sim.matchup_v2 import build_matchup_v2

from hit_ledger.backtest.config import BacktestConfig
from hit_ledger.backtest.data_fetcher import (
    fetch_historical_game_slate,
    fetch_pregame_bullpen,
    fetch_pregame_profiles,
    fetch_pregame_tto,
    fetch_pregame_workload,
)
from hit_ledger.backtest.leakage_check import assert_no_leakage

logger = logging.getLogger(__name__)


def replay_game(
    game: dict[str, Any],
    config: BacktestConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Produce pregame predictions for every batter in both lineups of
    one historical game.

    Returns `(predictions, skipped)` where `skipped` is a list of
    `{batter_id, reason}` dicts for batters that couldn't be scored
    (missing profile, matchup build failure, etc).
    """
    predictions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    game_pk = int(game["game_pk"])
    game_date = game["game_date"]
    home_team = game.get("home_team") or ""
    away_team = game.get("away_team") or ""
    venue = game.get("venue") or ""
    home_starter = game.get("home_starter_id")
    away_starter = game.get("away_starter_id")

    # Skip the whole game if either starter is unknown — we can't build
    # matchups without the opposing pitcher.
    if home_starter is None or away_starter is None:
        reason = "missing_starter"
        for side_lineup, _ in (
            (game.get("home_lineup") or [], "home"),
            (game.get("away_lineup") or [], "away"),
        ):
            for entry in side_lineup:
                skipped.append({"batter_id": entry["batter_id"], "reason": reason})
        logger.warning(
            "skip_game_missing_starter game_pk=%s home_starter=%s away_starter=%s",
            game_pk, home_starter, away_starter,
        )
        return predictions, skipped

    home_lineup = game.get("home_lineup") or []
    away_lineup = game.get("away_lineup") or []
    if not home_lineup or not away_lineup:
        reason = "missing_lineup"
        for entry in home_lineup + away_lineup:
            skipped.append({"batter_id": entry["batter_id"], "reason": reason})
        logger.warning(
            "skip_game_missing_lineup game_pk=%s home_n=%d away_n=%d",
            game_pk, len(home_lineup), len(away_lineup),
        )
        return predictions, skipped

    # Fetch pregame profiles — date-scoped & empirically leak-checked.
    all_batter_ids = [e["batter_id"] for e in (home_lineup + away_lineup)]
    pitcher_ids = [home_starter, away_starter]

    profile_bundle = fetch_pregame_profiles(
        game_pk=game_pk,
        game_date=game_date,
        batter_ids=all_batter_ids,
        pitcher_ids=pitcher_ids,
    )
    batter_profiles: dict[int, pd.DataFrame] = profile_bundle["batter_profiles"]
    pitcher_profiles: dict[int, pd.DataFrame] = profile_bundle["pitcher_profiles"]
    pitcher_arsenals: dict[int, tuple[str, dict]] = profile_bundle["pitcher_arsenals"]

    # Hard leakage check across every fetched DataFrame — one bad fetch
    # raises. This is the empirical guard; the cache-scoping is the
    # structural one. Both together is the invariant.
    leakage_inputs: dict[str, Any] = {
        f"batter_{bid}": df for bid, df in batter_profiles.items()
    }
    leakage_inputs.update({
        f"pitcher_{pid}": df for pid, df in pitcher_profiles.items()
    })
    assert_no_leakage(game_date, leakage_inputs)

    # Pregame workload / TTO / bullpen per team.
    workloads = {pid: fetch_pregame_workload(pid, game_date) for pid in pitcher_ids}
    tto_splits = {pid: fetch_pregame_tto(pid, game_date) for pid in pitcher_ids}
    bullpen_profiles: dict[str, dict] = {
        home_team: fetch_pregame_bullpen(home_team, game_date),
        away_team: fetch_pregame_bullpen(away_team, game_date),
    }

    # Pitcher handedness — the live `build_matchup_v2` takes
    # `starter_throws` as a string. Prefer the arsenal fetcher's
    # observed value; fall back to a StatsAPI lookup; worst case "R".
    pitcher_throws_map: dict[int, str] = {}
    for pid in pitcher_ids:
        arsenal_info = pitcher_arsenals.get(pid)
        if arsenal_info and arsenal_info[0]:
            pitcher_throws_map[pid] = arsenal_info[0]
            continue
        lookup = lineups_data.fetch_pitcher_handedness(pid)
        pitcher_throws_map[pid] = lookup or "R"

    # Build matchups for every batter in both lineups.
    matchups = []
    batter_context: dict[int, dict[str, Any]] = {}
    batter_to_slot: dict[int, int] = {}
    for side, lineup, team, opp_team, starter_id in (
        ("home", home_lineup, home_team, away_team, away_starter),
        ("away", away_lineup, away_team, home_team, home_starter),
    ):
        opp_starter_info = pitcher_arsenals.get(starter_id, ("R", {}))
        starter_throws = pitcher_throws_map.get(starter_id, "R")
        starter_arsenal = opp_starter_info[1] or {"FF": 0.55, "SL": 0.25, "CH": 0.20}
        pitcher_df = pitcher_profiles.get(starter_id, pd.DataFrame())
        opp_bullpen = bullpen_profiles.get(opp_team, {})

        for entry in lineup:
            bid = entry["batter_id"]
            bats = entry.get("bats") or "R"
            slot = int(entry.get("lineup_slot") or 5)
            total_pa = PA_BY_LINEUP_SLOT.get(slot, DEFAULT_PA)

            batter_df = batter_profiles.get(bid)
            if batter_df is None or batter_df.empty:
                skipped.append({"batter_id": bid, "reason": "batter_profile_empty"})
                continue

            try:
                matchup = build_matchup_v2(
                    batter_id=bid,
                    batter_df=batter_df,
                    starter_id=starter_id,
                    starter_throws=starter_throws,
                    starter_arsenal=starter_arsenal,
                    starter_workload=workloads.get(starter_id, {}),
                    tto_splits=tto_splits.get(starter_id, {"xba": {}, "pa": {}}),
                    bullpen_profile=opp_bullpen,
                    batter_stands=bats,
                    lineup_slot=slot,
                    total_pa=total_pa,
                    venue=venue,
                    umpire_k_dev=0.0,   # disabled in backtest per config decision
                    as_of=game_date,
                    pitcher_df=pitcher_df,
                    # Per-reliever bullpen is off in backtest (config asserts it)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "build_matchup_failed game_pk=%s batter_id=%s: %r",
                    game_pk, bid, exc,
                )
                skipped.append({"batter_id": bid, "reason": "build_matchup_failed"})
                continue

            matchups.append(matchup)
            batter_to_slot[bid] = slot
            batter_context[bid] = {
                "game_pk": game_pk,
                "game_date": game_date,
                "batter_id": bid,
                "batter_name": entry.get("batter_name"),
                "team": team,
                "opp_team": opp_team,
                "starter_id": starter_id,
                "lineup_slot": slot,
                "venue": venue,
                "umpire_k_dev": 0.0,
                "batter_profile_n_pa": _count_pa_ending(batter_df),
                "pitcher_profile_n_pa": _count_pa_ending(pitcher_df),
                "data_quality": getattr(matchup, "data_quality", "good"),
                "expected_pa_vs_starter": matchup.expected_pa_vs_starter,
                "expected_pa_vs_bullpen": matchup.expected_pa_vs_bullpen,
                "batter_stands": bats,
                "pitcher_throws": starter_throws,
                "starter_arsenal_summary": _top3_arsenal(starter_arsenal),
            }

    if not matchups:
        return predictions, skipped

    # Single sim call across the whole game.
    rng = np.random.default_rng(config.rng_seed)
    sim_results = simulate_v2(
        matchups,
        batter_to_slot,
        n_sims=config.n_sims,
        rng=rng,
        babip_noise_sd=0.0,  # deterministic path for reproducibility
    )

    for res in sim_results:
        ctx = batter_context.get(res.batter_id, {})
        predictions.append({
            **ctx,
            "pred_p_1_hit":       float(res.p_1_hit),
            "pred_p_2_hits":      float(res.p_2_hits),
            "pred_p_1_hr":        float(res.p_1_hr),
            "pred_expected_hits": float(res.expected_hits),
            "pred_expected_tb":   float(res.expected_tb),
            "pred_p_tb_over_1_5": float(res.p_tb_over_1_5),
            "pred_p_tb_over_2_5": float(res.p_tb_over_2_5),
        })

    return predictions, skipped


def replay_slate(
    slate_date: date,
    config: BacktestConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Replay every game on `slate_date`. Returns
    `(predictions, skipped, n_games_processed)`."""
    games = fetch_historical_game_slate(slate_date)
    if config.teams:
        teams = set(config.teams)
        games = [
            g for g in games
            if (g.get("home_team") in teams) or (g.get("away_team") in teams)
        ]
    if config.max_games_per_day is not None:
        games = games[: config.max_games_per_day]

    all_predictions: list[dict[str, Any]] = []
    all_skipped: list[dict[str, Any]] = []

    for g in games:
        try:
            preds, skipped = replay_game(g, config)
        except Exception as exc:  # noqa: BLE001 — LeakageError is caught here too
            # Re-raise leakage errors unconditionally; they're fatal.
            from hit_ledger.backtest.leakage_check import LeakageError
            if isinstance(exc, LeakageError):
                raise
            logger.warning(
                "replay_game_crashed game_pk=%s date=%s: %r",
                g.get("game_pk"), slate_date, exc,
            )
            # Game-level crash → skip every batter with a common reason
            skip_ids = [
                e["batter_id"] for e in (
                    (g.get("home_lineup") or []) + (g.get("away_lineup") or [])
                )
            ]
            all_skipped.extend(
                {"batter_id": bid, "reason": "game_crash"} for bid in skip_ids
            )
            continue
        all_predictions.extend(preds)
        all_skipped.extend(skipped)

    return all_predictions, all_skipped, len(games)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _count_pa_ending(df: pd.DataFrame) -> int:
    """Number of PA-ending rows in a Statcast frame — fed into diagnostics
    so we can slice residuals by sample depth later."""
    if df is None or df.empty or "events" not in df.columns:
        return 0
    return int((df["events"].notna() & (df["events"] != "")).sum())


def _top3_arsenal(arsenal: dict[str, float]) -> dict[str, float]:
    if not arsenal:
        return {}
    return dict(sorted(arsenal.items(), key=lambda kv: kv[1], reverse=True)[:3])
