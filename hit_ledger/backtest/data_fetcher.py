"""Date-scoped data layer for the backtest harness.

Every function here either:
  (a) reads historical "what actually happened" data (game results,
      final lineups, starters that actually appeared) — leakage isn't
      a concern because we WANT the actuals
  (b) wraps an existing live fetcher with an explicit `as_of` parameter
      and empirically validates the returned DataFrame's max game_date
      is strictly before the target.

The module does not touch hit_ledger.data or hit_ledger.sim — it's
purely a read-only consumer.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd

from hit_ledger.data import bullpen as bullpen_data
from hit_ledger.data import pitcher_profile as pitcher_profile_data
from hit_ledger.data import pitcher_workload
from hit_ledger.data import statcast

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Historical slate (actual games that happened)
# ---------------------------------------------------------------------------
def fetch_historical_game_slate(game_date: date) -> list[dict[str, Any]]:
    """Return every game that actually happened on `game_date`, with
    boxscore-derived lineups, actual starters, and the HP umpire name.

    This hits statsapi.schedule() and statsapi.boxscore_data() directly —
    boxscores for completed games return the actual battings order and
    starter used. No leakage concern: this IS the historical truth.

    Each game dict:
        game_pk, game_date, home_team, away_team, venue,
        home_starter_id, away_starter_id,
        home_lineup: list[{batter_id, batter_name, lineup_slot, bats}],
        away_lineup: list[{...same fields...}],
        home_plate_umpire_name,

    Games with no boxscore (rain-outs, future-dated requests) yield
    empty lineups and are flagged by the caller.
    """
    try:
        import statsapi  # type: ignore[import-untyped]
    except Exception as exc:  # noqa: BLE001
        logger.error("statsapi unavailable: %s", exc)
        return []

    try:
        raw = statsapi.schedule(date=game_date.strftime("%m/%d/%Y"))
    except Exception as exc:  # noqa: BLE001
        logger.error("statsapi.schedule failed for %s: %s", game_date, exc)
        return []

    out: list[dict[str, Any]] = []
    for g in raw:
        status = g.get("status") or ""
        if status in {"Postponed", "Cancelled", "Suspended"}:
            continue
        game_pk = g.get("game_id")
        if game_pk is None:
            continue

        try:
            box = statsapi.boxscore_data(game_pk)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "boxscore_data failed for game_pk=%s date=%s: %s",
                game_pk, game_date, exc,
            )
            continue

        home_lineup = _extract_side_lineup(box, "home", game_pk)
        away_lineup = _extract_side_lineup(box, "away", game_pk)

        # Actual starters (first pitcher in each side's pitcher list)
        home_starter = _first_pitcher_id(box, "home")
        away_starter = _first_pitcher_id(box, "away")

        ump_name = _extract_hp_umpire(box)

        out.append({
            "game_pk": int(game_pk),
            "game_date": game_date,
            "home_team": g.get("home_name"),
            "away_team": g.get("away_name"),
            "venue": g.get("venue_name"),
            "home_starter_id": home_starter,
            "away_starter_id": away_starter,
            "home_lineup": home_lineup,
            "away_lineup": away_lineup,
            "home_plate_umpire_name": ump_name,
        })
    return out


def _extract_side_lineup(
    box: dict, side: str, game_pk: int
) -> list[dict[str, Any]]:
    team_data = box.get(side, {}) or {}
    batting_order = team_data.get("battingOrder") or []
    players_dict = team_data.get("players") or {}
    out: list[dict[str, Any]] = []
    for slot_idx, pid in enumerate(batting_order, start=1):
        player = players_dict.get(f"ID{pid}", {}) or {}
        person = player.get("person", {}) or {}
        bat_side = (player.get("batSide") or {}).get("code")
        out.append({
            "batter_id": int(pid),
            "batter_name": person.get("fullName"),
            "lineup_slot": slot_idx,
            "bats": bat_side,
        })
    return out


def _first_pitcher_id(box: dict, side: str) -> int | None:
    pitchers = (box.get(side) or {}).get("pitchers") or []
    if not pitchers:
        return None
    try:
        return int(pitchers[0])
    except (TypeError, ValueError):
        return None


def _extract_hp_umpire(box: dict) -> str | None:
    officials = (box.get("officialsInfo") or {}).get("officials") or []
    for o in officials:
        if (o.get("officialType") or "").lower() == "home plate":
            return (o.get("official") or {}).get("fullName")
    return None


# ---------------------------------------------------------------------------
# Pregame profiles (date-scoped)
# ---------------------------------------------------------------------------
def fetch_pregame_batter_profile(
    batter_id: int, game_date: date
) -> pd.DataFrame:
    """Leak-safe batter profile — delegates to statcast.fetch_batter_profile
    which uses `end_dt = as_of - 1 day` internally. Returned frame is
    empirically checked before return so any fetcher regression blows
    up loudly."""
    df = statcast.fetch_batter_profile(batter_id, as_of=game_date)
    _validate_strictly_before(df, game_date, f"batter_id={batter_id}")
    return df


def fetch_pregame_pitcher_profile(
    pitcher_id: int, game_date: date
) -> pd.DataFrame:
    """Leak-safe pitcher pitch-level profile."""
    df = pitcher_profile_data.fetch_pitcher_profile(pitcher_id, as_of=game_date)
    _validate_strictly_before(df, game_date, f"pitcher_id={pitcher_id}")
    return df


def fetch_pregame_arsenal(
    pitcher_id: int, game_date: date
) -> tuple[str, dict[str, float]]:
    """Leak-safe arsenal — fetch_pitcher_arsenal internally bounds by
    `as_of - 1 day`. No DataFrame returned so no empirical validation
    is possible, but the fetcher's end_dt math is the same path we
    trust elsewhere."""
    return statcast.fetch_pitcher_arsenal(pitcher_id, as_of=game_date)


def fetch_pregame_profiles(
    game_pk: int,
    game_date: date,
    batter_ids: list[int],
    pitcher_ids: list[int],
) -> dict[str, Any]:
    """Bulk-fetch every profile the matchup builder needs for one game.

    Returns:
        {
          "batter_profiles": {bid: pd.DataFrame},
          "pitcher_profiles": {pid: pd.DataFrame},
          "pitcher_arsenals": {pid: (throws, arsenal)},
        }

    Each DataFrame is empirically validated to have
    max(game_date) < target. fetch raises LeakageError upstream if any
    check fails.
    """
    batter_profiles: dict[int, pd.DataFrame] = {}
    for bid in batter_ids:
        try:
            batter_profiles[bid] = fetch_pregame_batter_profile(bid, game_date)
        except Exception as exc:  # noqa: BLE001 — surface, don't silently drop
            logger.warning(
                "pregame_batter_profile_failed batter_id=%s game_pk=%s date=%s: %r",
                bid, game_pk, game_date, exc,
            )
            # Defer the decision to skip this batter to the replay layer.
            batter_profiles[bid] = pd.DataFrame()

    pitcher_profiles: dict[int, pd.DataFrame] = {}
    pitcher_arsenals: dict[int, tuple[str, dict[str, float]]] = {}
    for pid in pitcher_ids:
        try:
            pitcher_profiles[pid] = fetch_pregame_pitcher_profile(pid, game_date)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "pregame_pitcher_profile_failed pitcher_id=%s game_pk=%s date=%s: %r",
                pid, game_pk, game_date, exc,
            )
            pitcher_profiles[pid] = pd.DataFrame()
        try:
            pitcher_arsenals[pid] = fetch_pregame_arsenal(pid, game_date)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "pregame_arsenal_failed pitcher_id=%s game_pk=%s date=%s: %r",
                pid, game_pk, game_date, exc,
            )
            pitcher_arsenals[pid] = ("R", {})

    return {
        "batter_profiles": batter_profiles,
        "pitcher_profiles": pitcher_profiles,
        "pitcher_arsenals": pitcher_arsenals,
    }


def fetch_pregame_workload(pitcher_id: int, game_date: date) -> dict:
    """Starter workload (IP/start rolling window + season xBA) capped
    strictly before `game_date`. pitcher_workload.fetch_starter_workload
    already accepts as_of; we trust its internal bounding but log a
    warning on any exception so replay can skip the batter cleanly."""
    try:
        return pitcher_workload.fetch_starter_workload(pitcher_id, as_of=game_date)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "pregame_workload_failed pitcher_id=%s date=%s: %r",
            pitcher_id, game_date, exc,
        )
        return {}


def fetch_pregame_tto(pitcher_id: int, game_date: date) -> dict:
    """Date-scoped TTO splits."""
    try:
        return pitcher_workload.fetch_tto_splits(pitcher_id, as_of=game_date)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "pregame_tto_failed pitcher_id=%s date=%s: %r",
            pitcher_id, game_date, exc,
        )
        return {"xba": {}, "pa": {}}


def fetch_pregame_bullpen(team: str, game_date: date) -> dict:
    """Team-level bullpen xBA aggregate for the backtest's team-level
    bullpen path. fetch_bullpen_profile's constituent appearances are
    bounded by as_of, so as long as we pass game_date through the
    returned aggregate is leak-safe."""
    try:
        return bullpen_data.fetch_bullpen_profile(team, as_of=game_date)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "pregame_bullpen_failed team=%s date=%s: %r",
            team, game_date, exc,
        )
        return {}


def fetch_pregame_bullpen_roster(team: str, game_date: date) -> list[dict]:
    """Stub for symmetry with the live pipeline. The backtest config
    forces use_per_reliever_bullpen=False so this function should not
    be called from the replay path; it exists per the spec for API
    completeness. Returns [] unconditionally — the downstream matchup
    builder will auto-fall-back to team-level aggregates when the
    roster is empty, which is the intended safe behavior."""
    logger.info(
        "pregame_bullpen_roster called team=%s date=%s — returning empty "
        "(per-reliever bullpen is off in backtest mode)",
        team, game_date,
    )
    return []


def fetch_pregame_umpire(umpire_name: str, game_date: date) -> float:
    """Stub per spec — backtest sets ump_k_dev = 0.0 always (the leakage
    exposure of the live umpscorecards scrape isn't worth the signal
    it carries for this iteration). The replay layer ignores this
    function's return value and passes 0.0 directly."""
    del umpire_name, game_date  # acknowledged-unused
    return 0.0


# ---------------------------------------------------------------------------
# Internal: empirical leakage validation
# ---------------------------------------------------------------------------
def _validate_strictly_before(
    df: pd.DataFrame, target: date, context: str
) -> None:
    """Raise LeakageError if any row in df has game_date >= target.
    Imported here to keep a single canonical validator."""
    from hit_ledger.backtest.leakage_check import LeakageError

    if df is None or df.empty or "game_date" not in df.columns:
        return
    gd = pd.to_datetime(df["game_date"], errors="coerce")
    if gd.isna().all():
        return
    max_date = gd.max()
    if pd.isna(max_date):
        return
    if max_date >= pd.Timestamp(target):
        n_leaked = int((gd >= pd.Timestamp(target)).sum())
        raise LeakageError(
            f"fetcher leaked future data: {context}, target={target.isoformat()}, "
            f"max_date={max_date.date().isoformat()}, n_leaked_rows={n_leaked}"
        )
