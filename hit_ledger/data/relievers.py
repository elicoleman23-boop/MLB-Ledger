"""
Fix F — per-team bullpen roster + usage-probability heuristic.

`fetch_team_bullpen_roster` pulls the active relievers off a team's
roster and decorates each with workload/leverage metadata the usage
heuristic needs. `predict_reliever_usage_probs` is the pure-function
heuristic; no I/O, easy to test.

Both are designed to degrade gracefully when the external data sources
(MLB StatsAPI, pybaseball) are unavailable — on failure they emit a
structured warning via the module logger and return empty/fallback
values so the caller can branch on "no data" rather than crash.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from hit_ledger.data import cache

logger = logging.getLogger(__name__)

# Cache staleness: roster data refetches if older than 7 days. The cache
# key already includes game_date, so day-over-day we'll naturally get a
# fresh fetch; this is only a safety net for "don't trust a 2-week-old
# roster for today's lineups."
_ROSTER_MAX_STALE_DAYS = 7

# Leverage bands we use to classify roles
_MIDDLE_LI = (0.8, 1.3)
_SETUP_LI = (1.3, 2.0)
_HIGH_LEVERAGE_MIN = 1.5

# Usage heuristic constants
_SAME_HAND_BONUS = 1.3        # LHP vs LHH or RHP vs RHH
_CLOSER_9TH_WEIGHT = 0.65     # share the closer captures in the 9th when available
_MIN_APPS_FOR_LEVERAGE_ROLE = 5  # below this, a reliever can't be labeled closer / HL


# ---------------------------------------------------------------------------
# Roster fetch
# ---------------------------------------------------------------------------
def fetch_team_bullpen_roster(
    team: str,
    as_of: date,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """
    Return a list of reliever dicts for the team's active bullpen on
    `as_of`. Each dict:

        {
            "player_id":              int,
            "name":                   str,
            "throws":                 "L" | "R",
            "recent_ip":              float,    # last 15 days
            "recent_appearances":     int,      # last 15 days
            "avg_leverage_index":     float,    # season (fall back to 1.0)
            "is_closer":              bool,     # top save-leader on team
            "is_high_leverage":       bool,     # avg_LI >= 1.5
            "days_since_last_app":    int,      # large number if unknown / DL
            "back_to_back":           bool,     # pitched both of last 2 days
        }

    Empty list on failure — caller is expected to fall back to the
    team-level aggregator.

    Cached in SQLite per (game_date, team); refetch on force_refresh.
    """
    cached = cache.load_team_bullpen_roster(as_of, team)
    if cached is not None and not force_refresh:
        return cached

    try:
        roster = _scrape_team_bullpen_roster(team, as_of)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "bullpen_roster_fetch_failed team=%s as_of=%s error=%r — "
            "returning empty list; caller should fall back to team-level bullpen",
            team, as_of.isoformat(), exc,
        )
        roster = []

    cache.save_team_bullpen_roster(as_of, team, roster)
    return roster


def _scrape_team_bullpen_roster(team: str, as_of: date) -> list[dict[str, Any]]:
    """Pull 28-man roster from MLB StatsAPI, filter to pitchers, decorate
    each with workload + leverage metadata. This is the only place that
    actually talks to pybaseball / statsapi; wrapped in its own function
    so the outer fetcher can catch the whole blast radius and fall back."""
    import statsapi  # type: ignore[import-untyped]

    # Look up team_id by name (StatsAPI's lookup_team is the canonical path)
    team_records = statsapi.lookup_team(team)
    if not team_records:
        logger.warning("bullpen_roster_team_lookup_empty team=%s", team)
        return []
    team_id = team_records[0]["id"]

    # 28-man roster — pitchers only
    roster_payload = statsapi.roster(team_id, rosterType="active", date=as_of.isoformat())
    pitcher_ids: list[int] = []
    id_to_name: dict[int, str] = {}
    id_to_throws: dict[int, str] = {}
    # statsapi.roster returns a formatted string; for the raw payload we
    # go through get() instead:
    try:
        raw = statsapi.get(
            "team_roster",
            {"teamId": team_id, "rosterType": "active", "date": as_of.isoformat()},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "bullpen_roster_get_failed team=%s team_id=%s error=%r",
            team, team_id, exc,
        )
        return []

    for entry in raw.get("roster", []):
        position = entry.get("position", {}).get("abbreviation") or ""
        if position != "P":
            continue
        person = entry.get("person", {}) or {}
        pid = person.get("id")
        if pid is None:
            continue
        pitcher_ids.append(int(pid))
        id_to_name[int(pid)] = person.get("fullName") or f"Pitcher #{pid}"
        # Throws lives under the person-level lookup, not roster. Look up per-player:
        try:
            player_data = statsapi.lookup_player(person.get("fullName") or str(pid))
            if player_data:
                hand = (
                    (player_data[0].get("pitchHand") or {}).get("code")
                    or "R"
                )
                id_to_throws[int(pid)] = hand
            else:
                id_to_throws[int(pid)] = "R"
        except Exception:  # noqa: BLE001
            id_to_throws[int(pid)] = "R"

    if not pitcher_ids:
        return []

    # Workload: use pybaseball game logs to derive recent IP / appearances
    # / days since last appearance / back_to_back. This is the expensive
    # per-pitcher loop; cache is what makes this tractable day-to-day.
    roster: list[dict[str, Any]] = []
    for pid in pitcher_ids:
        workload = _fetch_pitcher_workload_window(pid, as_of)
        roster.append({
            "player_id": pid,
            "name": id_to_name.get(pid, f"Pitcher #{pid}"),
            "throws": id_to_throws.get(pid, "R"),
            **workload,
        })

    # Post-process: identify closer (most saves in last 30d among team)
    _tag_closer(roster)

    return roster


def _fetch_pitcher_workload_window(
    pitcher_id: int, as_of: date
) -> dict[str, Any]:
    """Last-15-day IP + appearances, days since last app, back-to-back
    flag, and season-avg LI. All fields degrade to neutral defaults on
    any failure — the usage heuristic tolerates zeros."""
    defaults = {
        "recent_ip": 0.0,
        "recent_appearances": 0,
        "avg_leverage_index": 1.0,
        "is_closer": False,
        "is_high_leverage": False,
        "days_since_last_app": 999,
        "back_to_back": False,
        "season_saves": 0,
    }
    try:
        from pybaseball import statcast_pitcher
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "workload_window_pybaseball_unavailable pitcher_id=%s error=%r",
            pitcher_id, exc,
        )
        return defaults

    try:
        start = as_of - timedelta(days=15)
        df = statcast_pitcher(
            start_dt=start.isoformat(),
            end_dt=(as_of - timedelta(days=1)).isoformat(),
            player_id=pitcher_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "workload_window_statcast_failed pitcher_id=%s error=%r",
            pitcher_id, exc,
        )
        return defaults

    if df is None or len(df) == 0 or "game_date" not in df.columns:
        return defaults

    import pandas as pd  # noqa: PLC0415 — tight local scope
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date"])
    if df.empty:
        return defaults

    # Recent appearances: unique game dates with >=1 pitch. IP approx by
    # outs; lacking outs columns we approximate appearances only and use
    # pitches as a proxy IP contribution (later fixes will replace this
    # with real IP via pybaseball.pitching_stats_range).
    app_dates = df["game_date"].dt.date.unique().tolist()
    app_dates.sort()
    n_apps = len(app_dates)
    # Rough IP: each appearance contributes median-length-outing ~1 inning.
    # When we have outs-column data this is better — for now, apps × 1 IP.
    recent_ip = float(n_apps) * 1.0

    last_app = app_dates[-1] if app_dates else None
    yesterday = as_of - timedelta(days=1)
    two_days_ago = as_of - timedelta(days=2)
    days_since = (as_of - last_app).days if last_app else 999
    back_to_back = (
        yesterday in app_dates and two_days_ago in app_dates
    )

    return {
        "recent_ip": recent_ip,
        "recent_appearances": n_apps,
        # We don't have LI from Statcast pitch-level. Default to 1.0
        # (league-neutral); a later fix can upgrade this using fangraphs
        # bullpen splits.
        "avg_leverage_index": 1.0,
        "is_closer": False,       # filled in by _tag_closer
        "is_high_leverage": False,  # filled in by _tag_closer
        "days_since_last_app": days_since,
        "back_to_back": back_to_back,
        "season_saves": 0,          # TODO: fetch via pybaseball.pitching_stats
    }


def _tag_closer(roster: list[dict[str, Any]]) -> None:
    """Pick the single pitcher with the most saves as the closer, and
    anyone with avg_LI >= 1.5 as high-leverage. Mutates roster in place."""
    if not roster:
        return
    # Closer: max season_saves among those with enough appearances
    eligible = [r for r in roster if r.get("recent_appearances", 0) >= _MIN_APPS_FOR_LEVERAGE_ROLE]
    if eligible:
        closer = max(eligible, key=lambda r: r.get("season_saves", 0))
        if closer.get("season_saves", 0) > 0:
            closer["is_closer"] = True
    # High leverage: avg_LI threshold
    for r in roster:
        if r.get("recent_appearances", 0) >= _MIN_APPS_FOR_LEVERAGE_ROLE:
            r["is_high_leverage"] = r.get("avg_leverage_index", 1.0) >= _HIGH_LEVERAGE_MIN


def fetch_all_team_rosters(
    teams: list[str],
    as_of: date,
    force_refresh: bool = False,
    progress_callback=None,
) -> dict[str, list[dict[str, Any]]]:
    """Batch fetcher with a progress callback — rosters come in one
    team at a time because the per-pitcher workload loop is serial."""
    out: dict[str, list[dict[str, Any]]] = {}
    total = len(teams)
    for i, team in enumerate(teams, start=1):
        out[team] = fetch_team_bullpen_roster(team, as_of, force_refresh=force_refresh)
        if progress_callback:
            progress_callback(i, total)
    return out


# ---------------------------------------------------------------------------
# Usage prediction heuristic (pure)
# ---------------------------------------------------------------------------
def predict_reliever_usage_probs(
    bullpen_roster: list[dict[str, Any]],
    pa_index_in_game: int,
    expected_inning: int,
    batter_stands: str,
    top_n: int = 6,
) -> dict[int, float]:
    """
    Return {pitcher_id: usage_prob} for the top-N most likely relievers to
    pitch this PA. Probabilities renormalize to 1.0 across the returned
    set. If no relievers are eligible, returns {} (caller should fall
    back to team-level).

    Heuristic (v1):
      - Ineligible: back_to_back OR days_since_last_app < 1.
      - Inning 9: closer captures 0.65 share if available AND eligible;
        remainder splits among high-leverage arms by LI.
      - Inning 8: setup men (LI in [1.3, 2.0], not closer) by LI and
        inverse days-rest.
      - Innings 6-7: middle relievers (LI in [0.8, 1.3]) by inverse rest.
      - Anything outside (innings 5-, extras): fall back to any eligible
        arm weighted by inverse rest.
      - Same-handedness bonus: LHP vs LHH (or RHP vs RHH) → weight × 1.3.

    `pa_index_in_game` is retained for future extension (e.g. conditional
    rest within a game) but unused at v1 because expected_inning already
    captures the relevant game-state.
    """
    del pa_index_in_game  # reserved for future use
    if not bullpen_roster:
        return {}

    eligible = [
        r for r in bullpen_roster
        if not r.get("back_to_back", False)
        and r.get("days_since_last_app", 999) >= 1
    ]
    if not eligible:
        return {}

    weights: dict[int, float] = {}

    if expected_inning >= 9:
        _weight_late_inning(eligible, weights, is_closer_inning=True)
    elif expected_inning == 8:
        _weight_setup_inning(eligible, weights)
    elif expected_inning in (6, 7):
        _weight_middle_inning(eligible, weights)
    else:
        # Early or extra innings — any eligible arm, weighted by rest
        for r in eligible:
            weights[r["player_id"]] = _rest_weight(r)

    # Same-handedness platoon bonus
    for pid in list(weights.keys()):
        entry = next((r for r in eligible if r["player_id"] == pid), None)
        if entry and _same_hand(entry.get("throws", "R"), batter_stands):
            weights[pid] *= _SAME_HAND_BONUS

    if not weights:
        return {}

    # Top-N, renormalize
    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    total = sum(w for _, w in ranked)
    if total <= 0:
        return {}
    return {pid: float(w / total) for pid, w in ranked}


def _rest_weight(reliever: dict[str, Any]) -> float:
    """Inverse-rest weighting: pitchers who rested longer get a larger
    weight (they're fresher). We use 1 / (1 + days_since_last_app) to
    avoid dividing by zero and keep the function bounded."""
    return 1.0 / (1.0 + float(reliever.get("days_since_last_app", 1)))


def _same_hand(throws: str, batter_stands: str) -> bool:
    """LHP vs LHH or RHP vs RHH. Switch hitters don't qualify."""
    if not throws or not batter_stands:
        return False
    return throws.upper() == batter_stands.upper()


def _weight_late_inning(
    eligible: list[dict[str, Any]],
    weights: dict[int, float],
    is_closer_inning: bool,
) -> None:
    closer = next((r for r in eligible if r.get("is_closer")), None)
    if is_closer_inning and closer is not None:
        weights[closer["player_id"]] = _CLOSER_9TH_WEIGHT
        remaining_budget = 1.0 - _CLOSER_9TH_WEIGHT
    else:
        remaining_budget = 1.0
    # High-leverage arms (non-closer) share the remainder by LI × rest
    pool = [r for r in eligible if not r.get("is_closer") and r.get("is_high_leverage")]
    if not pool:
        # Fall back to any eligible non-closer
        pool = [r for r in eligible if not r.get("is_closer")]
    if pool:
        raw = {r["player_id"]: r.get("avg_leverage_index", 1.0) * _rest_weight(r) for r in pool}
        total = sum(raw.values())
        if total > 0:
            for pid, w in raw.items():
                weights[pid] = weights.get(pid, 0.0) + remaining_budget * (w / total)


def _weight_setup_inning(
    eligible: list[dict[str, Any]],
    weights: dict[int, float],
) -> None:
    pool = [
        r for r in eligible
        if not r.get("is_closer")
        and _SETUP_LI[0] <= r.get("avg_leverage_index", 1.0) < _SETUP_LI[1]
    ]
    if not pool:
        pool = [r for r in eligible if not r.get("is_closer") and r.get("is_high_leverage")]
    if not pool:
        pool = [r for r in eligible if not r.get("is_closer")]
    for r in pool:
        weights[r["player_id"]] = (
            r.get("avg_leverage_index", 1.0) * _rest_weight(r)
        )


def _weight_middle_inning(
    eligible: list[dict[str, Any]],
    weights: dict[int, float],
) -> None:
    pool = [
        r for r in eligible
        if _MIDDLE_LI[0] <= r.get("avg_leverage_index", 1.0) <= _MIDDLE_LI[1]
    ]
    if not pool:
        pool = eligible
    for r in pool:
        weights[r["player_id"]] = _rest_weight(r)
