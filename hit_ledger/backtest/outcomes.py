"""Extract actual batter outcomes from a completed game.

Uses MLB StatsAPI's `boxscore_data` (player-level totals) and `get`
against the game feed endpoint (per-play detail). The two sources are
independent: boxscore is authoritative for counts like H/HR/TB, while
the play feed carries per-PA context (pitcher, inning, outcome) that
lets Phase 2 analysis validate starter-vs-bullpen routing.

All parsing is defensive — any missing field defaults to zero/None
rather than raising, and per-batter extraction returns `played=False`
with zero counts when the box has no entry for that player.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# StatsAPI at-bat result -> normalized outcome tag
_RESULT_TO_OUTCOME = {
    "single":         "1B",
    "double":         "2B",
    "triple":         "3B",
    "home_run":       "HR",
    "walk":           "BB",
    "intent_walk":    "BB",
    "hit_by_pitch":   "HBP",
    "strikeout":      "K",
    "strikeout_double_play": "K",
    "strikeout_triple_play": "K",
    "field_out":      "OUT",
    "force_out":      "OUT",
    "grounded_into_double_play": "OUT",
    "grounded_into_triple_play": "OUT",
    "double_play":    "OUT",
    "triple_play":    "OUT",
    "fielders_choice": "OUT",
    "fielders_choice_out": "OUT",
    "sac_fly":        "OUT",
    "sac_fly_double_play": "OUT",
    "sac_bunt":       "OUT",
    "catcher_interf": "OUT",
    "pickoff_1b":     "OUT",
    "pickoff_2b":     "OUT",
    "pickoff_3b":     "OUT",
    # Anything else falls through to "OUT" via `.get(..., "OUT")`.
}


def _empty_outcome(lineup_slot: int | None = None) -> dict[str, Any]:
    return {
        "pa_count": 0,
        "hits": 0,
        "hrs": 0,
        "total_bases": 0,
        "singles": 0,
        "doubles": 0,
        "triples": 0,
        "strikeouts": 0,
        "walks": 0,
        "hbp": 0,
        "lineup_slot": lineup_slot,
        "pa_sequence": [],
        "played": False,
    }


def extract_batter_outcomes(game_pk: int, batter_id: int) -> dict[str, Any]:
    """Return the actual game-level outcomes for `batter_id` in `game_pk`.

    On any fetch failure or missing player entry, returns an empty
    outcome dict with `played=False` — this is expected for scratches,
    pinch-hit-only batters not in the lineup, or data-feed lag.
    """
    try:
        import statsapi  # type: ignore[import-untyped]
    except Exception as exc:  # noqa: BLE001
        logger.error("statsapi unavailable: %s", exc)
        return _empty_outcome()

    try:
        box = statsapi.boxscore_data(game_pk)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "outcomes_boxscore_failed game_pk=%s batter_id=%s: %r",
            game_pk, batter_id, exc,
        )
        return _empty_outcome()

    # Find the player in either home or away. battingOrder records the
    # starting lineup slot; players_dict has per-player stats.
    lineup_slot = None
    stats = None
    for side in ("home", "away"):
        side_data = box.get(side, {}) or {}
        players = side_data.get("players", {}) or {}
        entry = players.get(f"ID{batter_id}")
        if entry is None:
            continue
        player_stats = (entry.get("stats") or {}).get("batting") or {}
        batting_order = side_data.get("battingOrder") or []
        if batter_id in batting_order:
            lineup_slot = batting_order.index(batter_id) + 1
        elif str(batter_id) in [str(x) for x in batting_order]:
            lineup_slot = [str(x) for x in batting_order].index(str(batter_id)) + 1
        stats = player_stats
        break

    if stats is None:
        # Not in either roster or no stats — treat as didn't play
        return _empty_outcome(lineup_slot)

    # Counts
    hits      = int(stats.get("hits", 0) or 0)
    hrs       = int(stats.get("homeRuns", 0) or 0)
    doubles   = int(stats.get("doubles", 0) or 0)
    triples   = int(stats.get("triples", 0) or 0)
    singles   = max(0, hits - doubles - triples - hrs)
    walks     = int(stats.get("baseOnBalls", 0) or 0)
    int_walks = int(stats.get("intentionalWalks", 0) or 0)
    strikeouts = int(stats.get("strikeOuts", 0) or 0)
    hbp       = int(stats.get("hitByPitch", 0) or 0)
    pa        = int(stats.get("plateAppearances", 0) or 0)
    total_bases = singles + 2 * doubles + 3 * triples + 4 * hrs

    pa_sequence = _extract_pa_sequence(game_pk, batter_id)

    return {
        "pa_count": pa,
        "hits": hits,
        "hrs": hrs,
        "total_bases": total_bases,
        "singles": singles,
        "doubles": doubles,
        "triples": triples,
        "strikeouts": strikeouts,
        # Treat intentional walks as walks for prop-betting parity with
        # the sim (which lumps BB / intent_walk under "BB").
        "walks": walks + int_walks,
        "hbp": hbp,
        "lineup_slot": lineup_slot,
        "pa_sequence": pa_sequence,
        "played": pa > 0 or hits > 0 or walks > 0 or strikeouts > 0,
    }


def _extract_pa_sequence(game_pk: int, batter_id: int) -> list[dict[str, Any]]:
    """Per-PA detail for this batter from the game feed. Returns [] if
    the feed is unavailable or the batter didn't hit in the game."""
    try:
        import statsapi  # type: ignore[import-untyped]
        feed = statsapi.get("game", {"gamePk": game_pk})
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "pa_sequence_feed_failed game_pk=%s batter_id=%s: %r",
            game_pk, batter_id, exc,
        )
        return []

    all_plays = ((feed.get("liveData") or {}).get("plays") or {}).get("allPlays") or []
    out: list[dict[str, Any]] = []
    for play in all_plays:
        matchup = play.get("matchup") or {}
        batter = matchup.get("batter") or {}
        if batter.get("id") != batter_id:
            continue
        pitcher = matchup.get("pitcher") or {}
        result = play.get("result") or {}
        event_type = result.get("eventType") or ""
        about = play.get("about") or {}
        out.append({
            "pitcher_id": pitcher.get("id"),
            "outcome": _RESULT_TO_OUTCOME.get(event_type, "OUT"),
            "inning": about.get("inning"),
            "raw_event": event_type,
        })
    return out
