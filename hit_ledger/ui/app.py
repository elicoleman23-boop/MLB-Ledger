"""
The Hit Ledger — v2 Streamlit UI.

Adds vs v1:
    - Per-PA breakdown in matchup expanders (PA 1 vs starter TTO1, PA 4 vs bullpen, etc.)
    - Umpire panel per game
    - Bullpen quality indicator
    - Starter workload indicator (IP/start)
    - BvP toggle in sidebar (off by default)
    - Opposing pitcher stats display
    - Matchup rating with letter grades

Run from project root:
    streamlit run app.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from hit_ledger.config import BVP_DEFAULT_ENABLED, LEAGUE_AVG_XBA
from hit_ledger.data import cache
from hit_ledger.sim.pipeline_v2 import run_daily_pipeline_v2
from hit_ledger.ui.styles import CSS
from hit_ledger.utils.teams import TEAM_FULL_TO_SHORT


def compute_matchup_score(
    p_hit: float,
    matchup_details: dict,
    batter_id: int,
) -> float | None:
    """
    Compute a raw 0-100 matchup score. Returns None when p_hit is missing.

    Edge-based when starter breakdown is available, falling back to raw p_hit.
    The denominator controls how tightly scores are scaled around league avg;
    a tighter denominator spreads edges more aggressively.
    """
    if p_hit is None or (isinstance(p_hit, float) and np.isnan(p_hit)):
        return None

    matchup = matchup_details.get(batter_id)

    if not matchup or not getattr(matchup, "starter_breakdown", None):
        score = (p_hit - 0.30) / 0.25 * 100
        return max(0.0, min(100.0, score))

    total_weighted_edge = 0.0
    total_share = 0.0
    for b in matchup.starter_breakdown:
        share = b.get("share", 0)
        edge = b.get("edge", 0)
        total_weighted_edge += edge * share
        total_share += share

    avg_edge = total_weighted_edge / total_share if total_share > 0 else 0.0
    score = 50 + (avg_edge / 0.015) * 40
    return max(0.0, min(100.0, score))


def _grade_from_percentile(pct: float) -> str:
    """Map percentile rank (0..1, higher = better) to a letter grade."""
    if pct >= 0.95:
        return "A+"
    if pct >= 0.80:
        return "A"
    if pct >= 0.60:
        return "A-"
    if pct >= 0.40:
        return "B+"
    if pct >= 0.20:
        return "B"
    if pct >= 0.10:
        return "B-"
    if pct >= 0.05:
        return "C+"
    return "C"


def compute_slate_grades(
    predictions: pd.DataFrame,
    matchup_details: dict,
) -> dict[int, str]:
    """
    Compute a grade for every batter in the slate using percentile-based buckets
    over the whole slate's score distribution.
    """
    if predictions.empty:
        return {}

    scores: dict[int, float] = {}
    for _, row in predictions.iterrows():
        bid = int(row["batter_id"])
        p_hit = row.get("p_1_hit")
        s = compute_matchup_score(p_hit, matchup_details, bid)
        if s is not None:
            scores[bid] = s

    if not scores:
        return {}

    score_series = pd.Series(scores)
    # rank(pct=True) gives percentile rank in 0..1 where 1 is the top
    ranks = score_series.rank(method="average", pct=True)
    return {int(bid): _grade_from_percentile(float(r)) for bid, r in ranks.items()}


st.set_page_config(
    page_title="The Hit Ledger",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _cached_pipeline(
    game_date_iso: str,
    force_refresh: bool = False,
    enable_bvp: bool = False,
    use_pbp: bool = False,
    n_sims: int | None = None,
):
    game_date = date.fromisoformat(game_date_iso)
    progress_ph = st.sidebar.empty()
    progress_bar = st.sidebar.progress(0.0)

    stage_weights = {
        "schedule": 0.03,
        "batters": 0.35,
        "pitchers": 0.15,
        "workload": 0.10,
        "tto": 0.08,
        "bullpens": 0.15,
        "umpires": 0.05,
        "matchups": 0.04,
        "simulate": 0.05,
    }
    stage_order = list(stage_weights.keys())
    completed = {s: 0.0 for s in stage_order}

    def progress_cb(stage: str, frac: float):
        completed[stage] = stage_weights.get(stage, 0) * frac
        idx = stage_order.index(stage) if stage in stage_order else len(stage_order)
        total = sum(stage_weights[s] for s in stage_order[:idx]) + completed[stage]
        progress_bar.progress(min(total, 1.0))
        progress_ph.markdown(
            f"<div style='font-family:JetBrains Mono,monospace;"
            f"font-size:0.75rem;color:#8a8679;text-transform:uppercase;"
            f"letter-spacing:0.15em'>{stage} · {int(frac * 100)}%</div>",
            unsafe_allow_html=True,
        )

    result = run_daily_pipeline_v2(
        game_date,
        progress=progress_cb,
        force_refresh=force_refresh,
        enable_bvp=enable_bvp,
        use_pbp=use_pbp,
        n_sims=n_sims,
    )
    progress_bar.empty()
    progress_ph.empty()
    return result


def render_sidebar() -> tuple[date, bool, bool, bool, bool, int | None]:
    st.sidebar.markdown(
        "<div style='font-family:Fraunces,serif;font-size:1.75rem;"
        "font-weight:800;letter-spacing:-0.02em;margin-bottom:0'>The Hit Ledger</div>"
        "<div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;"
        "text-transform:uppercase;letter-spacing:0.2em;color:#8a8679;"
        "margin-bottom:1.5rem'>v2 · TTO · Bullpen · Umpire</div>",
        unsafe_allow_html=True,
    )

    selected_date = st.sidebar.date_input(
        "Slate Date", value=date.today(), max_value=date.today()
    )
    run_clicked = st.sidebar.button("Run Engine", use_container_width=True)

    use_pbp = st.sidebar.checkbox(
        "Realistic sim (pitch-by-pitch)",
        value=False,
        help="Swap the fast PA-level engine for the pitch-by-pitch sim. "
             "Much slower (~20-50× per PA) but walks every pitch with count-aware "
             "logic. Default sims drop to 1000 for tractable runtime; increase to "
             "10000 for final-run quality. Also unlocks the full category set on "
             "the Leaderboards tab.",
    )

    # BvP is always enabled now
    enable_bvp = True

    with st.sidebar.expander("Advanced"):
        force_refresh = st.checkbox(
            "Force refresh (ignore cache)", value=False,
            help="Re-scrape all Statcast data. Slow.",
        )
        if use_pbp:
            n_sims = st.number_input(
                "PBP sims per batter",
                min_value=100, max_value=20_000, value=1_000, step=100,
                help="Higher = smoother probabilities but slower. 1000 is the "
                     "speed/quality sweet spot for interactive use.",
            )
        else:
            n_sims = None
        if st.button("Clear prediction cache for this date"):
            with cache._connect() as conn:  # noqa: SLF001
                conn.execute(
                    "DELETE FROM predictions WHERE game_date = ?",
                    (selected_date.isoformat(),),
                )
            st.cache_data.clear()
            st.success("Cleared.")

    sims_line = f"Sims per batter: {n_sims or 1_000}" if use_pbp else "Sims per batter: 10,000"
    mode_line = "Mode: pitch-by-pitch" if use_pbp else "Mode: fast (PA-level)"
    st.sidebar.markdown(
        "<div style='margin-top:2rem;padding-top:1rem;border-top:1px solid #2d2d2b;"
        "font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#8a8679'>"
        f"{mode_line}<br>"
        "Model: xBA × pitch-mix<br>"
        "Regression: k=200<br>"
        "Recency blend: 70/30<br>"
        "TTO: pitcher-specific + fallback<br>"
        "Bullpen: team xBA by handedness<br>"
        f"{sims_line}"
        "</div>",
        unsafe_allow_html=True,
    )

    return selected_date, run_clicked, force_refresh, enable_bvp, use_pbp, n_sims


def render_header(selected_date: date, engine_mode: str | None = None):
    badge = ""
    if engine_mode:
        badge_bg = "#5a8a6a" if engine_mode == "pbp" else "#3a3a38"
        badge_label = "pitch-by-pitch" if engine_mode == "pbp" else "fast (PA-level)"
        badge = (
            f"<span style='display:inline-block;margin-left:0.75rem;"
            f"padding:2px 8px;border-radius:10px;background:{badge_bg};"
            f"font-family:JetBrains Mono,monospace;font-size:0.55rem;"
            f"color:#0f0f0e;text-transform:uppercase;letter-spacing:0.15em;"
            f"vertical-align:middle'>{badge_label}</span>"
        )
    st.markdown(
        f"<div class='ledger-title'>The Hit Ledger{badge}</div>"
        f"<div class='ledger-subtitle'>"
        f"Monte Carlo · TTO-aware · {selected_date.strftime('%A, %B %d, %Y')}"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_games_table(games: pd.DataFrame, umpires: dict):
    if games.empty:
        return
    st.markdown("### Today's Slate")
    display = games.copy()
    display["Matchup"] = display["away_team"] + " @ " + display["home_team"]
    display["Park"] = display["venue"]
    # Convert UTC to EST (UTC-4 during EDT, UTC-5 during EST)
    game_times = pd.to_datetime(display["game_time"], errors="coerce", utc=True)
    # Convert to US/Eastern timezone
    display["First Pitch"] = game_times.dt.tz_convert("US/Eastern").dt.strftime("%I:%M %p ET")
    display["HP Umpire"] = display["game_pk"].map(
        lambda gpk: (umpires.get(gpk, {}) or {}).get("umpire_name") or "—"
    )
    st.dataframe(
        display[["Matchup", "Park", "First Pitch", "HP Umpire"]].reset_index(drop=True),
        use_container_width=True, hide_index=True,
    )


def _format_pitcher_stats(pitcher_id: int, pitcher_stats: dict, is_home: bool) -> str:
    """Format opposing pitcher stats for display above batting team."""
    stats = pitcher_stats.get(pitcher_id, {})
    if not stats:
        return ""

    parts = []
    throws = stats.get("throws", "R")
    parts.append(f"<span style='color:#d4a24c;font-weight:600'>{throws}HP</span>")

    era = stats.get("era")
    if era is not None:
        parts.append(f"ERA {era:.2f}")

    xba = stats.get("xba")
    if xba is not None:
        parts.append(f"xBA .{int(xba * 1000):03d}")

    hr9 = stats.get("hr_per_9")
    if hr9 is not None:
        parts.append(f"HR/9 {hr9:.2f}")

    k_pct = stats.get("k_pct")
    if k_pct is not None:
        parts.append(f"K% {k_pct * 100:.1f}")

    whip = stats.get("whip")
    if whip is not None:
        parts.append(f"WHIP {whip:.2f}")

    if len(parts) <= 1:
        return ""

    return (
        f"<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
        f"color:#8a8679;margin-bottom:0.5rem;padding:4px 8px;"
        f"background:#1a1a18;border-radius:4px'>"
        f"<span style='color:#e8e4d8;font-weight:600'>vs Starter:</span> "
        f"{' · '.join(parts)}</div>"
    )


def render_matchup_expanders(
    games: pd.DataFrame,
    lineups: pd.DataFrame,
    predictions: pd.DataFrame,
    matchup_details: dict,
    umpires: dict,
    bvp_annotations: dict,
    pitcher_stats: dict,
):
    if games.empty or predictions.empty:
        return
    st.markdown("### Matchup Breakdowns")

    # Percentile-based grades computed across the full slate
    slate_grades = compute_slate_grades(predictions, matchup_details)

    # Always show BvP column
    show_bvp = True

    for _, g in games.iterrows():
        game_pk = g["game_pk"]
        title = f"{g['away_team']} @ {g['home_team']} — {g.get('venue') or ''}"
        with st.expander(title):
            ump = umpires.get(game_pk, {}) or {}
            st.markdown(_format_umpire_line(ump), unsafe_allow_html=True)

            game_lineups = lineups[lineups["game_pk"] == game_pk]
            if game_lineups.empty:
                st.markdown("_Lineups not yet posted._")
                continue

            game_preds = predictions[predictions["game_pk"] == game_pk]
            merged = game_lineups.merge(game_preds, on="batter_id", how="left")

            col_away, col_home = st.columns(2)
            for col, team_name, is_home in [(col_away, g["away_team"], False), (col_home, g["home_team"], True)]:
                with col:
                    st.markdown(f"**{team_name}**")

                    # Get opposing pitcher ID for this team
                    if is_home:
                        opp_pitcher_id = g.get("away_pitcher_id")
                    else:
                        opp_pitcher_id = g.get("home_pitcher_id")

                    # Display opposing pitcher stats
                    if opp_pitcher_id and pd.notna(opp_pitcher_id):
                        pitcher_line = _format_pitcher_stats(int(opp_pitcher_id), pitcher_stats, is_home)
                        if pitcher_line:
                            st.markdown(pitcher_line, unsafe_allow_html=True)

                    # Convert full team name to short name for comparison with lineups
                    team_short = TEAM_FULL_TO_SHORT.get(team_name, team_name)
                    team_rows = merged[merged["team"] == team_short].sort_values("lineup_slot")
                    if team_rows.empty:
                        st.markdown("_Lineup not posted._")
                        continue

                    # Build table data for this team
                    table_data = []
                    for _, row in team_rows.iterrows():
                        batter_id = int(row["batter_id"])
                        raw_name = row.get("batter_name") or f"#{batter_id}"
                        bats = row.get("bats") or ""
                        # Format name with handedness below it
                        hand_label = ""
                        if bats:
                            if bats.upper() == "L":
                                hand_label = "LHB"
                            elif bats.upper() == "R":
                                hand_label = "RHB"
                            elif bats.upper() == "S":
                                hand_label = "SW"

                        slot = int(row["lineup_slot"]) if pd.notna(row["lineup_slot"]) else 0

                        # Store raw values for sorting (as numeric)
                        p_hit_raw = row.get('p_1_hit')
                        p_hr_raw = row.get('p_1_hr')
                        p_tb_raw = row.get('p_tb_over_1_5')

                        hit_val = p_hit_raw * 100 if pd.notna(p_hit_raw) else 0
                        hr_val = p_hr_raw * 100 if pd.notna(p_hr_raw) else 0
                        tb_val = p_tb_raw * 100 if pd.notna(p_tb_raw) else 0

                        # Percentile-based grade (precomputed across slate)
                        grade = slate_grades.get(batter_id, "—")

                        # Format name with handedness
                        if hand_label:
                            display_name = f"{raw_name} ({hand_label})"
                        else:
                            display_name = raw_name

                        row_data = {
                            "#": slot,
                            "Player": display_name,
                            "Grade": grade,
                            "1H%": hit_val,
                            "2TB%": tb_val,
                            "HR%": hr_val,
                        }

                        # Add BvP if enabled and available
                        if show_bvp:
                            bvp_text = bvp_annotations.get(batter_id, "—")
                            row_data["BvP"] = bvp_text if bvp_text else "—"

                        table_data.append(row_data)

                    team_df = pd.DataFrame(table_data)

                    column_config = {
                        "1H%": st.column_config.NumberColumn(format="%.0f%%"),
                        "2TB%": st.column_config.NumberColumn(format="%.0f%%"),
                        "HR%": st.column_config.NumberColumn(format="%.0f%%"),
                    }

                    st.dataframe(
                        team_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config=column_config,
                    )


def _format_umpire_line(ump: dict) -> str:
    if not ump or not ump.get("umpire_name"):
        return (
            "<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
            "color:#8a8679;margin-bottom:0.5rem'>HP Umpire: unavailable</div>"
        )
    name = ump["umpire_name"]
    dev = ump.get("k_pct_dev")
    if dev is None:
        dev_str = "no tendency data"
    else:
        sign = "+" if dev >= 0 else ""
        dev_str = f"K% dev: {sign}{dev * 100:.1f}pp"
    return (
        f"<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
        f"color:#8a8679;margin-bottom:0.5rem'>HP Umpire: {name} · {dev_str}</div>"
    )


# ---------------------------------------------------------------------------
# Leaderboards tab
# ---------------------------------------------------------------------------
# Category definitions. `key` is the dict key on batter_outcomes /
# pitcher_outcomes (both produced by the pbp engine). `kind` drives
# formatting: "pct" renders as percentage, "mean" as float.
# `min_engine` controls availability: "fast" means also computable from
# BatterSimResultV2, "pbp" means only the pitch-by-pitch engine produces
# the number.

BATTER_CATEGORIES = [
    # label,                   outcome key,        kind,  min_engine
    ("1+ Hit",                 "p_1_hit",          "pct",  "fast"),
    ("2+ Hits",                "p_2_hits",         "pct",  "fast"),
    ("3+ Hits",                "p_3_hits",         "pct",  "pbp"),
    ("1+ Single",              "p_1_single",       "pct",  "pbp"),
    ("2+ Singles",             "p_2_singles",      "pct",  "pbp"),
    ("1+ Double",              "p_1_double",       "pct",  "pbp"),
    ("1+ Triple",              "p_1_triple",       "pct",  "pbp"),
    ("1+ HR",                  "p_1_hr",           "pct",  "fast"),
    ("2+ HR",                  "p_2_hr",           "pct",  "pbp"),
    ("1+ Walk",                "p_1_walk",         "pct",  "pbp"),
    ("2+ Walks",               "p_2_walks",        "pct",  "pbp"),
    ("1+ Strikeout",           "p_1_k",            "pct",  "pbp"),
    ("2+ Strikeouts",          "p_2_k",            "pct",  "pbp"),
    ("1+ HBP",                 "p_1_hbp",          "pct",  "pbp"),
    ("2+ Total Bases",         "p_tb_2",           "pct",  "pbp"),
    ("3+ Total Bases",         "p_tb_3",           "pct",  "pbp"),
    ("4+ Total Bases",         "p_tb_4",           "pct",  "pbp"),
    ("Expected Hits",          "expected_hits",    "mean", "fast"),
    ("Expected Total Bases",   "expected_tb",      "mean", "fast"),
]

PITCHER_CATEGORIES = [
    # label,              outcome key,     kind,  min_engine
    ("Expected Ks (SP)",  "expected_k",    "mean", "pbp"),
]

LEADERBOARD_TOP_N = 10


def _fast_mode_batter_outcomes_from_preds(
    predictions: pd.DataFrame,
) -> dict[int, dict[str, float]]:
    """In fast mode the engine doesn't produce per-threshold outcome dicts;
    build a compatible dict from the BatterSimResultV2 columns instead so
    the leaderboard code has one uniform input shape."""
    if predictions.empty:
        return {}
    out: dict[int, dict[str, float]] = {}
    for _, row in predictions.iterrows():
        if pd.isna(row.get("batter_id")):
            continue
        bid = int(row["batter_id"])
        out[bid] = {
            "p_1_hit":       float(row["p_1_hit"])        if pd.notna(row.get("p_1_hit")) else 0.0,
            "p_2_hits":      float(row["p_2_hits"])       if pd.notna(row.get("p_2_hits")) else 0.0,
            "p_1_hr":        float(row["p_1_hr"])         if pd.notna(row.get("p_1_hr")) else 0.0,
            "p_tb_2":        float(row["p_tb_over_1_5"])  if pd.notna(row.get("p_tb_over_1_5")) else 0.0,
            "p_tb_3":        float(row["p_tb_over_2_5"])  if pd.notna(row.get("p_tb_over_2_5")) else 0.0,
            "expected_hits": float(row["expected_hits"])  if pd.notna(row.get("expected_hits")) else 0.0,
            "expected_tb":   float(row["expected_tb"])    if pd.notna(row.get("expected_tb")) else 0.0,
        }
    return out


def _batter_name_map(lineups: pd.DataFrame) -> dict[int, tuple[str, str]]:
    """batter_id → (name, team_short) for leaderboard row labels."""
    out: dict[int, tuple[str, str]] = {}
    if lineups.empty:
        return out
    for _, row in lineups.iterrows():
        if pd.isna(row.get("batter_id")):
            continue
        bid = int(row["batter_id"])
        name = row.get("batter_name") or f"Batter #{bid}"
        team = row.get("team") or ""
        out[bid] = (name, team)
    return out


def _pitcher_name_map(
    games: pd.DataFrame,
    pitcher_stats: dict,
) -> dict[int, tuple[str, str]]:
    """pitcher_id → (name_or_id, team_short). pitcher_stats may carry a
    'name' / 'full_name' / 'player_name' field depending on source; fall
    back to "#<id>" when nothing surfaces."""
    out: dict[int, tuple[str, str]] = {}
    if games.empty:
        return out
    for _, g in games.iterrows():
        for pid_col, team_col in (
            ("home_pitcher_id", "home_team"),
            ("away_pitcher_id", "away_team"),
        ):
            pid = g.get(pid_col)
            if pid is None or pd.isna(pid):
                continue
            pid = int(pid)
            stats = pitcher_stats.get(pid, {}) or {}
            name = (
                stats.get("name")
                or stats.get("full_name")
                or stats.get("player_name")
                or f"Pitcher #{pid}"
            )
            team = g.get(team_col) or ""
            out[pid] = (name, team)
    return out


def _format_value(value: float, kind: str) -> str:
    if kind == "pct":
        return f"{value * 100:.1f}%"
    return f"{value:.2f}"


def _render_category_leaderboard(
    label: str,
    rows: list[tuple[str, str, float]],
    kind: str,
):
    """rows = [(name, team, value)], already truncated to TOP_N and sorted."""
    with st.expander(label, expanded=False):
        if not rows:
            st.markdown(
                "<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
                "color:#8a8679'>No data available for this slate.</div>",
                unsafe_allow_html=True,
            )
            return
        table = pd.DataFrame([
            {
                "#": i + 1,
                "Player": name,
                "Team": team,
                "Value": _format_value(val, kind),
            }
            for i, (name, team, val) in enumerate(rows)
        ])
        st.dataframe(table, use_container_width=True, hide_index=True)


def render_leaderboards(
    engine_mode: str,
    batter_outcomes: dict[int, dict[str, float]],
    pitcher_outcomes: dict[int, dict[str, float]],
    lineups: pd.DataFrame,
    games: pd.DataFrame,
    pitcher_stats: dict,
):
    """Stacked expanders, one per category. Batter leaderboards first,
    then pitcher categories. Categories that require pbp are hidden in
    fast mode instead of showing an empty list."""
    st.markdown(
        "<div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
        "color:#8a8679;margin-bottom:0.5rem'>"
        f"Top {LEADERBOARD_TOP_N} per category. Expand any row to see the "
        f"ranked players and their simulated percentages or expected counts."
        "</div>",
        unsafe_allow_html=True,
    )

    if not batter_outcomes and not pitcher_outcomes:
        st.info(
            "Run the engine to populate the leaderboards. Enable "
            "**Realistic sim (pitch-by-pitch)** in the sidebar to unlock "
            "the full category set."
        )
        return

    batter_names = _batter_name_map(lineups)
    pitcher_names = _pitcher_name_map(games, pitcher_stats)

    # Batter section
    st.markdown("#### Batters")
    for label, key, kind, min_engine in BATTER_CATEGORIES:
        if min_engine == "pbp" and engine_mode != "pbp":
            continue
        rows: list[tuple[str, str, float]] = []
        for bid, outcomes in batter_outcomes.items():
            if key not in outcomes:
                continue
            name, team = batter_names.get(bid, (f"Batter #{bid}", ""))
            rows.append((name, team, outcomes[key]))
        rows.sort(key=lambda r: r[2], reverse=True)
        _render_category_leaderboard(label, rows[:LEADERBOARD_TOP_N], kind)

    # Pitcher section — only shown when there's at least one pbp category
    # available (currently only E[K] and only in pbp mode).
    pitcher_visible = [
        (label, key, kind, min_engine)
        for (label, key, kind, min_engine) in PITCHER_CATEGORIES
        if not (min_engine == "pbp" and engine_mode != "pbp")
    ]
    if pitcher_visible and pitcher_outcomes:
        st.markdown("#### Pitchers (starters)")
        for label, key, kind, _ in pitcher_visible:
            rows: list[tuple[str, str, float]] = []
            for pid, outcomes in pitcher_outcomes.items():
                if key not in outcomes:
                    continue
                name, team = pitcher_names.get(pid, (f"Pitcher #{pid}", ""))
                rows.append((name, team, outcomes[key]))
            rows.sort(key=lambda r: r[2], reverse=True)
            _render_category_leaderboard(label, rows[:LEADERBOARD_TOP_N], kind)


def main():
    (
        selected_date, run_clicked, force_refresh, enable_bvp,
        use_pbp, n_sims,
    ) = render_sidebar()

    predictions = cache.load_predictions(selected_date)
    games = cache.load_games(selected_date)
    lineups_df = cache.load_lineups(selected_date)
    matchup_details: dict = {}
    bvp_annotations: dict = {}
    umpires: dict = {}
    pitcher_stats: dict = {}
    batter_outcomes: dict = {}
    pitcher_outcomes: dict = {}
    engine_mode: str | None = None

    if run_clicked or force_refresh:
        if force_refresh:
            st.cache_data.clear()
        result = _cached_pipeline(
            selected_date.isoformat(),
            force_refresh=force_refresh,
            enable_bvp=enable_bvp,
            use_pbp=use_pbp,
            n_sims=n_sims,
        )
        predictions = result.predictions
        games = result.games
        lineups_df = result.lineups
        matchup_details = result.matchup_details
        bvp_annotations = result.bvp_annotations
        umpires = result.umpires
        pitcher_stats = getattr(result, 'pitcher_stats', {})
        batter_outcomes = getattr(result, 'batter_outcomes', {}) or {}
        pitcher_outcomes = getattr(result, 'pitcher_outcomes', {}) or {}
        engine_mode = result.summary.get("mode", "fast")
        st.sidebar.success(
            f"Ran {result.summary.get('n_matchups', 0)} matchups across "
            f"{result.summary.get('n_games', 0)} games ({engine_mode} engine)"
        )

        if not predictions.empty and "p_1_hit" in predictions.columns:
            p1h = predictions["p_1_hit"].dropna()
            if not p1h.empty:
                st.sidebar.markdown(
                    "<div style='margin-top:0.5rem;font-family:JetBrains Mono,monospace;"
                    "font-size:0.7rem;color:#8a8679'>"
                    f"P(1+ hit) · min {p1h.min() * 100:.1f}% · "
                    f"max {p1h.max() * 100:.1f}% · "
                    f"σ {p1h.std() * 100:.1f}pp"
                    "</div>",
                    unsafe_allow_html=True,
                )

    if engine_mode is None:
        engine_mode = "fast"

    # Fast-mode leaderboard feed is derived from predictions — the fast
    # engine doesn't produce the wide outcome dict natively.
    if engine_mode == "fast" and not batter_outcomes:
        batter_outcomes = _fast_mode_batter_outcomes_from_preds(predictions)

    render_header(selected_date, engine_mode=engine_mode)

    tab_matchups, tab_leaderboards = st.tabs(["Matchups", "Leaderboards"])

    with tab_matchups:
        render_games_table(games, umpires)
        st.markdown("---")
        render_matchup_expanders(
            games, lineups_df, predictions, matchup_details,
            umpires, bvp_annotations, pitcher_stats,
        )

    with tab_leaderboards:
        render_leaderboards(
            engine_mode=engine_mode,
            batter_outcomes=batter_outcomes,
            pitcher_outcomes=pitcher_outcomes,
            lineups=lineups_df,
            games=games,
            pitcher_stats=pitcher_stats,
        )


if __name__ == "__main__":
    main()
