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

from hit_ledger.config import BVP_DEFAULT_ENABLED, HOT_LIST_SIZE, LEAGUE_AVG_XBA
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


def render_sidebar() -> tuple[date, bool, bool, bool, bool, int | None, bool]:
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
             "10000 for final-run quality.",
    )

    show_detail = st.sidebar.checkbox(
        "Show pitch-mix detail",
        value=False,
        help="Expand each batter row with the per-pitch log-5 breakdown "
             "(batter vs pitcher vs blended on xBA, contact, and HR/contact), "
             "PA sequence with TTO labels, and — in pitch-by-pitch mode — a "
             "sample pitch sequence for that batter.",
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

    return selected_date, run_clicked, force_refresh, enable_bvp, use_pbp, n_sims, show_detail


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


def render_hot_list(
    predictions: pd.DataFrame,
    lineups: pd.DataFrame,
    games: pd.DataFrame,
    bvp_annotations: dict,
):
    if predictions.empty:
        st.info("No predictions yet. Click **Run Engine** in the sidebar.")
        return

    df = predictions.merge(
        lineups[["batter_id", "batter_name", "team", "lineup_slot", "bats"]],
        on="batter_id", how="left",
    )
    df = df.merge(
        games[["game_pk", "home_team", "away_team", "venue"]],
        on="game_pk", how="left",
    )

    st.markdown("### Top 5 Picks")

    # Create tabs for different categories
    tab_hit, tab_tb, tab_hr = st.tabs(["1+ Hit", "2+ TB", "HR"])

    def build_table(sorted_df, bvp_annotations):
        table_data = []
        for _, row in sorted_df.iterrows():
            batter_id = int(row["batter_id"])
            raw_name = row.get("batter_name") or f"Batter #{batter_id}"
            bats = row.get("bats") or ""
            hand_label = ""
            if bats:
                if bats.upper() == "L":
                    hand_label = "LHB"
                elif bats.upper() == "R":
                    hand_label = "RHB"
                elif bats.upper() == "S":
                    hand_label = "SW"

            if hand_label:
                display_name = f"{raw_name} ({hand_label})"
            else:
                display_name = raw_name

            hit_val = row['p_1_hit'] * 100 if pd.notna(row.get('p_1_hit')) else 0
            tb_val = row['p_tb_over_1_5'] * 100 if pd.notna(row.get('p_tb_over_1_5')) else 0
            hr_val = row['p_1_hr'] * 100 if pd.notna(row.get('p_1_hr')) else 0

            # Get BvP annotation
            bvp = bvp_annotations.get(batter_id, "—")

            table_data.append({
                "Name": display_name,
                "1H%": hit_val,
                "2TB%": tb_val,
                "HR%": hr_val,
                "BvP": bvp if bvp else "—",
            })
        return pd.DataFrame(table_data)

    column_config = {
        "1H%": st.column_config.NumberColumn(format="%.0f%%"),
        "2TB%": st.column_config.NumberColumn(format="%.0f%%"),
        "HR%": st.column_config.NumberColumn(format="%.0f%%"),
    }

    with tab_hit:
        top_hit = df.sort_values("p_1_hit", ascending=False).head(HOT_LIST_SIZE)
        st.dataframe(
            build_table(top_hit, bvp_annotations),
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
        )

    with tab_tb:
        top_tb = df.sort_values("p_tb_over_1_5", ascending=False).head(HOT_LIST_SIZE)
        st.dataframe(
            build_table(top_tb, bvp_annotations),
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
        )

    with tab_hr:
        top_hr = df.sort_values("p_1_hr", ascending=False).head(HOT_LIST_SIZE)
        st.dataframe(
            build_table(top_hr, bvp_annotations),
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
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
    show_detail: bool = False,
    pbp_sample_traces: dict | None = None,
):
    if games.empty or predictions.empty:
        return
    st.markdown("### Matchup Breakdowns")

    # Percentile-based grades computed across the full slate
    slate_grades = compute_slate_grades(predictions, matchup_details)

    pbp_sample_traces = pbp_sample_traces or {}

    # Predictions indexed by batter_id for fast lookup in the detail cards
    pred_by_bid: dict[int, pd.Series] = {}
    if not predictions.empty:
        for _, row in predictions.iterrows():
            if pd.notna(row.get("batter_id")):
                pred_by_bid[int(row["batter_id"])] = row

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

                    # Rich per-batter breakdown cards — hidden by default,
                    # revealed when "Show pitch-mix detail" is on in the
                    # sidebar. We render them INSIDE the team column so the
                    # detail stays visually scoped to its team.
                    if show_detail:
                        for _, row in team_rows.iterrows():
                            batter_id = int(row["batter_id"])
                            matchup = matchup_details.get(batter_id)
                            if matchup is None:
                                continue
                            raw_name = row.get("batter_name") or f"#{batter_id}"
                            bats = row.get("bats") or ""
                            hand_label = ""
                            if bats:
                                upper = bats.upper()
                                if upper == "L":
                                    hand_label = "LHB"
                                elif upper == "R":
                                    hand_label = "RHB"
                                elif upper == "S":
                                    hand_label = "SW"
                            display_name = (
                                f"{raw_name} ({hand_label})" if hand_label else raw_name
                            )
                            slot = int(row["lineup_slot"]) if pd.notna(row["lineup_slot"]) else 0
                            _render_batter_detail(
                                display_name=display_name,
                                slot=slot,
                                matchup=matchup,
                                pred_row=pred_by_bid.get(batter_id),
                                sample_trace=pbp_sample_traces.get(batter_id),
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


_SOURCE_LABEL = {
    "starter_tto_1": "SP TTO1",
    "starter_tto_2": "SP TTO2",
    "starter_tto_3": "SP TTO3",
    "bullpen": "BP",
}

_QUALITY_COLOR = {
    "strong": "#5a8a6a",
    "good": "#d4a24c",
    "limited": "#a07040",
    "no_data": "#5a5a5a",
}


def _render_batter_detail(
    display_name: str,
    slot: int,
    matchup,
    pred_row: pd.Series | None,
    sample_trace: list | None,
):
    """One compact rich card per batter: PA sequence with TTO labels, the
    full log-5 pitch-mix breakdown (batter / pitcher / blended on xBA,
    contact, HR-per-contact), and a sample pitch sequence when pbp mode
    ran this batter."""
    # Header line — slot, name, overall data_quality, top-line prob summary
    p_hit = pred_row.get("p_1_hit") if pred_row is not None else None
    p_hr = pred_row.get("p_1_hr") if pred_row is not None else None
    p_2h = pred_row.get("p_2_hits") if pred_row is not None else None
    hit_s = f"{p_hit * 100:.0f}%" if p_hit is not None and pd.notna(p_hit) else "—"
    hr_s = f"{p_hr * 100:.0f}%" if p_hr is not None and pd.notna(p_hr) else "—"
    two_s = f"{p_2h * 100:.0f}%" if p_2h is not None and pd.notna(p_2h) else "—"

    quality = getattr(matchup, "data_quality", None) or "good"
    q_color = _QUALITY_COLOR.get(quality, "#8a8679")

    header = (
        "<div style='display:flex;justify-content:space-between;align-items:center;"
        "padding:0.5rem 0 0.25rem 0;border-bottom:1px solid #2d2d2b;margin-top:0.75rem'>"
        f"<div>"
        f"<span style='font-family:JetBrains Mono,monospace;color:#8a8679'>{slot}.</span> "
        f"<span style='font-family:Fraunces,serif;font-weight:600;color:#e8e4d8'>{display_name}</span> "
        f"<span style='font-family:JetBrains Mono,monospace;font-size:0.6rem;"
        f"color:{q_color};text-transform:uppercase;letter-spacing:0.1em;margin-left:0.5rem'>"
        f"[{quality}]</span>"
        "</div>"
        "<div style='font-family:JetBrains Mono,monospace;font-size:0.75rem;color:#8a8679'>"
        f"<span style='color:#d4a24c'>1H {hit_s}</span> · "
        f"<span>2H {two_s}</span> · "
        f"<span>HR {hr_s}</span>"
        "</div></div>"
    )
    st.markdown(header, unsafe_allow_html=True)

    # PA sequence
    pa_rows = []
    for i, pa in enumerate(matchup.pa_probs, 1):
        src_label = _SOURCE_LABEL.get(pa.source, pa.source)
        color = "#5a8a6a" if "bullpen" in pa.source else "#d4a24c"
        pa_rows.append(
            "<div style='font-family:JetBrains Mono,monospace;font-size:0.68rem;"
            "color:#8a8679;padding-left:1rem'>"
            f"PA {i} · <span style='color:{color}'>{src_label}</span> · "
            f"<span style='color:#e8e4d8'>P(hit)={pa.p_hit:.3f}</span> · "
            f"P(HR)={pa.p_hr:.3f}"
            "</div>"
        )

    ctx_bits = [
        f"{matchup.expected_pa_vs_starter:.1f} PA vs SP",
        f"{matchup.expected_pa_vs_bullpen:.1f} PA vs BP",
    ]
    if matchup.bullpen_xba is not None:
        ctx_bits[-1] += f" (xBA {matchup.bullpen_xba:.3f})"
    if matchup.umpire_adjustment:
        # Reverse-convert the xBA adj back to K% deviation for readability.
        from hit_ledger.config import UMPIRE_K_XBA_SENSITIVITY
        k_dev_pp = (-matchup.umpire_adjustment / UMPIRE_K_XBA_SENSITIVITY)
        sign = "+" if k_dev_pp >= 0 else ""
        ctx_bits.append(f"Ump {sign}{k_dev_pp:.1f}pp K")
    ctx_line = (
        "<div style='font-family:JetBrains Mono,monospace;font-size:0.6rem;"
        "color:#8a8679;padding-left:1rem;margin-top:2px;text-transform:uppercase;"
        "letter-spacing:0.1em'>"
        f"{' · '.join(ctx_bits)}"
        "</div>"
    )

    # Pitch-mix rows with batter / pitcher / blended columns.
    # Uses a monospace layout so aligned columns read like a table.
    mix_rows = [
        "<div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;"
        "color:#6f6e68;padding-left:1rem;margin-top:6px;"
        "text-transform:uppercase;letter-spacing:0.08em'>"
        "Pitch-mix log-5 (batter · pitcher · blended)"
        "</div>"
    ]
    # Header row
    mix_rows.append(
        "<div style='font-family:JetBrains Mono,monospace;font-size:0.66rem;"
        "color:#8a8679;padding-left:1rem'>"
        "<code>"
        "Pt  Mix    xBA  B/P/Bl             Ct  B/P/Bl             "
        "HR/C B/P/Bl          Edge   n(B/P)"
        "</code></div>"
    )
    for b in matchup.starter_breakdown:
        edge = b.get("edge", 0.0)
        edge_cls = "matchup-edge-pos" if edge > 0 else "matchup-edge-neg"
        row = (
            "<div style='font-family:JetBrains Mono,monospace;font-size:0.66rem;"
            "color:#8a8679;padding-left:1rem'>"
            "<code>"
            f"{b['pitch_type']:3s}"
            f"{b['share'] * 100:4.0f}%  "
            f"{b.get('batter_xba', 0):.3f}/{b.get('pitcher_xba', 0):.3f}/"
            f"<span style='color:#e8e4d8'>{b.get('blended_xba', 0):.3f}</span>  "
            f"{b.get('batter_contact', 0):.2f}/{b.get('pitcher_contact', 0):.2f}/"
            f"<span style='color:#e8e4d8'>{b.get('blended_contact', 0):.2f}</span>  "
            f"{b.get('batter_hr_per_contact', 0):.3f}/"
            f"{b.get('pitcher_hr_per_contact', 0):.3f}/"
            f"<span style='color:#e8e4d8'>{b.get('blended_hr_per_contact', 0):.3f}</span>  "
            f"<span class='{edge_cls}'>{edge:+.3f}</span>  "
            f"{b.get('sample_pitches', 0)}/{b.get('pitcher_sample_pitches', 0)}"
            "</code></div>"
        )
        mix_rows.append(row)

    st.markdown("".join(pa_rows), unsafe_allow_html=True)
    st.markdown(ctx_line, unsafe_allow_html=True)
    st.markdown("".join(mix_rows), unsafe_allow_html=True)

    # Optional sample PA trace (pitch-by-pitch mode only)
    if sample_trace:
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;"
            "color:#6f6e68;padding-left:1rem;margin-top:8px;"
            "text-transform:uppercase;letter-spacing:0.08em'>"
            "Sample pitch-by-pitch PA trace"
            "</div>",
            unsafe_allow_html=True,
        )
        trace_rows = []
        for pa_i, pa in enumerate(sample_trace, 1):
            src = _SOURCE_LABEL.get(pa.get("source", ""), pa.get("source", ""))
            pa_header = (
                "<div style='font-family:JetBrains Mono,monospace;font-size:0.68rem;"
                "color:#e8e4d8;padding-left:1rem;margin-top:2px'>"
                f"PA {pa_i} <span style='color:#8a8679'>({src})</span> → "
                f"<span style='color:#d4a24c'>{pa['outcome']}</span> · "
                f"count {pa['final_count']} · {pa['n_pitches']} pitches"
                "</div>"
            )
            trace_rows.append(pa_header)
            for i, p in enumerate(pa["pitch_sequence"], 1):
                if p["swung"]:
                    if p["contact"]:
                        act = ("foul" if p["foul"]
                               else f"BIP ev={p['ev']:.0f} la={p['la']:.0f}")
                    else:
                        act = "whiff"
                else:
                    act = "take"
                zone = "Z" if p["in_zone"] else "O"
                trace_rows.append(
                    "<div style='font-family:JetBrains Mono,monospace;font-size:0.64rem;"
                    "color:#8a8679;padding-left:2rem'>"
                    f"{i}. {p['pitch_type']:3s}[{zone}]  {act}"
                    "</div>"
                )
        st.markdown("".join(trace_rows), unsafe_allow_html=True)


def main():
    (
        selected_date, run_clicked, force_refresh, enable_bvp,
        use_pbp, n_sims, show_detail,
    ) = render_sidebar()

    predictions = cache.load_predictions(selected_date)
    games = cache.load_games(selected_date)
    lineups_df = cache.load_lineups(selected_date)
    matchup_details: dict = {}
    bvp_annotations: dict = {}
    umpires: dict = {}
    pitcher_stats: dict = {}
    pbp_sample_traces: dict = {}
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
        pbp_sample_traces = getattr(result, 'pbp_sample_traces', {}) or {}
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

    # Inferred engine mode from cached data when a run hasn't fired yet.
    if engine_mode is None and pbp_sample_traces:
        engine_mode = "pbp"

    render_header(selected_date, engine_mode=engine_mode)

    # Today's Slate at the top
    render_games_table(games, umpires)

    st.markdown("---")

    # Top 5 picks table
    render_hot_list(predictions, lineups_df, games, bvp_annotations)

    st.markdown("---")

    # Matchup breakdowns (rich per-batter cards enabled via sidebar toggle)
    render_matchup_expanders(
        games, lineups_df, predictions, matchup_details,
        umpires, bvp_annotations, pitcher_stats,
        show_detail=show_detail,
        pbp_sample_traces=pbp_sample_traces,
    )


if __name__ == "__main__":
    main()
