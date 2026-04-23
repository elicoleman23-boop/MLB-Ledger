"""
v2 matchup probability construction — per-PA probabilities.

Every PA now has its own xBA and HR rate because:
    - PAs 1, 2, 3 are vs the starter at different TTO levels
    - Later PAs (typically 4+) are vs the bullpen
    - Umpire K% adjustment applies to all PAs in the game uniformly

The simulation engine consumes a list of per-PA probabilities per batter
instead of a single p_hit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from hit_ledger.config import (
    HIT_TYPE_DIST,
    LEAGUE_AVG_CONTACT_RATE,
    LEAGUE_AVG_HR_PER_BBE,
    LEAGUE_AVG_HR_PER_CONTACT,
    LEAGUE_AVG_K_RATE,
    LEAGUE_AVG_XBA,
    LEAGUE_HR_PER_CONTACT_BY_PITCH,
    LEAGUE_WHIFF_BY_PITCH,
    LEAGUE_XBA_BY_PITCH,
    MIN_PITCHES_PER_SPLIT,
    PARK_FACTORS_HITS,
    PARK_FACTORS_HR,
    RECENT_FORM_DAYS,
    RECENT_WEIGHT,
    REGRESSION_K,
    REGRESSION_K_CONTACT,
    REGRESSION_K_HR,
    REGRESSION_K_XBA,
    SEASON_WEIGHT,
    UMPIRE_K_XBA_SENSITIVITY,
)

# Minimum PA-ending events to trust batter-specific rates
MIN_PA_FOR_BATTER_CONTACT = 50
MIN_PA_FOR_BATTER_HR = 100
LEAGUE_HR_PER_PA = LEAGUE_AVG_HR_PER_BBE * 0.35
LEAGUE_HR_PER_9 = 1.17
DATA_QUALITY_SCORE = {"strong": 3, "good": 2, "limited": 1, "no_data": 0}
DATA_QUALITY_FROM_SCORE = {3: "strong", 2: "good", 1: "limited", 0: "no_data"}
from hit_ledger.data.pitcher_workload import (
    expected_pa_vs_starter,
    tto_penalty_to_apply,
)


# ---------------------------------------------------------------------------
# Output container (v2)
# ---------------------------------------------------------------------------
@dataclass
class PAProbability:
    """Outcome probabilities for a single plate appearance."""
    p_1b: float
    p_2b: float
    p_3b: float
    p_hr: float
    source: str  # 'starter_tto_1', 'starter_tto_2', 'starter_tto_3', 'bullpen'
    data_quality: str = "good"  # 'strong' | 'good' | 'limited' | 'no_data'

    @property
    def p_hit(self) -> float:
        return self.p_1b + self.p_2b + self.p_3b + self.p_hr


@dataclass
class MatchupV2:
    """Full v2 matchup: per-PA probability sequence for one batter-game."""
    batter_id: int
    starter_id: int
    pa_probs: list[PAProbability]       # length = total expected PAs
    expected_pa_vs_starter: float
    expected_pa_vs_bullpen: float
    # Diagnostic breakdown for UI
    starter_breakdown: list[dict] = field(default_factory=list)
    tto_penalties: dict = field(default_factory=dict)
    umpire_adjustment: float = 0.0
    bullpen_xba: float | None = None
    data_quality: str = "good"

    @property
    def weighted_p_hit(self) -> float:
        """Simple weighted avg P(hit) across all PAs, for display."""
        if not self.pa_probs:
            return 0.0
        return float(np.mean([pa.p_hit for pa in self.pa_probs]))


# ---------------------------------------------------------------------------
# Helpers (revamped for realistic probability calculation)
# ---------------------------------------------------------------------------
def _xba_and_contact_for_split(
    batter_df: pd.DataFrame,
    pitch_type: str,
    p_throws: str,
    since: date | None = None,
) -> tuple[float, float, int]:
    """
    Compute both contact quality AND contact rate for a (pitch_type, p_throws) split.

    Returns:
        (contact_quality, contact_rate, n_pitches)

    contact_rate  = (PAs with ball in play) / (total PAs)
    contact_quality = xwOBAcon-flavored skill metric: expected BA on contact,
                      derived from Statcast's estimated_ba_using_speedangle.
                      Measures how well the batter squared it up — a stable
                      skill signal, separate from BABIP luck (which is
                      injected in the sampling engine).
    """
    league_xba = LEAGUE_XBA_BY_PITCH.get(pitch_type, LEAGUE_AVG_XBA)
    league_whiff = LEAGUE_WHIFF_BY_PITCH.get(pitch_type, 0.25)
    # League contact rate ≈ 1 - (K_rate based on whiff)
    # Rough approximation: higher whiff → more Ks
    league_contact = max(0.50, 1.0 - league_whiff * 0.7)

    if batter_df.empty:
        return league_xba, league_contact, 0

    mask = (batter_df["pitch_type"] == pitch_type) & (batter_df["p_throws"] == p_throws)
    if since is not None:
        mask &= batter_df["game_date"] >= pd.Timestamp(since)
    pitches = batter_df[mask]
    n = len(pitches)

    if n == 0:
        return league_xba, league_contact, 0

    pa_ending = pitches[pitches["events"].notna() & (pitches["events"] != "")]
    if pa_ending.empty:
        return league_xba, league_contact, n

    # Calculate contact rate: PAs that weren't strikeouts / total PAs
    strikeout_events = {"strikeout", "strikeout_double_play"}
    walk_events = {"walk", "hit_by_pitch", "intent_walk"}

    n_pa = len(pa_ending)
    n_strikeouts = int(pa_ending["events"].isin(strikeout_events).sum())
    n_walks = int(pa_ending["events"].isin(walk_events).sum())
    n_contact = n_pa - n_strikeouts - n_walks

    # Contact rate (excluding walks from denominator for pure contact measure)
    contact_denom = n_pa - n_walks
    if contact_denom > 0:
        contact_rate = n_contact / contact_denom
    else:
        contact_rate = league_contact

    # Contact quality (expected BA on balls in play only)
    ball_in_play = pa_ending[~pa_ending["events"].isin(strikeout_events | walk_events)]
    if ball_in_play.empty:
        contact_quality = league_xba
    else:
        xba_series = ball_in_play["estimated_ba_using_speedangle"].copy()
        hit_events = {"single", "double", "triple", "home_run"}
        for idx in xba_series[xba_series.isna()].index:
            ev = ball_in_play.at[idx, "events"]
            xba_series.at[idx] = 1.0 if ev in hit_events else 0.0
        contact_quality = float(xba_series.mean())

    return contact_quality, contact_rate, n


def _xba_for_split(
    batter_df: pd.DataFrame,
    pitch_type: str,
    p_throws: str,
    since: date | None = None,
) -> tuple[float, int]:
    """Raw xBA for a (pitch_type, p_throws) split. Legacy wrapper."""
    xba, _, n = _xba_and_contact_for_split(batter_df, pitch_type, p_throws, since)
    return xba, n


def _regress(raw: float, n: int, prior: float, k: int = REGRESSION_K) -> float:
    return (n * raw + k * prior) / (n + k)


def _batter_hr_rate(batter_df: pd.DataFrame) -> float:
    if batter_df.empty:
        return LEAGUE_HR_PER_PA
    pa_ending = batter_df[batter_df["events"].notna() & (batter_df["events"] != "")]
    n_pa = len(pa_ending)
    if n_pa < MIN_PA_FOR_BATTER_HR:
        return LEAGUE_HR_PER_PA
    n_hr = int((pa_ending["events"] == "home_run").sum())
    raw = n_hr / n_pa
    return _regress(raw, n_pa, LEAGUE_HR_PER_PA, k=REGRESSION_K_HR)


def _pitcher_rates_for_pitch(
    pitcher_df: pd.DataFrame,
    pitch_type: str,
    batter_stands: str,
) -> tuple[float, float, int]:
    """
    Returns (pitcher_contact_quality, pitcher_contact_rate, n_pitches) for
    this pitcher's specific pitch type vs batters of `batter_stands`.

    Mirrors _xba_and_contact_for_split but computed from the pitcher's POV.
    Filters pitcher_df on (pitch_type, stand == batter_stands).
    """
    league_xba = LEAGUE_XBA_BY_PITCH.get(pitch_type, LEAGUE_AVG_XBA)
    league_whiff = LEAGUE_WHIFF_BY_PITCH.get(pitch_type, 0.25)
    league_contact = max(0.50, 1.0 - league_whiff * 0.7)

    if pitcher_df is None or pitcher_df.empty:
        return league_xba, league_contact, 0

    mask = pitcher_df["pitch_type"] == pitch_type
    if batter_stands in ("L", "R") and "stand" in pitcher_df.columns:
        mask &= pitcher_df["stand"] == batter_stands
    pitches = pitcher_df[mask]
    n = len(pitches)
    if n == 0:
        return league_xba, league_contact, 0

    pa_ending = pitches[pitches["events"].notna() & (pitches["events"] != "")]
    if pa_ending.empty:
        return league_xba, league_contact, n

    strikeout_events = {"strikeout", "strikeout_double_play"}
    walk_events = {"walk", "hit_by_pitch", "intent_walk"}

    n_pa = len(pa_ending)
    n_strikeouts = int(pa_ending["events"].isin(strikeout_events).sum())
    n_walks = int(pa_ending["events"].isin(walk_events).sum())
    n_contact = n_pa - n_strikeouts - n_walks

    contact_denom = n_pa - n_walks
    contact_rate = n_contact / contact_denom if contact_denom > 0 else league_contact

    ball_in_play = pa_ending[~pa_ending["events"].isin(strikeout_events | walk_events)]
    if ball_in_play.empty:
        contact_quality = league_xba
    else:
        xba_series = ball_in_play["estimated_ba_using_speedangle"].copy()
        hit_events = {"single", "double", "triple", "home_run"}
        for idx in xba_series[xba_series.isna()].index:
            ev = ball_in_play.at[idx, "events"]
            xba_series.at[idx] = 1.0 if ev in hit_events else 0.0
        contact_quality = float(xba_series.mean())

    return contact_quality, contact_rate, n


_NON_CONTACT_EVENTS = {
    "strikeout", "strikeout_double_play",
    "walk", "hit_by_pitch", "intent_walk",
}


def _batter_hr_per_contact_for_pitch(
    batter_df: pd.DataFrame,
    pitch_type: str,
    p_throws: str,
) -> tuple[float, int]:
    """
    Returns (hr_per_contact, n_contact_events) for this batter's contact
    against this pitch type from the given pitcher handedness.

    Contact events = PA-ending rows whose event is not a K, walk, HBP, or
    intentional walk. hr_per_contact = HR / contact_events.
    When there are no contact events, returns (0.0, 0) — the caller regresses
    toward a league prior which handles the empty case.
    """
    if batter_df.empty:
        return 0.0, 0
    mask = (batter_df["pitch_type"] == pitch_type) & (batter_df["p_throws"] == p_throws)
    rows = batter_df[mask]
    pa_ending = rows[rows["events"].notna() & (rows["events"] != "")]
    if pa_ending.empty:
        return 0.0, 0
    contact = pa_ending[~pa_ending["events"].isin(_NON_CONTACT_EVENTS)]
    n_contact = len(contact)
    if n_contact == 0:
        return 0.0, 0
    n_hr = int((contact["events"] == "home_run").sum())
    return n_hr / n_contact, n_contact


def _pitcher_hr_per_contact_for_pitch(
    pitcher_df: pd.DataFrame,
    pitch_type: str,
    batter_stands: str,
) -> tuple[float, int]:
    """HR allowed per contact event for this pitcher's pitch_type vs
    batters of `batter_stands` handedness. Mirrors the batter helper."""
    if pitcher_df is None or pitcher_df.empty:
        return 0.0, 0
    mask = pitcher_df["pitch_type"] == pitch_type
    if batter_stands in ("L", "R") and "stand" in pitcher_df.columns:
        mask &= pitcher_df["stand"] == batter_stands
    rows = pitcher_df[mask]
    pa_ending = rows[rows["events"].notna() & (rows["events"] != "")]
    if pa_ending.empty:
        return 0.0, 0
    contact = pa_ending[~pa_ending["events"].isin(_NON_CONTACT_EVENTS)]
    n_contact = len(contact)
    if n_contact == 0:
        return 0.0, 0
    n_hr = int((contact["events"] == "home_run").sum())
    return n_hr / n_contact, n_contact


def _log5_blend(p_batter: float, p_pitcher: float, p_league: float) -> float:
    """
    Bill James odds-ratio blend of a batter rate and pitcher rate against a
    league baseline. Returns the combined matchup probability.

        p_combined = (p_b · p_p / p_lg) /
                     (p_b · p_p / p_lg + (1 - p_b)·(1 - p_p)/(1 - p_lg))

    Clips all inputs to [0.001, 0.999] so the division and the 1 - p terms
    don't blow up on pathological inputs.
    """
    p_league = max(0.001, min(0.999, p_league))
    p_batter = max(0.001, min(0.999, p_batter))
    p_pitcher = max(0.001, min(0.999, p_pitcher))
    num = (p_batter * p_pitcher) / p_league
    denom = num + ((1 - p_batter) * (1 - p_pitcher)) / (1 - p_league)
    return num / denom


def _batter_overall_contact_rate(batter_df: pd.DataFrame) -> tuple[float, int]:
    """Batter's overall contact rate across all pitches, with n_pa for sufficiency checks."""
    if batter_df.empty:
        return LEAGUE_AVG_CONTACT_RATE, 0
    pa_ending = batter_df[batter_df["events"].notna() & (batter_df["events"] != "")]
    if pa_ending.empty:
        return LEAGUE_AVG_CONTACT_RATE, 0
    strikeout_events = {"strikeout", "strikeout_double_play"}
    walk_events = {"walk", "hit_by_pitch", "intent_walk"}
    n_pa = len(pa_ending)
    n_strikeouts = int(pa_ending["events"].isin(strikeout_events).sum())
    n_walks = int(pa_ending["events"].isin(walk_events).sum())
    n_contact = n_pa - n_strikeouts - n_walks
    denom = n_pa - n_walks
    if denom <= 0:
        return LEAGUE_AVG_CONTACT_RATE, n_pa
    raw = n_contact / denom
    return _regress(raw, n_pa, LEAGUE_AVG_CONTACT_RATE, k=REGRESSION_K_CONTACT), n_pa


def _compute_starter_matchup(
    batter_df: pd.DataFrame,
    pitcher_throws: str,
    pitcher_arsenal: dict[str, float],
    as_of: date,
    pitcher_df: pd.DataFrame | None = None,
    batter_stands: str = "R",
) -> tuple[float, float, float, list[dict], str]:
    """
    Compute weighted xBA on contact AND contact rate across pitch types,
    blending batter and pitcher per-pitch-type rates via log-5.

    BATTER SPLITS ARE ONE SIDE OF THE EDGE:
    - Batter's performance vs (pitch_type, p_throws) sets their half.
    - Pitcher's performance vs (pitch_type, batter_stands) sets the other.
    - Log-5 combines them against the league baseline for that pitch type.

    P(hit per PA) = P(contact) × P(hit | contact)
                  = contact_rate × contact_quality

    IMPORTANT: contact rate and xBA on contact covary across pitch types
    (fastballs tend to be high-contact AND high-xBA; sliders are low-contact
    AND low-xBA). So E[contact] · E[xBA] ≠ E[contact · xBA]. The weighted
    composite p_hit, `weighted_p_hit_on_contact = Σ (share_i · contact_i · xba_i)`,
    is the mathematically correct per-PA hit probability and should be
    preferred over `weighted_contact × weighted_xba` downstream.

    If `pitcher_df` is empty or the pitcher has too few samples for a pitch
    type (< MIN_PITCHES_PER_SPLIT), the pitcher side falls back fully to the
    league prior — which makes the log-5 blend degrade to the batter rate
    alone (preserving pre-Fix-B behavior when pitcher data is unavailable).

    Per-pitch HR rate is also modeled: for each pitch type we regress the
    batter's and pitcher's HR-per-contact toward LEAGUE_HR_PER_CONTACT_BY_PITCH,
    blend via log-5, and accumulate `weighted_hr_per_contact`. The caller
    multiplies that by contact rate to get HR/PA.

    Returns (weighted_xba, weighted_contact, weighted_p_hit_on_contact,
             weighted_hr_per_contact, breakdown, overall_data_quality).
    """
    if not pitcher_arsenal:
        pitcher_arsenal = {"FF": 0.55, "SL": 0.25, "CH": 0.20}

    # Compute batter's overall contact rate once — used as per-pitch-type regression
    # prior when sufficient. Falls back to league (0.765) when insufficient.
    batter_overall_contact, batter_overall_n_pa = _batter_overall_contact_rate(batter_df)
    has_batter_overall = batter_overall_n_pa >= MIN_PA_FOR_BATTER_CONTACT

    recent_cutoff = as_of - timedelta(days=RECENT_FORM_DAYS)
    breakdown = []
    weighted_xba = 0.0
    weighted_contact = 0.0
    weighted_p_hit_on_contact = 0.0
    weighted_hr_per_contact = 0.0
    total_share = 0.0

    for pitch_type, share in pitcher_arsenal.items():
        xba_prior = LEAGUE_XBA_BY_PITCH.get(pitch_type, LEAGUE_AVG_XBA)
        whiff_rate = LEAGUE_WHIFF_BY_PITCH.get(pitch_type, 0.25)
        league_contact_for_pitch = max(0.50, 1.0 - whiff_rate * 0.7)
        contact_prior = batter_overall_contact if has_batter_overall else LEAGUE_AVG_CONTACT_RATE

        # Get BATTER'S ACTUAL performance vs this pitch type
        xba_season, contact_season, n_season = _xba_and_contact_for_split(
            batter_df, pitch_type, pitcher_throws
        )
        xba_recent, contact_recent, n_recent = _xba_and_contact_for_split(
            batter_df, pitch_type, pitcher_throws, since=recent_cutoff
        )

        # How much do we trust the batter's actual data?
        # With REGRESSION_K_XBA = 120:
        #   n=30  → 20% batter, 80% league (not enough data)
        #   n=120 → 50% batter, 50% league (decent)
        #   n=240 → 67% batter, 33% league (good)
        #   n=480 → 80% batter, 20% league (strong)

        if n_season < MIN_PITCHES_PER_SPLIT:
            # Not enough data - use league average
            adjusted_xba = xba_prior
            adjusted_contact = contact_prior
            data_quality = "no_data"
        else:
            # USE BATTER'S ACTUAL SPLITS (with light regression for stability)
            season_xba_adj = _regress(xba_season, n_season, xba_prior, k=REGRESSION_K_XBA)
            recent_xba_adj = (
                _regress(xba_recent, n_recent, xba_prior, k=REGRESSION_K_XBA) if n_recent >= 5 else season_xba_adj
            )
            adjusted_xba = SEASON_WEIGHT * season_xba_adj + RECENT_WEIGHT * recent_xba_adj

            season_contact_adj = _regress(contact_season, n_season, contact_prior, k=REGRESSION_K_CONTACT)
            recent_contact_adj = (
                _regress(contact_recent, n_recent, contact_prior, k=REGRESSION_K_CONTACT) if n_recent >= 5 else season_contact_adj
            )
            adjusted_contact = SEASON_WEIGHT * season_contact_adj + RECENT_WEIGHT * recent_contact_adj

            # Determine data quality for UI
            batter_weight = n_season / (n_season + REGRESSION_K)
            if batter_weight >= 0.7:
                data_quality = "strong"
            elif batter_weight >= 0.5:
                data_quality = "good"
            else:
                data_quality = "limited"

        # PITCHER SIDE — rates vs this pitch type and batter handedness.
        # Regress toward the league prior; below MIN sample, fall back fully.
        pitcher_xba_raw, pitcher_contact_raw, n_pitcher = _pitcher_rates_for_pitch(
            pitcher_df, pitch_type, batter_stands
        )
        if n_pitcher < MIN_PITCHES_PER_SPLIT:
            pitcher_adjusted_xba = xba_prior
            pitcher_adjusted_contact = league_contact_for_pitch
        else:
            pitcher_adjusted_xba = _regress(
                pitcher_xba_raw, n_pitcher, xba_prior, k=REGRESSION_K_XBA
            )
            pitcher_adjusted_contact = _regress(
                pitcher_contact_raw, n_pitcher, league_contact_for_pitch, k=REGRESSION_K_CONTACT
            )

        # LOG-5 BLEND vs league baseline for this pitch type. When either
        # side equals the league, log-5 collapses to the other side.
        blended_xba = _log5_blend(adjusted_xba, pitcher_adjusted_xba, xba_prior)
        blended_contact = _log5_blend(
            adjusted_contact, pitcher_adjusted_contact, league_contact_for_pitch
        )

        # HR per contact — per-pitch-type modeling. Regress both sides toward
        # the pitch-specific league prior with HR's heavier K, then log-5 blend.
        hr_per_contact_prior = LEAGUE_HR_PER_CONTACT_BY_PITCH.get(
            pitch_type, LEAGUE_AVG_HR_PER_CONTACT
        )
        batter_hr_raw, batter_hr_n = _batter_hr_per_contact_for_pitch(
            batter_df, pitch_type, pitcher_throws
        )
        pitcher_hr_raw, pitcher_hr_n = _pitcher_hr_per_contact_for_pitch(
            pitcher_df, pitch_type, batter_stands
        )
        batter_hr_adj = _regress(
            batter_hr_raw, batter_hr_n, hr_per_contact_prior, k=REGRESSION_K_HR
        )
        pitcher_hr_adj = _regress(
            pitcher_hr_raw, pitcher_hr_n, hr_per_contact_prior, k=REGRESSION_K_HR
        )
        blended_hr_per_contact = _log5_blend(
            batter_hr_adj, pitcher_hr_adj, hr_per_contact_prior
        )

        weighted_xba += share * blended_xba
        weighted_contact += share * blended_contact
        # Mathematically correct composite: weight per-pitch p_hit, don't
        # multiply two independently-weighted averages (they covary).
        weighted_p_hit_on_contact += share * blended_contact * blended_xba
        weighted_hr_per_contact += share * blended_hr_per_contact
        total_share += share

        # Pitch-level matchup hit probability uses the BLENDED rates.
        pitch_hit_prob = blended_contact * blended_xba
        league_hit_prob = league_contact_for_pitch * xba_prior

        # Edge = full matchup vs league avg. Reflects batter strength AND
        # pitcher weakness together.
        raw_edge = pitch_hit_prob - league_hit_prob

        breakdown.append({
            "pitch_type": pitch_type,
            "share": share,
            "batter_xba": adjusted_xba,
            "batter_contact": adjusted_contact,
            "pitcher_xba": pitcher_adjusted_xba,
            "pitcher_contact": pitcher_adjusted_contact,
            "blended_xba": blended_xba,
            "blended_contact": blended_contact,
            "batter_hr_per_contact": batter_hr_adj,
            "pitcher_hr_per_contact": pitcher_hr_adj,
            "blended_hr_per_contact": blended_hr_per_contact,
            "league_hr_per_contact": hr_per_contact_prior,
            "hit_prob": pitch_hit_prob,
            "league_xba": xba_prior,
            "league_contact": league_contact_for_pitch,
            "league_hit_prob": league_hit_prob,
            "edge": raw_edge,
            "weighted_edge": raw_edge * share,  # Impact on total matchup
            "sample_pitches": n_season,
            "pitcher_sample_pitches": n_pitcher,
            "data_quality": data_quality,
        })

    if total_share > 0:
        weighted_xba /= total_share
        weighted_contact /= total_share
        weighted_p_hit_on_contact /= total_share
        weighted_hr_per_contact /= total_share

    if breakdown:
        q_sum = sum(DATA_QUALITY_SCORE[b["data_quality"]] * b["share"] for b in breakdown)
        share_sum = sum(b["share"] for b in breakdown)
        if share_sum > 0:
            overall_quality = DATA_QUALITY_FROM_SCORE[round(q_sum / share_sum)]
        else:
            overall_quality = "no_data"
    else:
        overall_quality = "no_data"

    return (
        weighted_xba,
        weighted_contact,
        weighted_p_hit_on_contact,
        weighted_hr_per_contact,
        breakdown,
        overall_quality,
    )


def _compute_starter_xba(
    batter_df: pd.DataFrame,
    pitcher_throws: str,
    pitcher_arsenal: dict[str, float],
    as_of: date,
) -> tuple[float, list[dict]]:
    """Legacy wrapper - returns xBA only for backward compatibility."""
    xba, _, _, _, breakdown, _ = _compute_starter_matchup(
        batter_df, pitcher_throws, pitcher_arsenal, as_of
    )
    return xba, breakdown


def _split_xba_into_components(
    p_hit: float, p_hr: float
) -> tuple[float, float, float]:
    """Distribute non-HR hit probability across 1B/2B/3B using league shares."""
    non_hr_hit = max(0.0, p_hit - p_hr)
    denom = 1 - HIT_TYPE_DIST["HR"]
    p_1b = non_hr_hit * HIT_TYPE_DIST["1B"] / denom
    p_2b = non_hr_hit * HIT_TYPE_DIST["2B"] / denom
    p_3b = non_hr_hit * HIT_TYPE_DIST["3B"] / denom
    return p_1b, p_2b, p_3b


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_matchup_v2(
    batter_id: int,
    batter_df: pd.DataFrame,
    starter_id: int,
    starter_throws: str,
    starter_arsenal: dict[str, float],
    starter_workload: dict,         # {'avg_ip_per_start', 'starts_sampled', 'season_xba'}
    tto_splits: dict,               # {'xba': {1,2,3: float}, 'pa': {1,2,3: int}}
    bullpen_profile: dict,          # {'xba_vs_r', 'xba_vs_l', 'pa_vs_r', 'pa_vs_l'}
    batter_stands: str,             # 'L', 'R', or 'S'
    lineup_slot: int,
    total_pa: float,                # total expected PAs (from PA_BY_LINEUP_SLOT)
    venue: str | None = None,
    umpire_k_dev: float | None = None,
    as_of: date | None = None,
    pitcher_hr9: float | None = None,  # DEPRECATED: pitcher HR tendency is
                                       # now modeled via _pitcher_hr_per_contact_for_pitch.
                                       # Kept for API stability; unused.
    pitcher_df: pd.DataFrame | None = None,
) -> MatchupV2:
    """
    Construct the per-PA probability sequence for one batter vs one starter+bullpen.

    CRITICAL: Hit probability per PA is calculated as:
        P(hit) = P(contact) × P(hit | contact)
               = contact_rate × contact_quality

    This is much lower than raw xBA (~0.16-0.20 per PA, not 0.24-0.28).
    (BABIP luck — variance between expected and actual outcomes on balls
    in play — is injected separately in the sampling engine, so downstream
    P(hit) here is purely the contact-quality skill signal.)

    Algorithm:
        1. Compute starter xBA AND contact rate via pitch-mix weighting
        2. P(hit per PA) = contact_rate × xba (realistic, not inflated)
        3. Compute expected PAs vs starter based on IP/start + lineup slot
        4. For each TTO, apply the TTO penalty
        5. Remaining PAs are bullpen PAs
        6. Apply park factor + umpire K% adjustment
    """
    as_of = as_of or date.today()

    # Step 1: starter matchup (xBA, contact rate, composite p_hit, and per-pitch-type
    # HR-per-contact — all pitch-mix weighted)
    (
        starter_xba_base,
        starter_contact_base,
        starter_p_hit_on_contact_base,
        starter_hr_per_contact_base,
        starter_breakdown,
        overall_data_quality,
    ) = _compute_starter_matchup(
        batter_df, starter_throws, starter_arsenal, as_of,
        pitcher_df=pitcher_df, batter_stands=batter_stands,
    )

    # Step 2: expected PAs vs starter
    avg_ip = starter_workload.get("avg_ip_per_start", 5.3)
    exp_starter_pa = expected_pa_vs_starter(avg_ip, lineup_slot)
    exp_starter_pa = min(exp_starter_pa, total_pa)  # can't exceed total
    exp_bullpen_pa = max(0.0, total_pa - exp_starter_pa)

    # Step 3+4: distribute starter PAs across TTO and apply penalties
    season_xba = starter_workload.get("season_xba", LEAGUE_AVG_XBA)
    tto_penalties = {}
    for tto in [1, 2, 3]:
        tto_penalties[tto] = tto_penalty_to_apply(tto_splits, season_xba, tto)

    # Park factors from real data
    park_mult_hits = PARK_FACTORS_HITS.get(venue or "", PARK_FACTORS_HITS["_default"])
    park_mult_hr = PARK_FACTORS_HR.get(venue or "", PARK_FACTORS_HR["_default"])

    # Umpire adjustment
    ump_dev = umpire_k_dev if umpire_k_dev is not None else 0.0
    ump_xba_adj = -ump_dev * 100 * UMPIRE_K_XBA_SENSITIVITY

    # HR rate per PA: derived from the pitch-mix-weighted HR-per-contact
    # (batter & pitcher blended via log-5 per pitch type) times the
    # pitch-mix-weighted contact rate. The flat pitcher_hr9 multiplier is
    # no longer applied — pitcher HR tendency is already baked into
    # _pitcher_hr_per_contact_for_pitch; multiplying again would double-count.
    # pitcher_hr9 is kept in the signature (deprecated) for API stability.
    _ = pitcher_hr9  # deprecated; intentionally unused
    p_hr_raw = starter_hr_per_contact_base * starter_contact_base

    # Step 5+6: build the per-PA sequence
    pa_probs: list[PAProbability] = []
    bullpen_xba_used = None

    # Starter PAs - use batter's actual contact rate from their splits
    n_starter_pa_int = int(np.floor(exp_starter_pa))

    for pa_idx in range(n_starter_pa_int + 1):  # +1 for possible fractional PA
        tto_num = min(pa_idx + 1, 3)
        penalty = tto_penalties[tto_num]

        # TTO penalty and umpire adjustment are expressed in xBA-space.
        # Convert to p_hit-space by multiplying by contact rate, then compose
        # with the pitch-mix-weighted p_hit and apply the park multiplier.
        penalty_adj = penalty * starter_contact_base
        ump_adj = ump_xba_adj * starter_contact_base

        p_hit = starter_p_hit_on_contact_base + penalty_adj + ump_adj
        p_hit = p_hit * park_mult_hits
        p_hit = float(np.clip(p_hit, 0.0, 0.42))

        p_hr = float(np.clip(p_hr_raw * park_mult_hr, 0.0, 0.08))

        p_1b, p_2b, p_3b = _split_xba_into_components(p_hit, p_hr)
        pa_probs.append(PAProbability(
            p_1b=p_1b, p_2b=p_2b, p_3b=p_3b, p_hr=p_hr,
            source=f"starter_tto_{tto_num}",
            data_quality=overall_data_quality,
        ))

    # Bullpen PAs
    # Use bullpen's actual xBA vs this batter's handedness (from Statcast data)
    if batter_stands == "S":
        # Switch hitters bat opposite-handed vs the pitcher, so average both sides
        bullpen_xba_base = (
            bullpen_profile.get("xba_vs_l", LEAGUE_AVG_XBA)
            + bullpen_profile.get("xba_vs_r", LEAGUE_AVG_XBA)
        ) / 2
    else:
        bullpen_xba_key = "xba_vs_l" if batter_stands == "L" else "xba_vs_r"
        bullpen_xba_base = bullpen_profile.get(bullpen_xba_key, LEAGUE_AVG_XBA)

    bullpen_xba_used = bullpen_xba_base

    # Compute batter's overall contact rate (not vs-starter-specific) for bullpen PAs
    if batter_df.empty:
        bullpen_contact = LEAGUE_AVG_CONTACT_RATE
    else:
        pa_ending = batter_df[batter_df["events"].notna() & (batter_df["events"] != "")]
        if pa_ending.empty:
            bullpen_contact = LEAGUE_AVG_CONTACT_RATE
        else:
            strikeouts = pa_ending["events"].isin({"strikeout", "strikeout_double_play"}).sum()
            walks = pa_ending["events"].isin({"walk", "hit_by_pitch", "intent_walk"}).sum()
            n_pa = len(pa_ending)
            n_contact = n_pa - strikeouts - walks
            denom = n_pa - walks
            raw = n_contact / denom if denom > 0 else LEAGUE_AVG_CONTACT_RATE
            bullpen_contact = _regress(raw, n_pa, LEAGUE_AVG_CONTACT_RATE)

    n_bullpen_pa_int = int(np.ceil(exp_bullpen_pa))
    for _ in range(n_bullpen_pa_int):
        contact = float(np.clip(bullpen_contact, 0.40, 0.95))
        xba = (bullpen_xba_base + ump_xba_adj) * park_mult_hits

        # Bullpen xBA here is a team-level aggregate (already a composite rate,
        # not pitch-type decomposed), so the E[X]·E[Y] ≠ E[X·Y] correction
        # doesn't apply — contact × xba is fine here. Fix pending for when
        # bullpen data gets pitch-type splits.
        p_hit = contact * xba
        p_hit = float(np.clip(p_hit, 0.0, 0.42))

        p_hr = float(np.clip(p_hr_raw * park_mult_hr, 0.0, 0.08))
        p_1b, p_2b, p_3b = _split_xba_into_components(p_hit, p_hr)
        pa_probs.append(PAProbability(
            p_1b=p_1b, p_2b=p_2b, p_3b=p_3b, p_hr=p_hr,
            source="bullpen",
            data_quality=overall_data_quality,
        ))

    return MatchupV2(
        batter_id=batter_id,
        starter_id=starter_id,
        pa_probs=pa_probs,
        expected_pa_vs_starter=exp_starter_pa,
        expected_pa_vs_bullpen=exp_bullpen_pa,
        starter_breakdown=starter_breakdown,
        tto_penalties=tto_penalties,
        umpire_adjustment=ump_xba_adj,
        bullpen_xba=bullpen_xba_used,
        data_quality=overall_data_quality,
    )
