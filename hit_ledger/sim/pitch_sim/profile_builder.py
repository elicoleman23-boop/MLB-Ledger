"""
Batter and pitcher profile builders for the pitch-by-pitch sim.

The sim itself (pitch-by-pitch state machine + count-aware sampling) is
intentionally NOT in this phase. Phase 1 is purely the input layer:
given a batter's or pitcher's Statcast pitch-level DataFrame (the same
frames already produced by hit_ledger.data.statcast and
hit_ledger.data.pitcher_profile), produce structured, regressed
per-pitch-type profiles that later phases will consume.

Both builders tolerate empty or sparse DataFrames and fall back to
league priors defined in hit_ledger.config. Nothing in here crashes
when data is missing — upstream callers can build profiles
unconditionally and downstream sim code can trust the schema.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from hit_ledger.config import (
    LEAGUE_EV_LA,
    LEAGUE_O_CONTACT_BY_PITCH,
    LEAGUE_O_SWING_BY_PITCH,
    LEAGUE_Z_CONTACT_BY_PITCH,
    LEAGUE_Z_SWING_BY_PITCH,
    MIN_EV_LA_SAMPLE,
    MIN_PITCHES_PER_SPLIT,
    REGRESSION_K_CONTACT,
)


# Statcast description semantics.
#   - Swing = description NOT in _NON_SWING_DESCS
#   - Whiff = description in _WHIFF_DESCS
#   - Contact-on-swing = swung and not whiff (includes fouls, in-play)
_NON_SWING_DESCS = frozenset({
    "ball", "blocked_ball", "called_strike", "hit_by_pitch",
    "pitchout", "automatic_ball", "automatic_strike",
})
_WHIFF_DESCS = frozenset({
    "swinging_strike", "swinging_strike_blocked", "missed_bunt",
})

# Zone codes (Statcast): 1-9 are the 3x3 strike-zone grid; 11-14 are the
# four out-of-zone quadrants. Anything else (NaN, 0) we treat as unknown
# and drop from zone-conditioned rates.
_IN_ZONE = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9})
_OUT_OF_ZONE = frozenset({11, 12, 13, 14})

# Count-bucket classification for pitcher zone rate. Per spec:
#   'ahead'       = (0,1), (0,2), (1,2)       # pitcher's count
#   'even'        = (0,0), (1,1), (2,2)
#   'behind'      = (1,0), (2,0), (2,1), (3,1), (3,2)   # hitter's count
#   'must_strike' = (3,0), (3,1)              # pitcher must throw strike
# (3,1) falls in both 'behind' and 'must_strike' per spec; we route it to
# 'must_strike' because that bucket captures the stronger behavioral signal.
_COUNT_BUCKET = {
    (0, 0): "even",  (1, 1): "even",  (2, 2): "even",
    (0, 1): "ahead", (0, 2): "ahead", (1, 2): "ahead",
    (1, 0): "behind", (2, 0): "behind", (2, 1): "behind", (3, 2): "behind",
    (3, 0): "must_strike", (3, 1): "must_strike",
}
_COUNT_BUCKETS: tuple[str, ...] = ("ahead", "even", "behind", "must_strike")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _pitch_types() -> Iterable[str]:
    """Iterate over the 12 pitch types we carry league priors for."""
    return LEAGUE_Z_SWING_BY_PITCH.keys()


def _classify_swing_whiff(descriptions: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Vectorized (swung, whiff) flags for a description column."""
    desc = descriptions.fillna("").astype(str)
    swung = ~desc.isin(_NON_SWING_DESCS)
    whiff = desc.isin(_WHIFF_DESCS)
    return swung, whiff


def _regress_rate(raw: float, n: int, prior: float, k: int = REGRESSION_K_CONTACT) -> float:
    """Simple Bayesian regression toward league prior. Matches sim.matchup_v2._regress."""
    if n <= 0:
        return prior
    return (n * raw + k * prior) / (n + k)


def _batter_overall_ev_la(batter_df: pd.DataFrame) -> dict | None:
    """Batter's overall EV/LA distribution across all pitch types, or None if no data."""
    if batter_df.empty:
        return None
    has_both = (
        batter_df.get("launch_speed", pd.Series(dtype=float)).notna()
        & batter_df.get("launch_angle", pd.Series(dtype=float)).notna()
    )
    contact = batter_df[has_both]
    if contact.empty:
        return None
    return _ev_la_stats(contact["launch_speed"], contact["launch_angle"])


def _ev_la_stats(ev: pd.Series, la: pd.Series) -> dict:
    """Compute (mean_ev, sd_ev, mean_la, sd_la, corr_ev_la, n) for a joint EV/LA sample."""
    n = int(len(ev))
    if n == 0:
        return {**LEAGUE_EV_LA, "n": 0}
    ev_arr = np.asarray(ev, dtype=float)
    la_arr = np.asarray(la, dtype=float)
    mean_ev = float(np.mean(ev_arr))
    mean_la = float(np.mean(la_arr))
    sd_ev = float(np.std(ev_arr, ddof=1)) if n > 1 else LEAGUE_EV_LA["sd_ev"]
    sd_la = float(np.std(la_arr, ddof=1)) if n > 1 else LEAGUE_EV_LA["sd_la"]
    if n > 1 and sd_ev > 0 and sd_la > 0:
        corr = float(np.corrcoef(ev_arr, la_arr)[0, 1])
    else:
        corr = LEAGUE_EV_LA["corr_ev_la"]
    return {
        "mean_ev": mean_ev,
        "sd_ev": sd_ev,
        "mean_la": mean_la,
        "sd_la": sd_la,
        "corr_ev_la": corr,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------
def build_batter_pitch_profile(batter_df: pd.DataFrame) -> dict:
    """
    Build the batter-side inputs for the pitch-by-pitch sim.

    Returns a dict with these keys:
        z_swing_rate:    {pitch_type: float}
        o_swing_rate:    {pitch_type: float}
        z_contact_rate:  {pitch_type: float}
        o_contact_rate:  {pitch_type: float}
        ev_la_by_pitch:  {pitch_type: {mean_ev, sd_ev, mean_la, sd_la, corr_ev_la, n}}
        overall_batter_handedness: 'L' | 'R' | 'S' (most common 'stand')

    Rates are regressed toward LEAGUE_*_BY_PITCH priors using
    REGRESSION_K_CONTACT. EV/LA per pitch falls back to batter overall, then
    to LEAGUE_EV_LA, when samples are thin (< MIN_EV_LA_SAMPLE contact events).
    Empty/sparse DataFrames return a pure-prior profile without raising.
    """
    overall_ev_la = _batter_overall_ev_la(batter_df) if not batter_df.empty else None

    # Handedness: mode of 'stand' column when available
    if not batter_df.empty and "stand" in batter_df.columns:
        stand_mode = batter_df["stand"].dropna().mode()
        overall_hand = str(stand_mode.iat[0]) if not stand_mode.empty else "R"
    else:
        overall_hand = "R"

    profile = {
        "z_swing_rate": {},
        "o_swing_rate": {},
        "z_contact_rate": {},
        "o_contact_rate": {},
        "ev_la_by_pitch": {},
        "overall_batter_handedness": overall_hand,
    }

    if batter_df.empty or "description" not in batter_df.columns:
        # No data → emit league priors for every pitch type so the schema
        # is complete.
        for pt in _pitch_types():
            profile["z_swing_rate"][pt] = LEAGUE_Z_SWING_BY_PITCH[pt]
            profile["o_swing_rate"][pt] = LEAGUE_O_SWING_BY_PITCH[pt]
            profile["z_contact_rate"][pt] = LEAGUE_Z_CONTACT_BY_PITCH[pt]
            profile["o_contact_rate"][pt] = LEAGUE_O_CONTACT_BY_PITCH[pt]
            profile["ev_la_by_pitch"][pt] = {**LEAGUE_EV_LA, "n": 0}
        return profile

    swung_all, whiff_all = _classify_swing_whiff(batter_df["description"])
    zone_col = batter_df["zone"] if "zone" in batter_df.columns else pd.Series(
        [np.nan] * len(batter_df), index=batter_df.index
    )
    in_zone = zone_col.isin(_IN_ZONE)
    out_of_zone = zone_col.isin(_OUT_OF_ZONE)

    for pt in _pitch_types():
        pt_mask = batter_df["pitch_type"] == pt

        # Swing + contact rates, zone-conditioned
        for zone_name, zone_mask in (("z", in_zone), ("o", out_of_zone)):
            swing_prior = (
                LEAGUE_Z_SWING_BY_PITCH[pt] if zone_name == "z"
                else LEAGUE_O_SWING_BY_PITCH[pt]
            )
            contact_prior = (
                LEAGUE_Z_CONTACT_BY_PITCH[pt] if zone_name == "z"
                else LEAGUE_O_CONTACT_BY_PITCH[pt]
            )

            pitch_rows = pt_mask & zone_mask
            n_pitches = int(pitch_rows.sum())
            if n_pitches < MIN_PITCHES_PER_SPLIT:
                profile[f"{zone_name}_swing_rate"][pt] = swing_prior
                profile[f"{zone_name}_contact_rate"][pt] = contact_prior
                continue

            swung_sub = swung_all[pitch_rows]
            whiff_sub = whiff_all[pitch_rows]
            n_swings = int(swung_sub.sum())
            raw_swing = n_swings / n_pitches if n_pitches else 0.0
            if n_swings > 0:
                n_contact = int((swung_sub & ~whiff_sub).sum())
                raw_contact = n_contact / n_swings
            else:
                raw_contact = contact_prior
            profile[f"{zone_name}_swing_rate"][pt] = _regress_rate(
                raw_swing, n_pitches, swing_prior
            )
            profile[f"{zone_name}_contact_rate"][pt] = _regress_rate(
                raw_contact, n_swings, contact_prior
            )

        # EV/LA distribution from contact events of this pitch type
        pt_contact = batter_df[
            pt_mask
            & batter_df.get("launch_speed", pd.Series(dtype=float)).notna()
            & batter_df.get("launch_angle", pd.Series(dtype=float)).notna()
        ]
        if len(pt_contact) >= MIN_EV_LA_SAMPLE:
            profile["ev_la_by_pitch"][pt] = _ev_la_stats(
                pt_contact["launch_speed"], pt_contact["launch_angle"]
            )
        elif overall_ev_la is not None and overall_ev_la["n"] >= MIN_EV_LA_SAMPLE:
            profile["ev_la_by_pitch"][pt] = {**overall_ev_la}
        else:
            profile["ev_la_by_pitch"][pt] = {**LEAGUE_EV_LA, "n": int(len(pt_contact))}

    return profile


def build_pitcher_pitch_profile(
    pitcher_df: pd.DataFrame,
    arsenal: dict[str, float],
) -> dict:
    """
    Build the pitcher-side inputs for the pitch-by-pitch sim.

    Returns a dict with these keys:
        arsenal:         {pitch_type: share}  (passed through)
        zone_rate:       {pitch_type: {bucket: rate}}   # 4 count buckets
        whiff_rate:      {pitch_type: {'z': rate, 'o': rate}}
        ev_suppression:  {pitch_type: float}   # pitcher's mean_ev − league_mean
        la_influence:    {pitch_type: float}   # pitcher's mean_la − league_mean
        pitcher_handedness: 'L' | 'R'

    Count-bucket zone rates are shrunk toward the pitcher's overall zone rate
    for that pitch with k=50 (count effects are real but sample-noisy).
    If a pitcher has fewer than MIN_PITCHES_PER_SPLIT pitches of a type, the
    entire per-pitch block falls back to league priors / zero suppression.
    """
    # Handedness
    if not pitcher_df.empty and "p_throws" in pitcher_df.columns:
        throws_mode = pitcher_df["p_throws"].dropna().mode()
        pitcher_hand = str(throws_mode.iat[0]) if not throws_mode.empty else "R"
    else:
        pitcher_hand = "R"

    profile = {
        "arsenal": dict(arsenal or {}),
        "zone_rate": {},
        "whiff_rate": {},
        "ev_suppression": {},
        "la_influence": {},
        "pitcher_handedness": pitcher_hand,
    }

    if pitcher_df.empty or "description" not in pitcher_df.columns:
        for pt in _pitch_types():
            profile["zone_rate"][pt] = {b: _league_zone_rate(pt) for b in _COUNT_BUCKETS}
            profile["whiff_rate"][pt] = _league_whiff_rate(pt)
            profile["ev_suppression"][pt] = 0.0
            profile["la_influence"][pt] = 0.0
        return profile

    swung_all, whiff_all = _classify_swing_whiff(pitcher_df["description"])
    zone_col = pitcher_df["zone"] if "zone" in pitcher_df.columns else pd.Series(
        [np.nan] * len(pitcher_df), index=pitcher_df.index
    )
    in_zone = zone_col.isin(_IN_ZONE)
    out_of_zone = zone_col.isin(_OUT_OF_ZONE)

    # Count buckets from balls/strikes columns if present
    if {"balls", "strikes"}.issubset(pitcher_df.columns):
        bs_pairs = list(zip(pitcher_df["balls"].fillna(-1).astype(int),
                            pitcher_df["strikes"].fillna(-1).astype(int)))
        bucket_col = pd.Series(
            [_COUNT_BUCKET.get(bs) for bs in bs_pairs],
            index=pitcher_df.index,
        )
    else:
        bucket_col = pd.Series([None] * len(pitcher_df), index=pitcher_df.index)

    for pt in _pitch_types():
        pt_mask = pitcher_df["pitch_type"] == pt
        n_pt = int(pt_mask.sum())
        if n_pt < MIN_PITCHES_PER_SPLIT:
            profile["zone_rate"][pt] = {b: _league_zone_rate(pt) for b in _COUNT_BUCKETS}
            profile["whiff_rate"][pt] = _league_whiff_rate(pt)
            profile["ev_suppression"][pt] = 0.0
            profile["la_influence"][pt] = 0.0
            continue

        # Overall zone rate for this pitch type (used as the bucket-level prior)
        overall_zone_rate = float((pt_mask & in_zone).sum() / n_pt)

        bucket_rates: dict[str, float] = {}
        for bucket in _COUNT_BUCKETS:
            bucket_mask = pt_mask & (bucket_col == bucket)
            n_bucket = int(bucket_mask.sum())
            if n_bucket == 0:
                bucket_rates[bucket] = overall_zone_rate
                continue
            raw = int((bucket_mask & in_zone).sum()) / n_bucket
            bucket_rates[bucket] = _regress_rate(raw, n_bucket, overall_zone_rate, k=50)
        profile["zone_rate"][pt] = bucket_rates

        # Whiff rate split by zone (conditional on swing)
        whiff_block = {}
        for zone_name, zone_mask in (("z", in_zone), ("o", out_of_zone)):
            swings_mask = pt_mask & zone_mask & swung_all
            n_swings = int(swings_mask.sum())
            if n_swings == 0:
                whiff_block[zone_name] = 0.0
            else:
                whiff_block[zone_name] = float((swings_mask & whiff_all).sum() / n_swings)
        profile["whiff_rate"][pt] = whiff_block

        # EV / LA suppression vs league (single baseline; per-pitch deltas
        # are dominated by the pitcher's own contact, not pitch-type mean).
        pt_contact = pitcher_df[
            pt_mask
            & pitcher_df.get("launch_speed", pd.Series(dtype=float)).notna()
            & pitcher_df.get("launch_angle", pd.Series(dtype=float)).notna()
        ]
        if len(pt_contact) >= MIN_EV_LA_SAMPLE:
            profile["ev_suppression"][pt] = float(
                pt_contact["launch_speed"].mean() - LEAGUE_EV_LA["mean_ev"]
            )
            profile["la_influence"][pt] = float(
                pt_contact["launch_angle"].mean() - LEAGUE_EV_LA["mean_la"]
            )
        else:
            profile["ev_suppression"][pt] = 0.0
            profile["la_influence"][pt] = 0.0

    return profile


def _league_zone_rate(_pitch_type: str) -> float:
    """League-average in-zone share per pitch — rough proxy when pitcher data is thin."""
    # Zone rates cluster around 0.48-0.52 across pitch types; a single 0.50
    # prior is adequate for the Phase 1 fallback. Future refinement can add
    # a per-pitch league table if this turns out to matter.
    return 0.50


def _league_whiff_rate(pitch_type: str) -> dict[str, float]:
    """League whiff = 1 - league contact, per zone. Used as the 'no-data'
    fallback so downstream log-5 contact blending degrades cleanly to the
    batter rate instead of getting pushed by a spurious zero-whiff signal."""
    return {
        "z": 1.0 - LEAGUE_Z_CONTACT_BY_PITCH.get(pitch_type, 0.82),
        "o": 1.0 - LEAGUE_O_CONTACT_BY_PITCH.get(pitch_type, 0.60),
    }
