"""Sample (exit velocity, launch angle) for a ball-in-play event.

Draws from the batter's per-pitch-type joint distribution (built in
profile_builder), adjusts for pitcher EV/LA influence, and suppresses
EV for out-of-zone contact (harder to square up a pitch off the plate).
"""
from __future__ import annotations

import math

import numpy as np

from hit_ledger.config import LEAGUE_EV_LA

_OOZ_EV_SUPPRESSION_MPH = 3.0
_EV_CLIP = (40.0, 120.0)
_LA_CLIP = (-40.0, 90.0)


def sample_ev_la(
    pitch_type: str,
    in_zone: bool,
    batter_profile: dict,
    pitcher_profile: dict,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """
    Sample a correlated (EV, LA) pair using the Cholesky-like construction:
        z1, z2 ~ N(0, 1) iid
        ev = mean_ev + sd_ev · z1
        la = mean_la + sd_la · (corr · z1 + sqrt(1 − corr²) · z2)

    Then apply pitcher-specific EV suppression and LA influence for this
    pitch type, subtract an additional 3 mph for out-of-zone contact,
    and clip to plausible physical ranges.
    """
    block = batter_profile.get("ev_la_by_pitch", {}).get(pitch_type) or {**LEAGUE_EV_LA, "n": 0}
    mean_ev = float(block.get("mean_ev", LEAGUE_EV_LA["mean_ev"]))
    sd_ev = float(block.get("sd_ev", LEAGUE_EV_LA["sd_ev"]))
    mean_la = float(block.get("mean_la", LEAGUE_EV_LA["mean_la"]))
    sd_la = float(block.get("sd_la", LEAGUE_EV_LA["sd_la"]))
    corr = float(block.get("corr_ev_la", LEAGUE_EV_LA["corr_ev_la"]))
    # Keep correlation in (-1, 1) so sqrt(1 - corr²) is real and nonzero
    corr = max(-0.999, min(0.999, corr))

    z1, z2 = rng.standard_normal(2)
    ev = mean_ev + sd_ev * z1
    la = mean_la + sd_la * (corr * z1 + math.sqrt(1.0 - corr * corr) * z2)

    ev -= float(pitcher_profile.get("ev_suppression", {}).get(pitch_type, 0.0))
    la += float(pitcher_profile.get("la_influence", {}).get(pitch_type, 0.0))

    if not in_zone:
        ev -= _OOZ_EV_SUPPRESSION_MPH

    ev = float(np.clip(ev, *_EV_CLIP))
    la = float(np.clip(la, *_LA_CLIP))
    return ev, la
