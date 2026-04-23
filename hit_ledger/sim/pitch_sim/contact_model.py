"""Decide whether a swing results in contact (fair/foul) vs a whiff.

Combines the batter's contact rate (from build_batter_pitch_profile) with
the pitcher's contact rate (= 1 − whiff_rate) using Bill James log-5
against the league contact rate for that pitch type and zone. This is
the same odds-ratio machinery used for xBA in matchup_v2.py (Fix B).
"""
from __future__ import annotations

import numpy as np

from hit_ledger.config import (
    LEAGUE_O_CONTACT_BY_PITCH,
    LEAGUE_Z_CONTACT_BY_PITCH,
)
from hit_ledger.sim.matchup_v2 import _log5_blend


def sample_contact(
    pitch_type: str,
    in_zone: bool,
    batter_profile: dict,
    pitcher_profile: dict,
    rng: np.random.Generator,
    tto_whiff_mult: float = 1.0,
) -> bool:
    """
    Log-5 blend of batter contact and pitcher contact-allowed against the
    pitch-and-zone league baseline. Bernoulli draw.

    When the pitcher's whiff for this (pitch, zone) is 0 (treated as
    "no data" by the profile), pitcher_contact defaults to the league
    contact rate so log-5 collapses to the batter's rate cleanly.

    tto_whiff_mult scales the pitcher's whiff rate (batter timing improves
    later in the game → pitcher whiff drops). <1.0 makes the pitcher less
    effective at missing bats; values >1.0 would exaggerate whiff.
    """
    if in_zone:
        zone_key = "z"
        batter_key = "z_contact_rate"
        league_contact = LEAGUE_Z_CONTACT_BY_PITCH.get(pitch_type, 0.82)
    else:
        zone_key = "o"
        batter_key = "o_contact_rate"
        league_contact = LEAGUE_O_CONTACT_BY_PITCH.get(pitch_type, 0.60)

    batter_contact = batter_profile.get(batter_key, {}).get(pitch_type, league_contact)

    pitcher_whiff = pitcher_profile.get("whiff_rate", {}).get(pitch_type, {}).get(
        zone_key, None
    )
    if pitcher_whiff is None or pitcher_whiff <= 0.0:
        pitcher_contact = league_contact
    else:
        adj_whiff = max(0.0, min(1.0, pitcher_whiff * tto_whiff_mult))
        pitcher_contact = 1.0 - adj_whiff

    blended = _log5_blend(batter_contact, pitcher_contact, league_contact)
    return bool(rng.random() < blended)
