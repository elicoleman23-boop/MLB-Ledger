"""Decide whether the batter swings at a given pitch."""
from __future__ import annotations

import numpy as np

# Count-conditional multipliers on the base swing rate.
_SWING_MULT: dict[tuple[int, int], float] = {
    (3, 0): 0.25,  # take
    (3, 1): 0.60,
    (0, 2): 1.15,  # protect
    (1, 2): 1.15,
}
_SWING_CAP = 0.95


def sample_swing(
    pitch_type: str,
    in_zone: bool,
    count: tuple[int, int],
    batter_profile: dict,
    rng: np.random.Generator,
    ump_k_dev: float = 0.0,
    tto_o_swing_mult: float = 1.0,
) -> bool:
    """
    Look up the batter's z/o swing rate for this pitch type, apply a
    count multiplier (3-0 take, 3-1 semi-take, 0-2/1-2 protect), and
    draw a Bernoulli. Probabilities are clipped to [0, _SWING_CAP] so
    even max-protect counts leave room for a take.

    ump_k_dev: when the umpire calls more strikes (ump_k_dev > 0), batters
    chase less on out-of-zone pitches because they know close pitches will
    be called strikes anyway — subtract ump_k_dev · 0.3 from o-swing.
    In-zone swing is unaffected (the batter is swinging at a real strike).

    tto_o_swing_mult: batter sees the pitcher's stuff better later in the
    game, so chase rate drops. Applied to o-swing only, multiplicatively.
    """
    rate_key = "z_swing_rate" if in_zone else "o_swing_rate"
    # Default is league-neutral 0.50 if pitch type is unknown.
    base = batter_profile.get(rate_key, {}).get(pitch_type, 0.50)
    mult = _SWING_MULT.get(count, 1.0)
    rate = base * mult
    if not in_zone:
        rate -= ump_k_dev * 0.3
        rate *= tto_o_swing_mult
        rate = max(rate, 0.05)  # floor so no zero-chase on any count
    rate = min(rate, _SWING_CAP)
    rate = max(rate, 0.0)
    return bool(rng.random() < rate)
