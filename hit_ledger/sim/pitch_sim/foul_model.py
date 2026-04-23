"""Given contact was made, decide foul vs fair (ball in play)."""
from __future__ import annotations

import numpy as np

# Base foul rates (2024 league averages). Out-of-zone contact fouls off
# more often than in-zone contact because reached-for swings produce less
# barreled contact and more off-the-end-of-the-bat fouls.
_FOUL_RATE_IN_ZONE = 0.35
_FOUL_RATE_OUT_OF_ZONE = 0.52

# Additional foul probability when protecting with two strikes. Hitters
# choke up, shorten their swing, and foul off borderline pitches to stay
# alive — real-MLB effect is ~7-10 pp vs <2 strikes, so +0.08 is a
# reasonable middle estimate.
_TWO_STRIKE_FOUL_BUMP = 0.08

# Out-of-zone breaking balls foul off more than out-of-zone fastballs:
# hitters reach and get bat on ball but can't drive the movement, so
# contact produces fouls rather than fair BIP.
_OOZ_BREAKING_PITCHES = frozenset({"SL", "ST", "CU", "KC", "FS", "FO"})
_OOZ_BREAKING_BUMP = 0.04

_MAX_FOUL_RATE = 0.75


def sample_foul_given_contact(
    pitch_type: str,
    in_zone: bool,
    count: tuple[int, int],
    rng: np.random.Generator,
) -> bool:
    """
    Bernoulli draw on the foul rate conditional on contact being made.

    The rate combines three effects:
      - base rate by zone (in-zone 0.35 / out-of-zone 0.52)
      - two-strike bump (+0.08) capturing the protect-and-foul-off behavior
        that keeps 2-strike PAs alive without striking the batter out
      - out-of-zone breaking-ball bump (+0.04) for SL/ST/CU/KC/FS/FO
        because reached-for off-speed contact is rarely squared up

    Capped at _MAX_FOUL_RATE so extreme combinations can't produce a
    foul-certain outcome. Two-strike fouls themselves don't advance the
    count — the PA engine (pa_engine.simulate_pa) handles that.
    """
    base = _FOUL_RATE_IN_ZONE if in_zone else _FOUL_RATE_OUT_OF_ZONE
    if count[1] >= 2:
        base += _TWO_STRIKE_FOUL_BUMP
    if (not in_zone) and pitch_type in _OOZ_BREAKING_PITCHES:
        base += _OOZ_BREAKING_BUMP
    base = min(base, _MAX_FOUL_RATE)
    return bool(rng.random() < base)
