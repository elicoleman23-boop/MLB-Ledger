"""Decide whether a given pitch lands in the strike zone."""
from __future__ import annotations

import numpy as np

from hit_ledger.sim.pitch_sim.profile_builder import _COUNT_BUCKET


def sample_in_zone(
    pitch_type: str,
    pitcher_profile: dict,
    count: tuple[int, int],
    rng: np.random.Generator,
    ump_k_dev: float = 0.0,
) -> bool:
    """
    Bernoulli draw on the pitcher's zone rate for (pitch_type, count bucket).

    Unknown counts fall back to 'even'. Unknown pitch types fall back to
    0.50 so samplers don't raise on arsenal/profile mismatches.

    ump_k_dev is the home plate umpire's K% deviation from league average
    (same units as matchup_v2 uses — e.g. +0.02 = +2pp K). A strict-zone
    ump effectively expands the called zone from the pitcher's POV, so
    we add ump_k_dev · 0.8 to the observed rate. Clipped to [0.30, 0.75]
    so extreme ump tendencies can't collapse or balloon the zone.
    """
    bucket = _COUNT_BUCKET.get(count, "even")
    pt_block = pitcher_profile.get("zone_rate", {}).get(pitch_type)
    if not pt_block:
        rate = 0.50
    else:
        rate = pt_block.get(bucket, pt_block.get("even", 0.50))
    rate += ump_k_dev * 0.8
    rate = max(0.30, min(0.75, rate))
    return bool(rng.random() < rate)
