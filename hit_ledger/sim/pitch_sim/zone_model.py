"""Decide whether a given pitch lands in the strike zone."""
from __future__ import annotations

import numpy as np

from hit_ledger.sim.pitch_sim.profile_builder import _COUNT_BUCKET


def sample_in_zone(
    pitch_type: str,
    pitcher_profile: dict,
    count: tuple[int, int],
    rng: np.random.Generator,
) -> bool:
    """
    Bernoulli draw on the pitcher's zone rate for (pitch_type, count bucket).

    Unknown counts fall back to 'even'. Unknown pitch types fall back to
    0.50 so samplers don't raise on arsenal/profile mismatches.
    """
    bucket = _COUNT_BUCKET.get(count, "even")
    pt_block = pitcher_profile.get("zone_rate", {}).get(pitch_type)
    if not pt_block:
        rate = 0.50
    else:
        rate = pt_block.get(bucket, pt_block.get("even", 0.50))
    return bool(rng.random() < rate)
