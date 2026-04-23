"""Given contact was made, decide foul vs fair (ball in play)."""
from __future__ import annotations

import numpy as np

# League approximation: ~30% of in-zone contact is foul; out-of-zone
# contact fouls off more like 45% of the time because it's reached-for.
# Two-strike count logic (fouls don't strike the batter out) is handled
# by the caller in pa_engine.
_FOUL_RATE_IN_ZONE = 0.30
_FOUL_RATE_OUT_OF_ZONE = 0.45


def sample_foul_given_contact(
    pitch_type: str,
    in_zone: bool,
    count: tuple[int, int],
    rng: np.random.Generator,
) -> bool:
    """Bernoulli draw on the foul rate conditional on contact being made."""
    rate = _FOUL_RATE_IN_ZONE if in_zone else _FOUL_RATE_OUT_OF_ZONE
    return bool(rng.random() < rate)
