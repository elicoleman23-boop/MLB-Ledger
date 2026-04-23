"""Map (EV, LA) to outcome probabilities and sample a batted-ball outcome.

Phase-2 coarse grid approximating the 2024 Statcast EV/LA → outcome
surface. The probabilities below are hand-picked bucket averages; each
tuple sums to 1.0 and represents (out, 1B, 2B, 3B, HR).

TODO: replace this with a data-driven grid built from historical Statcast
(KDE or a 2D gradient-boosted classifier on at-risk events).
"""
from __future__ import annotations

import numpy as np

_OUTCOME_KEYS = ("out", "1B", "2B", "3B", "HR")


def _bucket(ev: float, la: float) -> tuple[float, float, float, float, float]:
    """Return (out, 1B, 2B, 3B, HR) for a single (ev, la) cell."""
    # Weakest contact regardless of LA
    if ev < 60:
        return (0.95, 0.05, 0.00, 0.00, 0.00)

    if la < 0:  # grounder
        if ev < 85:
            return (0.73, 0.22, 0.04, 0.01, 0.00)
        if ev < 95:
            return (0.66, 0.26, 0.06, 0.02, 0.00)
        return (0.58, 0.32, 0.07, 0.03, 0.00)

    if la < 10:  # low liner
        if ev < 90:
            return (0.55, 0.38, 0.06, 0.01, 0.00)
        return (0.30, 0.52, 0.15, 0.03, 0.00)

    if la < 25:  # liner / barrel approach
        if ev < 90:
            return (0.48, 0.40, 0.10, 0.02, 0.00)
        if ev < 100:
            return (0.30, 0.42, 0.22, 0.03, 0.03)
        return (0.10, 0.35, 0.28, 0.03, 0.24)

    if la < 40:  # fly ball / HR zone
        if ev < 90:
            return (0.72, 0.18, 0.07, 0.01, 0.02)
        if ev < 95:
            return (0.55, 0.15, 0.14, 0.02, 0.14)
        if ev < 100:
            return (0.30, 0.08, 0.10, 0.02, 0.50)
        return (0.08, 0.02, 0.03, 0.01, 0.86)

    if la < 50:  # high fly / barrel edge
        if ev < 95:
            return (0.85, 0.08, 0.05, 0.00, 0.02)
        return (0.45, 0.10, 0.10, 0.00, 0.35)

    # la >= 50: infield popup
    return (0.98, 0.02, 0.00, 0.00, 0.00)


def ev_la_to_outcome_probs(ev: float, la: float) -> dict[str, float]:
    """Dict form of the bucket lookup; returned dict sums to 1.0."""
    probs = _bucket(ev, la)
    return dict(zip(_OUTCOME_KEYS, probs))


def sample_outcome_from_ev_la(
    ev: float,
    la: float,
    rng: np.random.Generator,
    park_hr_mult: float = 1.0,
    park_hit_mult: float = 1.0,
) -> str:
    """
    Apply park adjustments — HR prob scaled by park_hr_mult, hits by
    park_hit_mult — renormalize, and categorical-sample the outcome.
    """
    probs = _bucket(ev, la)
    p_out, p_1b, p_2b, p_3b, p_hr = probs
    if park_hr_mult != 1.0:
        p_hr *= park_hr_mult
    if park_hit_mult != 1.0:
        p_1b *= park_hit_mult
        p_2b *= park_hit_mult
        p_3b *= park_hit_mult

    total = p_out + p_1b + p_2b + p_3b + p_hr
    if total <= 0:
        return "out"
    norm = np.array([p_out, p_1b, p_2b, p_3b, p_hr], dtype=np.float64) / total
    idx = rng.choice(len(_OUTCOME_KEYS), p=norm)
    return _OUTCOME_KEYS[idx]
