"""Sample a pitch type from the pitcher's arsenal, with count-aware shifts."""
from __future__ import annotations

import numpy as np

from hit_ledger.sim.pitch_sim.profile_builder import _COUNT_BUCKET

# Counts where the pitcher is forced to throw a strike — shift mix toward
# whichever pitch the profile marks with the highest must_strike zone_rate.
_MUST_STRIKE_BOOST = 0.15
_MUST_STRIKE_COUNTS = frozenset({(3, 0), (3, 1), (2, 0)})

# Two-strike, zero/one-ball counts where the pitcher chases a whiff —
# shift toward the highest-whiff pitch.
_WHIFF_BOOST = 0.10
_TWO_STRIKE_COUNTS = frozenset({(0, 2), (1, 2)})


def _normalize(shares: dict[str, float]) -> dict[str, float]:
    total = sum(shares.values())
    if total <= 0:
        return shares
    return {k: v / total for k, v in shares.items()}


def _bump_share(shares: dict[str, float], target: str, bump: float) -> dict[str, float]:
    """
    Bump a target pitch's share by *exactly* `bump` percentage points
    (i.e. +0.15 → target lands at original + 0.15), and shrink the other
    pitches' shares proportionally so the result still sums to 1.0.

    The naive approach (add bump, then renormalize over all shares) dilutes
    the additive constant because the divisor grows — a +0.15 add on a
    normalized arsenal produces only a ~+6-7 pp bump at the target after
    renormalization. This function preserves the intended tilt.
    """
    if target not in shares:
        return _normalize(shares)
    original = shares[target]
    others_sum = sum(v for k, v in shares.items() if k != target)
    target_new = min(1.0, max(0.0, original + bump))
    remainder = 1.0 - target_new
    out = dict(shares)
    if others_sum > 0 and remainder > 0:
        scale = remainder / others_sum
        for k in out:
            if k != target:
                out[k] *= scale
    else:
        for k in out:
            if k != target:
                out[k] = 0.0
    out[target] = target_new
    return out


def _best_pitch_by(
    arsenal: dict[str, float],
    score: dict[str, float],
) -> str | None:
    """Pick the pitch with the highest score value, restricted to arsenal keys."""
    candidates = [(pt, score.get(pt, 0.0)) for pt in arsenal]
    if not candidates:
        return None
    return max(candidates, key=lambda kv: kv[1])[0]


def sample_pitch_type(
    pitcher_profile: dict,
    count: tuple[int, int],
    rng: np.random.Generator,
) -> str:
    """
    Sample a pitch type from the pitcher's arsenal. In hitter's counts
    (3-0, 3-1, 2-0), shift mix by +15% toward the pitch with the highest
    zone_rate['must_strike']. In 0-2 / 1-2, shift +10% toward the pitch
    with the highest whiff rate (max of z and o). Otherwise use the flat
    arsenal.

    If the arsenal is empty, fall back to 'FF'.
    """
    arsenal = pitcher_profile.get("arsenal") or {}
    if not arsenal:
        return "FF"

    shares = dict(arsenal)

    if count in _MUST_STRIKE_COUNTS:
        must_strike_scores = {
            pt: pitcher_profile.get("zone_rate", {}).get(pt, {}).get("must_strike", 0.0)
            for pt in arsenal
        }
        target = _best_pitch_by(arsenal, must_strike_scores)
        if target is not None:
            shares = _bump_share(shares, target, _MUST_STRIKE_BOOST)
    elif count in _TWO_STRIKE_COUNTS:
        whiff_scores = {}
        for pt in arsenal:
            whiff_block = pitcher_profile.get("whiff_rate", {}).get(pt, {}) or {}
            whiff_scores[pt] = max(whiff_block.get("z", 0.0), whiff_block.get("o", 0.0))
        target = _best_pitch_by(arsenal, whiff_scores)
        if target is not None:
            shares = _bump_share(shares, target, _WHIFF_BOOST)
    else:
        shares = _normalize(shares)
    pitch_types = list(shares.keys())
    probs = np.array([shares[pt] for pt in pitch_types], dtype=np.float64)
    # Guard against numerical drift keeping probs summing to ~1
    probs = probs / probs.sum()
    idx = rng.choice(len(pitch_types), p=probs)
    return pitch_types[idx]
