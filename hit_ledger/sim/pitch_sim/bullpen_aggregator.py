"""
Team bullpen pitch profile.

Fix E Phase 3 pooled every reliever's pitch-level frame into one synthetic
pitcher at equal weight — this captured the bullpen's average flavor but
ignored leverage (the closer counts the same as the mop-up guy).

Fix F adds optional `usage_probs`: when supplied, each reliever's frame is
replicated in proportion to their usage probability before pooling. The
pool-level arsenal and downstream per-pitch-type regression therefore
reflect who's actually likely to pitch, not who happens to have thrown
the most Statcast pitches in the season.

Per-PA reliever sampling (drawing one reliever per PA, running simulate_pa
against THAT reliever's profile) was considered and deferred — it requires
plumbing a reliever_id through pa_engine and simulate_pbp, and the
usage-weighted pool approach was explicitly picked as the simpler path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hit_ledger.sim.pitch_sim.profile_builder import build_pitcher_pitch_profile


# Replication counts per reliever are derived from normalized usage probs
# times this constant; 100 gives enough resolution (1 pp of usage = 1
# replication) that even tail relievers contribute fractions of the
# pooled stats without the total blowing up.
_USAGE_WEIGHT_SCALE = 100


def build_team_bullpen_pitch_profile(
    reliever_ids: list[int],
    reliever_profiles: dict[int, pd.DataFrame],
    reliever_arsenals: dict[int, tuple[str, dict]],
    usage_probs: dict[int, float] | None = None,
) -> dict:
    """
    Build one synthetic pitch profile for a team's bullpen by pooling
    every reliever's pitch-level Statcast frame and running it through
    `build_pitcher_pitch_profile`.

    When `usage_probs` is supplied, each reliever's frame is replicated
    by `ceil(_USAGE_WEIGHT_SCALE * prob / total_prob)` before pooling —
    a usage-weighted mixture rather than the equal-weight pool. Relievers
    that appear in `reliever_ids` but not in `usage_probs` get a tiny
    residual weight (1 replication) so they're present but don't dominate.

    Empty inputs produce a pure-league-prior profile without raising.
    """
    if not reliever_ids:
        return build_pitcher_pitch_profile(pd.DataFrame(), {})

    frames: list[pd.DataFrame] = []
    norm_probs = _normalize_usage_probs(usage_probs, reliever_ids)

    for rid in reliever_ids:
        df = reliever_profiles.get(rid)
        if df is None or df.empty:
            continue
        weight = norm_probs.get(rid, 0.0)
        reps = max(1, int(np.ceil(_USAGE_WEIGHT_SCALE * weight)))
        for _ in range(reps):
            frames.append(df)

    if not frames:
        return build_pitcher_pitch_profile(pd.DataFrame(), {})

    combined = pd.concat(frames, ignore_index=True)

    combined_arsenal: dict[str, float] = {}
    total = len(combined)
    if total > 0 and "pitch_type" in combined.columns:
        counts = combined["pitch_type"].value_counts()
        combined_arsenal = {pt: float(n) / total for pt, n in counts.items()}

    return build_pitcher_pitch_profile(combined, combined_arsenal)


def _normalize_usage_probs(
    usage_probs: dict[int, float] | None,
    reliever_ids: list[int],
) -> dict[int, float]:
    """Renormalize usage_probs over the relievers we actually have
    frames for; fill in missing relievers with a tiny residual so every
    roster member still contributes at least one replication."""
    if not usage_probs:
        # Equal weight fallback (matches Fix E Phase 3 behavior)
        if not reliever_ids:
            return {}
        uniform = 1.0 / len(reliever_ids)
        return {rid: uniform for rid in reliever_ids}

    # Clip to the relievers actually present in this roster
    present = {rid: float(usage_probs.get(rid, 0.0)) for rid in reliever_ids}
    total = sum(present.values())
    if total <= 0:
        uniform = 1.0 / max(len(reliever_ids), 1)
        return {rid: uniform for rid in reliever_ids}

    normed = {rid: w / total for rid, w in present.items()}
    # Give never-mentioned relievers a tiny floor so they still appear
    floor = 0.01 / max(len(reliever_ids), 1)
    for rid in reliever_ids:
        if normed.get(rid, 0.0) <= 0:
            normed[rid] = floor
    # Final renormalization
    total = sum(normed.values())
    return {rid: w / total for rid, w in normed.items()}
