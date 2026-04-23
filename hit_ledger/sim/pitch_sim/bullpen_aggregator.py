"""
Team-level bullpen pitch profile — Phase 3 stopgap.

Aggregates a team's relievers into a single synthetic pitcher profile by
pooling their Statcast pitch-level frames. Pitches are effectively
volume-weighted (a reliever who throws more appears more often in the
pooled frame). This is a placeholder: it captures the rough flavor of
what the team's bullpen does on average, but it ignores leverage and
usage probability entirely.

TODO(Fix F): replace this with a usage-weighted mixture. A proper model
needs per-reliever usage probability (by batter handedness, score state,
inning) and should either (a) sample one reliever per PA and use that
reliever's profile directly, or (b) compute a usage-weighted average of
the structured profile dicts. Until then, downstream callers should
treat this output as a rough team-level approximation.
"""
from __future__ import annotations

import pandas as pd

from hit_ledger.sim.pitch_sim.profile_builder import build_pitcher_pitch_profile


def build_team_bullpen_pitch_profile(
    reliever_ids: list[int],
    reliever_profiles: dict[int, pd.DataFrame],
    reliever_arsenals: dict[int, tuple[str, dict]],
) -> dict:
    """
    Build one synthetic pitch profile for a team's bullpen by concatenating
    every reliever's pitch-level Statcast frame and running it through
    build_pitcher_pitch_profile. The combined arsenal is the pool-level
    pitch mix (shares summing to 1.0 across all pooled pitches).

    Returns a profile with the same shape as build_pitcher_pitch_profile.
    Empty inputs produce a pure-league-prior profile without raising.
    """
    if not reliever_ids:
        return build_pitcher_pitch_profile(pd.DataFrame(), {})

    frames = []
    for rid in reliever_ids:
        df = reliever_profiles.get(rid)
        if df is None or df.empty:
            continue
        frames.append(df)

    if not frames:
        return build_pitcher_pitch_profile(pd.DataFrame(), {})

    combined = pd.concat(frames, ignore_index=True)

    # Pool-level arsenal: the empirical pitch-type mix across all relievers
    combined_arsenal: dict[str, float] = {}
    total = len(combined)
    if total > 0 and "pitch_type" in combined.columns:
        counts = combined["pitch_type"].value_counts()
        combined_arsenal = {pt: float(n) / total for pt, n in counts.items()}

    return build_pitcher_pitch_profile(combined, combined_arsenal)
