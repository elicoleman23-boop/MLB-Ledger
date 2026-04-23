"""Pitch-by-pitch simulator (Fix E) — Phase 1: profile scaffolding only.

The sim state machine and count-aware sampling arrive in later phases.
This phase exposes the two input builders so other modules can start
threading pitch profiles through the pipeline without depending on
unfinished sim internals.
"""
from hit_ledger.sim.pitch_sim.pa_engine import simulate_pa
from hit_ledger.sim.pitch_sim.profile_builder import (
    build_batter_pitch_profile,
    build_pitcher_pitch_profile,
)

__all__ = [
    "build_batter_pitch_profile",
    "build_pitcher_pitch_profile",
    "simulate_pa",
]
