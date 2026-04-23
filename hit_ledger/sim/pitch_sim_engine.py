"""
Top-level pitch-by-pitch game engine (Fix E Phase 3).

For each batter-sim, this walks every PA through the pitch-level
simulator in hit_ledger.sim.pitch_sim.pa_engine.simulate_pa, respects the
starter→bullpen transition encoded on each MatchupV2's pa_probs sources,
and aggregates outcomes into the same BatterSimResultV2 shape as
engine_v2.simulate_v2 so downstream callers (pipeline, UI, caching) can
consume either result type interchangeably.

This engine is intentionally slower than engine_v2 (Python per-pitch
loop, no vectorization across sims). The default `n_sims=1000` keeps a
full slate under ~30s; `n_sims=10_000` is feasible for final-run quality
but won't match engine_v2's throughput.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from hit_ledger.config import (
    DEFAULT_PA,
    PA_BY_LINEUP_SLOT,
    RNG_SEED,
)
from hit_ledger.config import UMPIRE_K_XBA_SENSITIVITY
from hit_ledger.sim.engine_v2 import BatterSimResultV2
from hit_ledger.sim.matchup_v2 import MatchupV2, PAProbability
from hit_ledger.sim.pitch_sim import simulate_pa


def _pas_for_slot(slot: int | None) -> float:
    if slot is None:
        return DEFAULT_PA
    return PA_BY_LINEUP_SLOT.get(int(slot), DEFAULT_PA)


_TB_BY_OUTCOME = {"1B": 1, "2B": 2, "3B": 3, "HR": 4}


def _tto_level_from_source(source: str) -> int:
    """Map the PA source tag to a TTO level. Bullpen PAs get level 0 so the
    pa_engine's TTO adjustments are skipped entirely."""
    if source == "starter_tto_1":
        return 1
    if source == "starter_tto_2":
        return 2
    if source == "starter_tto_3":
        return 3
    return 0  # "bullpen" or anything else


def _build_pa_contexts(m: MatchupV2) -> list[dict]:
    """Reconstruct per-PA (ump, TTO, bullpen) context from a MatchupV2.

    ump_k_dev is recovered by inverting the transformation in build_matchup_v2:
        ump_xba_adj = -ump_k_dev * 100 * UMPIRE_K_XBA_SENSITIVITY
    which means
        ump_k_dev = -ump_xba_adj / (100 * UMPIRE_K_XBA_SENSITIVITY)
    We apply the same ump_k_dev to every PA in the game — the home plate
    umpire is constant across PAs.
    """
    if UMPIRE_K_XBA_SENSITIVITY:
        ump_k_dev = -m.umpire_adjustment / (100 * UMPIRE_K_XBA_SENSITIVITY)
    else:
        ump_k_dev = 0.0
    contexts = []
    for pa in m.pa_probs:
        contexts.append({
            "tto": _tto_level_from_source(pa.source),
            "ump_k_dev": ump_k_dev,
            "is_bullpen": pa.source == "bullpen",
        })
    return contexts


def simulate_pbp(
    matchups: list[MatchupV2],
    batter_pitch_profiles: dict[int, dict],
    pitcher_pitch_profiles: dict[int, dict],
    batter_bullpen_profiles: dict[int, dict],
    lineup_slots: dict[int, int],
    n_sims: int = 1_000,
    rng: np.random.Generator | None = None,
    batter_park_mults: dict[int, tuple[float, float]] | None = None,
) -> list[BatterSimResultV2]:
    """
    Run the pitch-by-pitch Monte Carlo and return engine_v2-compatible
    BatterSimResultV2 objects.

    Parameters
    ----------
    matchups : list[MatchupV2]
        Same objects engine_v2 consumes; their pa_probs[i].source tag
        ("starter_tto_*" vs "bullpen") drives which pitcher profile to
        use for PA slot i.
    batter_pitch_profiles : dict[int, dict]
        batter_id → output of build_batter_pitch_profile.
    pitcher_pitch_profiles : dict[int, dict]
        starter_id → output of build_pitcher_pitch_profile. Starter profile
        applies to every PA whose source starts with "starter".
    batter_bullpen_profiles : dict[int, dict]
        batter_id → the bullpen pitch profile for the OPPOSING team's
        bullpen. Keyed by batter rather than team so the engine doesn't
        need to know team routing; the pipeline resolves that upstream.
    lineup_slots : dict[int, int]
        batter_id → lineup slot. Drives total PA count per sim via the
        same floor + Bernoulli(frac) logic engine_v2 uses.
    n_sims : int
        Monte Carlo iterations per batter. Default 1000 (10× lower than
        engine_v2's default) since each PA walks pitch-by-pitch in Python.
    rng : np.random.Generator | None
    batter_park_mults : dict[int, tuple[float, float]] | None
        batter_id → (park_hit_mult, park_hr_mult). Missing entries default
        to (1.0, 1.0).
    """
    rng = rng or np.random.default_rng(RNG_SEED)
    results: list[BatterSimResultV2] = []

    for m in matchups:
        bp = batter_pitch_profiles.get(m.batter_id, {})
        starter_prof = pitcher_pitch_profiles.get(m.starter_id, {})
        bullpen_prof = batter_bullpen_profiles.get(m.batter_id, {})

        park_hit_mult, park_hr_mult = (batter_park_mults or {}).get(
            m.batter_id, (1.0, 1.0)
        )

        slot_pa = _pas_for_slot(lineup_slots.get(m.batter_id))
        floor_pa = int(np.floor(slot_pa))
        frac_pa = float(slot_pa - floor_pa)

        # Precompute per-PA-slot pitcher profile lookup from the matchup's
        # starter/bullpen source tags. PA indices beyond the length of
        # pa_probs (possible when the Bernoulli fractional PA fires but
        # matchup was built with a tight slot estimate) fall back to the
        # bullpen profile, since extra PAs happen late in games.
        per_slot_is_starter: list[bool] = [
            pa.source.startswith("starter") for pa in m.pa_probs
        ]
        per_slot_context: list[dict] = _build_pa_contexts(m)
        # Fallback context used for deep-extras PAs (beyond pa_probs length):
        # treat as bullpen, use the ump_k_dev we already recovered, TTO=0.
        _fallback_context = {
            "tto": 0,
            "ump_k_dev": per_slot_context[0]["ump_k_dev"] if per_slot_context else 0.0,
            "is_bullpen": True,
        }

        hits_per_sim = np.zeros(n_sims, dtype=np.int32)
        tb_per_sim = np.zeros(n_sims, dtype=np.int32)
        hrs_per_sim = np.zeros(n_sims, dtype=np.int32)

        for sim in range(n_sims):
            n_pa = floor_pa + (1 if rng.random() < frac_pa else 0)
            for pa_idx in range(n_pa):
                if pa_idx < len(per_slot_is_starter):
                    use_starter = per_slot_is_starter[pa_idx]
                    ctx = per_slot_context[pa_idx]
                else:
                    use_starter = False  # deep extras → bullpen
                    ctx = _fallback_context
                pitcher_prof = starter_prof if use_starter else bullpen_prof

                result = simulate_pa(
                    bp,
                    pitcher_prof,
                    rng,
                    park_hr_mult=park_hr_mult,
                    park_hit_mult=park_hit_mult,
                    pa_context=ctx,
                )
                outcome = result["outcome"]
                tb = _TB_BY_OUTCOME.get(outcome, 0)
                if tb > 0:
                    hits_per_sim[sim] += 1
                    tb_per_sim[sim] += tb
                    if outcome == "HR":
                        hrs_per_sim[sim] += 1

        results.append(BatterSimResultV2(
            batter_id=m.batter_id,
            p_1_hit=float((hits_per_sim >= 1).mean()),
            p_2_hits=float((hits_per_sim >= 2).mean()),
            p_1_hr=float((hrs_per_sim >= 1).mean()),
            p_tb_over_1_5=float((tb_per_sim >= 2).mean()),
            p_tb_over_2_5=float((tb_per_sim >= 3).mean()),
            expected_hits=float(hits_per_sim.mean()),
            expected_tb=float(tb_per_sim.mean()),
        ))

    return results


def sample_one_trace(
    matchup: MatchupV2,
    batter_pitch_profile: dict,
    pitcher_pitch_profile: dict,
    bullpen_pitch_profile: dict,
    rng: np.random.Generator,
    park_hit_mult: float = 1.0,
    park_hr_mult: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Run one full sim's worth of PAs for a single batter and return the
    raw PA records (pitch_sequence included) so the UI can display a
    sample trace. Not used in the simulate_pbp hot loop.
    """
    slot_pa = DEFAULT_PA
    # We don't know the slot here without a lookup, so just run one PA per
    # source slot the matchup already encodes. That's a reasonable "one
    # game" sample for inspection purposes.
    contexts = _build_pa_contexts(matchup)
    traces = []
    for pa, ctx in zip(matchup.pa_probs, contexts):
        use_starter = pa.source.startswith("starter")
        pitcher_prof = pitcher_pitch_profile if use_starter else bullpen_pitch_profile
        rec = simulate_pa(
            batter_pitch_profile,
            pitcher_prof,
            rng,
            park_hr_mult=park_hr_mult,
            park_hit_mult=park_hit_mult,
            pa_context=ctx,
        )
        rec["source"] = pa.source
        traces.append(rec)
    return traces
