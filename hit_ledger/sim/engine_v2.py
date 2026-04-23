"""
v2 Monte Carlo engine — per-PA probability sequences.

Key difference from v1: each batter now has its own sequence of PA
probabilities (PA 1 = starter TTO 1, PA 2 = starter TTO 2, etc.).
The sim builds a per-batter, per-PA-slot probability matrix and samples
outcomes in parallel via NumPy.

Performance: still vectorized. The only overhead vs v1 is a small
per-batter loop to populate the per-PA cumulative probs matrix,
but sampling remains one big tensor op.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hit_ledger.config import (
    BABIP_NOISE_SD,
    DEFAULT_PA,
    N_SIMULATIONS,
    PA_BY_LINEUP_SLOT,
    RNG_SEED,
)
from hit_ledger.sim.matchup_v2 import MatchupV2


OUTCOME_TOTAL_BASES = np.array([0, 1, 2, 3, 4], dtype=np.int8)
OUTCOME_IS_HIT = np.array([0, 1, 1, 1, 1], dtype=np.int8)
OUTCOME_IS_HR = np.array([0, 0, 0, 0, 1], dtype=np.int8)


@dataclass
class BatterSimResultV2:
    batter_id: int
    p_1_hit: float
    p_2_hits: float
    p_1_hr: float
    p_tb_over_1_5: float
    p_tb_over_2_5: float
    expected_hits: float
    expected_tb: float


def _pas_for_slot(slot: int | None) -> float:
    if slot is None:
        return DEFAULT_PA
    return PA_BY_LINEUP_SLOT.get(int(slot), DEFAULT_PA)


def simulate_v2(
    matchups: list[MatchupV2],
    lineup_slots: dict[int, int],
    n_sims: int = N_SIMULATIONS,
    rng: np.random.Generator | None = None,
    babip_noise_sd: float = BABIP_NOISE_SD,
) -> list[BatterSimResultV2]:
    """
    Run the Monte Carlo using per-PA probabilities.

    For each batter, we have a list of PAProbability objects representing
    the ordered sequence of PAs. The length of that list is the maximum
    possible PAs; the actual count per sim is drawn from the lineup-slot
    PA distribution (e.g., 4.3 PAs = 4 guaranteed + 30% of a 5th).

    BABIP noise: when `babip_noise_sd > 0`, a normally-distributed
    multiplicative factor ~ N(1, babip_noise_sd), clipped to [0.5, 1.5],
    is applied to the 1B/2B/3B probabilities. One noise draw per
    (batter, sim) is SHARED across all of that batter's PAs in the
    simulated game — a first-order model of persistent within-game
    effects (weather, umpire, defensive form, starter's "stuff today").
    Per-PA independent noise averages out across 4-5 PAs per game and
    barely moves dispersion; game-level correlated noise is what
    produces realistic fat tails on P(≥2 hits), team totals, and
    alt-line markets. HR probability is left unchanged — HRs are not
    BABIP. The noise is centered at 1.0 so expected means are preserved.
    Setting `babip_noise_sd=0` short-circuits to the deterministic path
    and reproduces pre-noise results bit-for-bit.
    """
    if not matchups:
        return []

    rng = rng or np.random.default_rng(RNG_SEED)
    n_batters = len(matchups)

    # Build per-(batter, pa_idx, outcome) probability tensor.
    # Find max PA length across all matchups, pad shorter sequences with the
    # last available PA's probs (shouldn't matter since those slots are masked).
    max_pa_len = max(len(m.pa_probs) for m in matchups)

    probs = np.zeros((n_batters, max_pa_len, 5), dtype=np.float64)
    pa_counts = np.zeros(n_batters, dtype=np.float64)

    for i, m in enumerate(matchups):
        slot_pa = _pas_for_slot(lineup_slots.get(m.batter_id))
        pa_counts[i] = slot_pa

        # Fill probs for each available PA slot
        for j, pa in enumerate(m.pa_probs):
            p_1b = max(0.0, pa.p_1b)
            p_2b = max(0.0, pa.p_2b)
            p_3b = max(0.0, pa.p_3b)
            p_hr = max(0.0, pa.p_hr)
            total_hit = p_1b + p_2b + p_3b + p_hr
            if total_hit > 1.0:
                scale = 1.0 / total_hit
                p_1b, p_2b, p_3b, p_hr = (p_1b * scale, p_2b * scale,
                                           p_3b * scale, p_hr * scale)
                total_hit = 1.0
            probs[i, j, 0] = 1.0 - total_hit
            probs[i, j, 1] = p_1b
            probs[i, j, 2] = p_2b
            probs[i, j, 3] = p_3b
            probs[i, j, 4] = p_hr

        # Pad any remaining slots with the last available PA's probs
        # (these slots should be masked out anyway by pa_mask)
        if len(m.pa_probs) < max_pa_len and len(m.pa_probs) > 0:
            last_j = len(m.pa_probs) - 1
            for j in range(len(m.pa_probs), max_pa_len):
                probs[i, j] = probs[i, last_j]

    # Fractional PAs → randomize last PA's existence
    floor_pa = np.floor(pa_counts).astype(np.int32)
    frac_pa = pa_counts - floor_pa
    # Total simulated PA slots per batter cap at max_pa_len
    sim_max_pa = min(max_pa_len, int(floor_pa.max()) + 1)

    # Draw uniforms for every (batter, sim, pa): (n_batters, n_sims, sim_max_pa)
    uniforms = rng.random((n_batters, n_sims, sim_max_pa))

    outcomes = np.empty((n_batters, n_sims, sim_max_pa), dtype=np.int8)

    if babip_noise_sd <= 0:
        # Deterministic path — shared CDF across sims per (batter, PA slot).
        # Bit-for-bit identical to the pre-BABIP-noise engine.
        cum_probs = np.cumsum(probs[:, :sim_max_pa, :], axis=2)
        cum_probs[:, :, -1] = 1.0
        for i in range(n_batters):
            for j in range(sim_max_pa):
                outcomes[i, :, j] = np.searchsorted(cum_probs[i, j], uniforms[i, :, j])
    else:
        # BABIP noise path — per (batter, sim, PA) perturbation of 1B/2B/3B
        # probabilities. HR is left untouched. Loop over batters to keep
        # peak memory modest; each iteration allocates (n_sims, sim_max_pa, 5).
        static_probs = probs[:, :sim_max_pa, :]  # (n_batters, sim_max_pa, 5)
        for i in range(n_batters):
            # Game-level correlated noise: one draw per (batter, sim) shared
            # across all of that batter's PAs in the simulated game. Models
            # the persistent within-game effects (weather, umpire, defensive
            # form, starter's "stuff today") that make real MLB within-game
            # variance larger than independent Bernoulli trials would produce.
            # Per-PA independent noise averages out across 4-5 PAs per game
            # and yields near-zero dispersion growth; a single per-game draw
            # is a first-order model of the persistent effects.
            noise = rng.normal(
                loc=1.0, scale=babip_noise_sd, size=(n_sims, 1)
            )
            np.clip(noise, 0.5, 1.5, out=noise)
            # Broadcast batter i's per-PA-slot probs across all sims
            noisy_i = np.broadcast_to(
                static_probs[i][None, :, :], (n_sims, sim_max_pa, 5)
            ).copy()
            # Scale 1B/2B/3B slots by per-(sim, PA) noise; HR (slot 4) unchanged
            noise_3d = noise[:, :, None]  # (n_sims, sim_max_pa, 1)
            noisy_i[:, :, 1:4] *= noise_3d
            # Recompute out prob = 1 - (hits + HR), clip total hit prob
            total_hit = noisy_i[:, :, 1:].sum(axis=2)
            np.clip(total_hit, 0.0, 0.95, out=total_hit)
            noisy_i[:, :, 0] = 1.0 - total_hit
            # Cumulative + inverse-CDF, fully vectorized across (sim, PA)
            cum_i = np.cumsum(noisy_i, axis=2)
            cum_i[:, :, -1] = 1.0
            u_i = uniforms[i, :, :, None]  # (n_sims, sim_max_pa, 1)
            outcomes[i] = (u_i > cum_i).sum(axis=-1).astype(np.int8)

    # PA-existence mask (same logic as v1)
    pa_idx = np.arange(sim_max_pa).reshape(1, 1, sim_max_pa)
    base_mask = pa_idx < floor_pa.reshape(n_batters, 1, 1)
    extra_coin = rng.random((n_batters, n_sims))
    extra_pa_present = extra_coin < frac_pa.reshape(n_batters, 1)
    extra_mask = (
        (pa_idx == floor_pa.reshape(n_batters, 1, 1))
        & extra_pa_present[:, :, None]
    )
    pa_mask = base_mask | extra_mask

    # Aggregate outcomes
    hits_per_pa = OUTCOME_IS_HIT[outcomes] * pa_mask
    hrs_per_pa = OUTCOME_IS_HR[outcomes] * pa_mask
    tb_per_pa = OUTCOME_TOTAL_BASES[outcomes] * pa_mask

    hits_per_game = hits_per_pa.sum(axis=2)
    hrs_per_game = hrs_per_pa.sum(axis=2)
    tb_per_game = tb_per_pa.sum(axis=2)

    results: list[BatterSimResultV2] = []
    for i, m in enumerate(matchups):
        results.append(BatterSimResultV2(
            batter_id=m.batter_id,
            p_1_hit=float((hits_per_game[i] >= 1).mean()),
            p_2_hits=float((hits_per_game[i] >= 2).mean()),
            p_1_hr=float((hrs_per_game[i] >= 1).mean()),
            p_tb_over_1_5=float((tb_per_game[i] >= 2).mean()),
            p_tb_over_2_5=float((tb_per_game[i] >= 3).mean()),
            expected_hits=float(hits_per_game[i].mean()),
            expected_tb=float(tb_per_game[i].mean()),
        ))
    return results


def results_to_df_v2(results: list[BatterSimResultV2]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    return pd.DataFrame([r.__dict__ for r in results])
