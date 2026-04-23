"""PA-level simulator — drives the pitch-by-pitch loop.

Walks a single plate appearance pitch by pitch using the component
samplers (pitch type, zone, swing, contact, foul, EV/LA) and the
outcome grid. Returns a structured record of the outcome, the final
count, and every pitch's state for downstream analytics.

Scope: one plate appearance. Aggregating PAs into games/innings is
Phase 3.
"""
from __future__ import annotations

import numpy as np

from hit_ledger.sim.pitch_sim.contact_model import sample_contact
from hit_ledger.sim.pitch_sim.contact_quality import sample_ev_la
from hit_ledger.sim.pitch_sim.foul_model import sample_foul_given_contact
from hit_ledger.sim.pitch_sim.outcome_grid import sample_outcome_from_ev_la
from hit_ledger.sim.pitch_sim.pitch_selection import sample_pitch_type
from hit_ledger.sim.pitch_sim.swing_decision import sample_swing
from hit_ledger.sim.pitch_sim.zone_model import sample_in_zone

# Rough league HBP rate conditional on a ball being called (out-of-zone,
# no-swing). Keeps HBP a rare outcome instead of ignoring it entirely.
_HBP_PROB_PER_CALLED_BALL = 0.004


def simulate_pa(
    batter_profile: dict,
    pitcher_profile: dict,
    rng: np.random.Generator,
    park_hr_mult: float = 1.0,
    park_hit_mult: float = 1.0,
    max_pitches: int = 12,
) -> dict:
    """
    Simulate a single plate appearance pitch by pitch.

    Returns a dict:
        outcome:        'out' | '1B' | '2B' | '3B' | 'HR'
                        | 'BB' | 'K_swinging' | 'K_looking' | 'HBP'
        n_pitches:      int — pitches thrown in the PA
        final_count:    (balls, strikes)
        pitch_sequence: list[dict] — per-pitch telemetry with keys
                        pitch_type, in_zone, swung, contact, foul, ev, la.
                        Fields irrelevant to a given pitch are None.

    The loop short-circuits on K (3 strikes), BB (4 balls), ball in play,
    or HBP. If `max_pitches` is hit without terminating (very rare), the
    PA is resolved as an out — this is a belt-and-suspenders safety
    bound, not a real MLB termination condition.
    """
    count = (0, 0)
    pitch_sequence: list[dict] = []

    def _record(**kwargs) -> dict:
        entry = {
            "pitch_type": None,
            "in_zone": None,
            "swung": None,
            "contact": None,
            "foul": None,
            "ev": None,
            "la": None,
        }
        entry.update(kwargs)
        pitch_sequence.append(entry)
        return entry

    def _result(outcome: str) -> dict:
        return {
            "outcome": outcome,
            "n_pitches": len(pitch_sequence),
            "final_count": count,
            "pitch_sequence": pitch_sequence,
        }

    for _ in range(max_pitches):
        pt = sample_pitch_type(pitcher_profile, count, rng)
        in_zone = sample_in_zone(pt, pitcher_profile, count, rng)
        swung = sample_swing(pt, in_zone, count, batter_profile, rng)

        if not swung:
            _record(pitch_type=pt, in_zone=in_zone, swung=False)
            if in_zone:
                count = (count[0], count[1] + 1)
                if count[1] >= 3:
                    return _result("K_looking")
            else:
                # Rare HBP on called-ball pitches; otherwise ball → count+1
                if rng.random() < _HBP_PROB_PER_CALLED_BALL:
                    return _result("HBP")
                count = (count[0] + 1, count[1])
                if count[0] >= 4:
                    return _result("BB")
            continue

        contact = sample_contact(pt, in_zone, batter_profile, pitcher_profile, rng)
        if not contact:
            _record(pitch_type=pt, in_zone=in_zone, swung=True, contact=False)
            count = (count[0], count[1] + 1)
            if count[1] >= 3:
                return _result("K_swinging")
            continue

        is_foul = sample_foul_given_contact(pt, in_zone, count, rng)
        if is_foul:
            _record(pitch_type=pt, in_zone=in_zone, swung=True, contact=True,
                    foul=True)
            # Foul with < 2 strikes advances to another strike; with 2
            # strikes the count stays put (can't strike out on a foul).
            if count[1] < 2:
                count = (count[0], count[1] + 1)
            continue

        # Ball in play — sample EV/LA, apply park, pick final outcome
        ev, la = sample_ev_la(pt, in_zone, batter_profile, pitcher_profile, rng)
        outcome = sample_outcome_from_ev_la(
            ev, la, rng,
            park_hr_mult=park_hr_mult,
            park_hit_mult=park_hit_mult,
        )
        _record(pitch_type=pt, in_zone=in_zone, swung=True, contact=True,
                foul=False, ev=ev, la=la)
        return _result(outcome)

    # max_pitches guard — resolve as an out so downstream aggregators
    # always see a concrete outcome.
    return _result("out")
