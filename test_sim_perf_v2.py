"""v2 sim perf test - per-PA probs should still run fast."""
import sys
import types
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Stub heavy deps
for mod_name in ['pybaseball', 'statsapi', 'streamlit', 'requests']:
    stub = types.ModuleType(mod_name)
    if mod_name == 'pybaseball':
        stub.statcast_batter = lambda **kw: None
        stub.statcast_pitcher = lambda **kw: None
        stub.statcast = lambda **kw: None
    sys.modules[mod_name] = stub

import numpy as np

from hit_ledger.sim.engine_v2 import simulate_v2
from hit_ledger.sim.matchup_v2 import MatchupV2, PAProbability


def make_matchup(batter_id: int, rng: np.random.Generator) -> MatchupV2:
    """Build a realistic per-PA matchup."""
    # Starter PAs with escalating TTO penalty
    starter_base_xba = rng.uniform(0.20, 0.32)
    tto_penalties = [0.0, 0.010, 0.022]  # TTO 1/2/3

    # Bullpen xBA
    bullpen_xba = rng.uniform(0.225, 0.265)

    # HR rate
    p_hr = rng.uniform(0.02, 0.06)

    pa_probs = []
    # 3 starter PAs at escalating TTOs
    for i, pen in enumerate(tto_penalties):
        xba = starter_base_xba + pen
        non_hr = xba - p_hr
        pa_probs.append(PAProbability(
            p_1b=non_hr * 0.75, p_2b=non_hr * 0.22, p_3b=non_hr * 0.03,
            p_hr=p_hr, source=f"starter_tto_{i+1}",
        ))
    # 2 bullpen PAs
    for _ in range(2):
        non_hr = bullpen_xba - p_hr
        pa_probs.append(PAProbability(
            p_1b=non_hr * 0.75, p_2b=non_hr * 0.22, p_3b=non_hr * 0.03,
            p_hr=p_hr, source="bullpen",
        ))

    return MatchupV2(
        batter_id=batter_id,
        starter_id=99999,
        pa_probs=pa_probs,
        expected_pa_vs_starter=3.0,
        expected_pa_vs_bullpen=1.5,
    )


def main():
    rng = np.random.default_rng(42)

    for n_batters in [50, 200, 300]:
        matchups = [make_matchup(i, rng) for i in range(n_batters)]
        slots = {m.batter_id: (i % 9) + 1 for i, m in enumerate(matchups)}

        t0 = time.perf_counter()
        results = simulate_v2(matchups, slots, n_sims=10_000, rng=rng)
        elapsed = time.perf_counter() - t0

        avg_p_hit = np.mean([r.p_1_hit for r in results])
        avg_p_2h = np.mean([r.p_2_hits for r in results])
        avg_p_hr = np.mean([r.p_1_hr for r in results])

        status = "OK" if elapsed < 2.0 else "SLOW"
        print(
            f"[{status}] {n_batters} batters, v2 (per-PA probs): "
            f"{elapsed:.3f}s | "
            f"1H={avg_p_hit:.3f} 2H={avg_p_2h:.3f} HR={avg_p_hr:.3f}"
        )


if __name__ == "__main__":
    main()
