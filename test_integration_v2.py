"""v2 integration test: synthetic batter -> build_matchup_v2 -> simulate_v2."""
import sys
import types
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

for mod_name in ['pybaseball', 'statsapi', 'streamlit', 'requests']:
    stub = types.ModuleType(mod_name)
    if mod_name == 'pybaseball':
        stub.statcast_batter = lambda **kw: None
        stub.statcast_pitcher = lambda **kw: None
        stub.statcast = lambda **kw: None
    sys.modules[mod_name] = stub

import numpy as np
import pandas as pd

from hit_ledger.sim.engine_v2 import simulate_v2
from hit_ledger.sim.matchup_v2 import build_matchup_v2


def synth_batter_df(n_pitches: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    pitch_types = rng.choice(
        ["FF", "SI", "SL", "CU", "CH", "FC"],
        size=n_pitches,
        p=[0.35, 0.18, 0.20, 0.10, 0.12, 0.05],
    )
    p_throws = rng.choice(["L", "R"], size=n_pitches, p=[0.3, 0.7])
    is_pa_end = rng.random(n_pitches) < 0.4
    events = np.where(is_pa_end,
                      np.where(rng.random(n_pitches) < 0.28, "single", "field_out"),
                      "")
    hr_mask = is_pa_end & (rng.random(n_pitches) < 0.04)
    events = np.where(hr_mask, "home_run", events)
    xba = np.where(is_pa_end, rng.beta(2, 5, size=n_pitches), np.nan)

    return pd.DataFrame({
        "game_date": pd.date_range("2024-04-01", periods=n_pitches, freq="6h"),
        "pitch_type": pitch_types,
        "p_throws": p_throws,
        "stand": "R",
        "events": events,
        "description": "",
        "estimated_ba_using_speedangle": xba,
        "launch_speed": 90.0,
        "launch_angle": 15.0,
        "pitcher": rng.integers(100000, 999999, size=n_pitches),
    })


def main():
    batter_df = synth_batter_df()

    # Three test scenarios
    scenarios = [
        {
            "name": "Elite starter, avg bullpen, neutral ump, Yankee Stadium",
            "starter_workload": {
                "avg_ip_per_start": 6.2,  # deep into games
                "starts_sampled": 5,
                "season_xba": 0.215,       # elite
            },
            "tto_splits": {
                "xba": {1: 0.210, 2: 0.225, 3: 0.245},  # big TTO3 jump
                "pa":  {1: 250, 2: 220, 3: 180},
            },
            "bullpen_profile": {
                "xba_vs_r": 0.243, "xba_vs_l": 0.245,
                "pa_vs_r": 800, "pa_vs_l": 700,
            },
            "umpire_k_dev": 0.0,
            "venue": "Yankee Stadium",
        },
        {
            "name": "Soft starter (short outings), nasty bullpen, pitcher-friendly ump",
            "starter_workload": {
                "avg_ip_per_start": 4.5,   # yanked early
                "starts_sampled": 5,
                "season_xba": 0.275,       # below-avg
            },
            "tto_splits": {
                "xba": {1: 0.270, 2: 0.290, 3: 0.310},
                "pa":  {1: 160, 2: 140, 3: 90},  # 3rd TTO below min sample
            },
            "bullpen_profile": {
                "xba_vs_r": 0.215, "xba_vs_l": 0.220,
                "pa_vs_r": 600, "pa_vs_l": 500,
            },
            "umpire_k_dev": 0.02,  # +2pp K above league
            "venue": "Petco Park",
        },
        {
            "name": "Rookie starter (no TTO data), avg bullpen, hitter ump, Coors",
            "starter_workload": {
                "avg_ip_per_start": 5.3,
                "starts_sampled": 3,
                "season_xba": 0.260,
            },
            "tto_splits": {
                "xba": {1: None, 2: None, 3: None},
                "pa":  {1: 40, 2: 30, 3: 15},   # all below min
            },
            "bullpen_profile": {
                "xba_vs_r": 0.244, "xba_vs_l": 0.248,
                "pa_vs_r": 500, "pa_vs_l": 500,
            },
            "umpire_k_dev": -0.015,
            "venue": "Coors Field",
        },
    ]

    arsenal = {"FF": 0.45, "SL": 0.30, "CH": 0.15, "CU": 0.10}
    slot = 3
    from hit_ledger.config import PA_BY_LINEUP_SLOT
    total_pa = PA_BY_LINEUP_SLOT[slot]

    for scen in scenarios:
        print("\n" + "=" * 70)
        print(scen["name"])
        print("=" * 70)

        mp = build_matchup_v2(
            batter_id=12345,
            batter_df=batter_df,
            starter_id=67890,
            starter_throws="R",
            starter_arsenal=arsenal,
            starter_workload=scen["starter_workload"],
            tto_splits=scen["tto_splits"],
            bullpen_profile=scen["bullpen_profile"],
            batter_stands="R",
            lineup_slot=slot,
            total_pa=total_pa,
            venue=scen["venue"],
            umpire_k_dev=scen["umpire_k_dev"],
            as_of=date(2025, 6, 15),
        )

        print(f"Expected PA vs starter: {mp.expected_pa_vs_starter:.2f}")
        print(f"Expected PA vs bullpen: {mp.expected_pa_vs_bullpen:.2f}")
        print(f"TTO penalties: {mp.tto_penalties}")
        print(f"Umpire xBA adj: {mp.umpire_adjustment:+.4f}")
        print(f"Bullpen xBA used: {mp.bullpen_xba:.3f}")
        print(f"\nPer-PA probabilities:")
        for i, pa in enumerate(mp.pa_probs, 1):
            print(f"  PA {i}: {pa.source:18s} P(hit)={pa.p_hit:.3f}  "
                  f"P(HR)={pa.p_hr:.3f}  "
                  f"[1B={pa.p_1b:.3f}, 2B={pa.p_2b:.3f}, 3B={pa.p_3b:.3f}]")

        # Simulate
        results = simulate_v2([mp], {12345: slot}, n_sims=10_000)
        r = results[0]
        print(f"\nSim (10k games, slot 3, {total_pa} total PAs):")
        print(f"  P(1+ hit)   = {r.p_1_hit:.3f}")
        print(f"  P(2+ hits)  = {r.p_2_hits:.3f}")
        print(f"  P(HR)       = {r.p_1_hr:.3f}")
        print(f"  E[hits]     = {r.expected_hits:.2f}")
        print(f"  E[TB]       = {r.expected_tb:.2f}")


if __name__ == "__main__":
    main()
