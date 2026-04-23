"""Fix E Phase 1 validation — pitch-sim profile builders.

Builds a synthetic pitch-level Statcast frame (with the `zone`,
`description`, `balls`, `strikes`, `launch_speed`, `launch_angle`
columns the builders need) for both a batter and a pitcher, runs each
builder, asserts the output schema and sanity bounds, and confirms
that empty/sparse DataFrames produce a pure-prior profile instead of
crashing.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

for mod_name in ["pybaseball", "statsapi", "streamlit", "requests"]:
    stub = types.ModuleType(mod_name)
    if mod_name == "pybaseball":
        stub.statcast_batter = lambda **kw: None
        stub.statcast_pitcher = lambda **kw: None
        stub.statcast = lambda **kw: None
    sys.modules[mod_name] = stub

import numpy as np
import pandas as pd

from hit_ledger.config import (
    LEAGUE_EV_LA,
    LEAGUE_O_CONTACT_BY_PITCH,
    LEAGUE_O_SWING_BY_PITCH,
    LEAGUE_Z_CONTACT_BY_PITCH,
    LEAGUE_Z_SWING_BY_PITCH,
    MIN_EV_LA_SAMPLE,
)
from hit_ledger.sim.pitch_sim import (
    build_batter_pitch_profile,
    build_pitcher_pitch_profile,
)

PITCH_TYPES = ("FF", "SI", "SL", "CU", "CH", "FC")
BUCKETS = ("ahead", "even", "behind", "must_strike")


def synth_pitch_level_df(
    n: int = 3000,
    is_pitcher: bool = False,
    seed: int = 17,
) -> pd.DataFrame:
    """Synthetic pitch-level Statcast frame with columns the profile
    builders actually consume."""
    rng = np.random.default_rng(seed)
    pitch_types = rng.choice(PITCH_TYPES, size=n, p=[0.38, 0.15, 0.22, 0.08, 0.12, 0.05])
    # 1-9 in zone, 11-14 out (~55/45 split)
    zones = rng.choice(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14],
        size=n,
        p=[0.06] * 9 + [0.115] * 4,
    )
    in_zone = zones <= 9

    # Swing propensity: higher in zone
    swing_prob = np.where(in_zone, 0.68, 0.28)
    swung = rng.random(n) < swing_prob
    # Conditional on swing: ~20% whiff, rest contact
    whiff = swung & (rng.random(n) < 0.20)

    # Build description column matching Statcast vocabulary
    descriptions = np.where(
        ~swung,
        np.where(in_zone, "called_strike", "ball"),
        np.where(whiff, "swinging_strike", "hit_into_play"),
    )

    # Counts: distribute realistically
    balls = rng.integers(0, 4, size=n)   # 0-3
    strikes = rng.integers(0, 3, size=n) # 0-2

    # Contact → launch_speed / launch_angle
    contact_mask = swung & ~whiff
    launch_speed = np.where(contact_mask, rng.normal(89.0, 13.0, size=n), np.nan)
    launch_angle = np.where(contact_mask, rng.normal(13.0, 27.0, size=n), np.nan)

    # Events only on PA-ending rows (small fraction)
    pa_ending = rng.random(n) < 0.25
    events = np.where(
        pa_ending & swung & ~whiff,
        np.where(rng.random(n) < 0.32, "single", "field_out"),
        "",
    )
    events = np.where(pa_ending & swung & whiff, "strikeout", events)
    events = np.where(pa_ending & ~swung & ~in_zone, "walk", events)

    df = pd.DataFrame({
        "game_date": pd.date_range("2024-04-01", periods=n, freq="1h"),
        "pitch_type": pitch_types,
        "p_throws": "R",
        "stand": "R",
        "zone": zones,
        "balls": balls,
        "strikes": strikes,
        "description": descriptions,
        "events": events,
        "estimated_ba_using_speedangle": np.where(contact_mask, rng.beta(2, 5, size=n), np.nan),
        "launch_speed": launch_speed,
        "launch_angle": launch_angle,
    })
    # Differentiate column expected by each builder
    if is_pitcher:
        df["batter"] = rng.integers(100000, 999999, size=n)
    else:
        df["pitcher"] = rng.integers(100000, 999999, size=n)
    return df


def _assert_rate(x: float, low: float = 0.0, high: float = 1.0, name: str = ""):
    assert low <= x <= high, f"{name} out of bounds: {x:.4f}"


def test_batter_profile_schema_and_sanity():
    df = synth_pitch_level_df(n=4000, is_pitcher=False, seed=13)
    prof = build_batter_pitch_profile(df)

    # Schema
    for key in ("z_swing_rate", "o_swing_rate", "z_contact_rate",
                "o_contact_rate", "ev_la_by_pitch", "overall_batter_handedness"):
        assert key in prof, f"missing key: {key}"

    # Every pitch type in league tables must be present
    for pt in LEAGUE_Z_SWING_BY_PITCH:
        assert pt in prof["z_swing_rate"]
        assert pt in prof["o_swing_rate"]
        assert pt in prof["z_contact_rate"]
        assert pt in prof["o_contact_rate"]
        assert pt in prof["ev_la_by_pitch"]
        _assert_rate(prof["z_swing_rate"][pt], name=f"z_swing[{pt}]")
        _assert_rate(prof["o_swing_rate"][pt], name=f"o_swing[{pt}]")
        _assert_rate(prof["z_contact_rate"][pt], name=f"z_contact[{pt}]")
        _assert_rate(prof["o_contact_rate"][pt], name=f"o_contact[{pt}]")
        ev_block = prof["ev_la_by_pitch"][pt]
        for sub in ("mean_ev", "sd_ev", "mean_la", "sd_la", "corr_ev_la", "n"):
            assert sub in ev_block, f"missing ev_la subkey {sub} for {pt}"

    # Handedness should be 'R' (synth data is all R)
    assert prof["overall_batter_handedness"] == "R"

    # Directional sanity: z_swing > o_swing (hitters chase less)
    # Average across observed pitch types to smooth MC noise
    zs = np.mean([prof["z_swing_rate"][pt] for pt in PITCH_TYPES])
    os_ = np.mean([prof["o_swing_rate"][pt] for pt in PITCH_TYPES])
    assert zs > os_, f"expected z_swing ({zs:.3f}) > o_swing ({os_:.3f})"


def test_pitcher_profile_schema_and_sanity():
    df = synth_pitch_level_df(n=6000, is_pitcher=True, seed=29)
    arsenal = {"FF": 0.45, "SL": 0.25, "CH": 0.15, "CU": 0.10, "FC": 0.05}
    prof = build_pitcher_pitch_profile(df, arsenal)

    for key in ("arsenal", "zone_rate", "whiff_rate", "ev_suppression",
                "la_influence", "pitcher_handedness"):
        assert key in prof

    assert prof["arsenal"] == arsenal
    assert prof["pitcher_handedness"] == "R"

    for pt in LEAGUE_Z_SWING_BY_PITCH:
        assert pt in prof["zone_rate"]
        assert pt in prof["whiff_rate"]
        assert pt in prof["ev_suppression"]
        assert pt in prof["la_influence"]
        # Each zone_rate must have all 4 buckets
        for b in BUCKETS:
            assert b in prof["zone_rate"][pt], f"bucket {b} missing for {pt}"
            _assert_rate(prof["zone_rate"][pt][b], name=f"zone_rate[{pt}][{b}]")
        # Whiff rates between 0 and 1
        _assert_rate(prof["whiff_rate"][pt]["z"], name=f"whiff_z[{pt}]")
        _assert_rate(prof["whiff_rate"][pt]["o"], name=f"whiff_o[{pt}]")


def test_empty_batter_falls_back_to_priors():
    empty = pd.DataFrame()
    prof = build_batter_pitch_profile(empty)
    # Empty DF → pure league priors, no crash
    for pt in LEAGUE_Z_SWING_BY_PITCH:
        assert prof["z_swing_rate"][pt] == LEAGUE_Z_SWING_BY_PITCH[pt]
        assert prof["o_swing_rate"][pt] == LEAGUE_O_SWING_BY_PITCH[pt]
        assert prof["z_contact_rate"][pt] == LEAGUE_Z_CONTACT_BY_PITCH[pt]
        assert prof["o_contact_rate"][pt] == LEAGUE_O_CONTACT_BY_PITCH[pt]
        ev = prof["ev_la_by_pitch"][pt]
        assert ev["mean_ev"] == LEAGUE_EV_LA["mean_ev"]
        assert ev["mean_la"] == LEAGUE_EV_LA["mean_la"]
        assert ev["n"] == 0


def test_empty_pitcher_falls_back_to_priors():
    empty = pd.DataFrame()
    prof = build_pitcher_pitch_profile(empty, {"FF": 1.0})
    assert prof["arsenal"] == {"FF": 1.0}
    for pt in LEAGUE_Z_SWING_BY_PITCH:
        assert pt in prof["zone_rate"]
        for b in BUCKETS:
            assert b in prof["zone_rate"][pt]
        assert prof["ev_suppression"][pt] == 0.0
        assert prof["la_influence"][pt] == 0.0


def _print_profile(label: str, prof: dict):
    print(f"\n--- {label} ---")
    for top, value in prof.items():
        if isinstance(value, dict):
            print(f"  {top}:")
            preview = list(value.items())[:4]
            for k, v in preview:
                print(f"    {k}: {v}")
            if len(value) > 4:
                print(f"    ... ({len(value) - 4} more)")
        else:
            print(f"  {top}: {value}")


def main():
    print("=" * 70)
    print("Fix E Phase 1 — pitch-sim profile builders")
    print("=" * 70)

    test_batter_profile_schema_and_sanity()
    print("batter profile: schema + sanity OK")
    test_pitcher_profile_schema_and_sanity()
    print("pitcher profile: schema + sanity OK")
    test_empty_batter_falls_back_to_priors()
    print("empty batter: priors OK")
    test_empty_pitcher_falls_back_to_priors()
    print("empty pitcher: priors OK")

    # Visual inspection: one of each
    df_b = synth_pitch_level_df(n=4000, is_pitcher=False, seed=13)
    _print_profile("Batter profile (synth, n=4000)", build_batter_pitch_profile(df_b))
    df_p = synth_pitch_level_df(n=6000, is_pitcher=True, seed=29)
    _print_profile(
        "Pitcher profile (synth, n=6000)",
        build_pitcher_pitch_profile(df_p, {"FF": 0.45, "SL": 0.25, "CH": 0.15, "CU": 0.10, "FC": 0.05}),
    )

    print("\nAll profile-builder validations passed.")


if __name__ == "__main__":
    main()
