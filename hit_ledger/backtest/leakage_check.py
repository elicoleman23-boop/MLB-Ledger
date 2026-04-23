"""Empirical leakage guard for the backtest harness.

The live data layer's `as_of` plumbing should prevent future data from
leaking into pregame profiles, but that's trust-based. These helpers
inspect the actual DataFrames we're about to feed to the matchup
builder and raise immediately if any row's `game_date` is not strictly
before the target date.

Catching leakage empirically guards against subtle bugs in the fetchers
(timezone drift, typing issues, refactor regressions) that would
otherwise produce a backtest that reads cleaner than the model is.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd


class LeakageError(RuntimeError):
    """Raised when any pregame profile contains data from `game_date`
    or later. Carries enough context for a clean error message and
    actionable log line."""


def validate_no_leakage(
    game_date: date,
    profiles: dict[str, Any],
) -> list[str]:
    """Return a list of violation strings (empty = clean).

    `profiles` is a dict of name → pd.DataFrame. For each DataFrame with
    a `game_date` column, we check that every row strictly precedes the
    target `game_date`. Keys that map to None or to DataFrames without
    a `game_date` column are skipped silently — the caller is
    responsible for passing only DataFrames with a date column.
    """
    violations: list[str] = []
    target = pd.Timestamp(game_date)
    for name, df in profiles.items():
        if df is None:
            continue
        if not isinstance(df, pd.DataFrame):
            continue
        if df.empty:
            continue
        if "game_date" not in df.columns:
            continue
        # Coerce to datetime for a defensive comparison; if some rows are
        # non-datelike (typing bugs) treat those as violations.
        gd = pd.to_datetime(df["game_date"], errors="coerce")
        if gd.isna().any():
            n_bad = int(gd.isna().sum())
            violations.append(
                f"profile={name}: {n_bad} row(s) have unparseable game_date"
            )
        max_date = gd.max()
        if pd.isna(max_date):
            continue
        if max_date >= target:
            violations.append(
                f"profile={name}: max game_date {max_date.date()} is "
                f"not strictly before target {target.date()} "
                f"(n_leaked_rows={int((gd >= target).sum())})"
            )
    return violations


def assert_no_leakage(game_date: date, profiles: dict[str, Any]) -> None:
    """Raise LeakageError on any violation — no soft mode, no warn-only.

    The backtest config deliberately has no override flag for this
    check. Leakage isn't a debugging nuisance; it silently flatters the
    model's backtest score. Better to hard-fail an entire slate than to
    produce cooked numbers.
    """
    violations = validate_no_leakage(game_date, profiles)
    if not violations:
        return
    preview = "; ".join(violations[:3])
    suffix = "" if len(violations) <= 3 else f" (+{len(violations) - 3} more)"
    raise LeakageError(
        f"leakage detected targeting game_date={game_date.isoformat()}: "
        f"{preview}{suffix}"
    )
