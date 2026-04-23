"""Backtest run configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


@dataclass
class BacktestConfig:
    """Parameters for a single backtest run.

    The safety-critical defaults live here: umpire adjustment is never
    applied in backtest (explicit 0.0), and the per-reliever bullpen
    model is disabled because its data layer isn't date-scoped yet.
    Both decisions err on the side of a cleaner leakage story at the
    cost of some predictive signal; the run summary surfaces this.
    """

    start_date: date
    end_date: date
    teams: list[str] | None = None
    # Kept for API symmetry with the live pipeline, but must be False in
    # backtest. relievers.fetch_team_bullpen_roster isn't as_of-aware
    # yet — a follow-up needs to plumb as_of through before per-reliever
    # backtesting is leak-safe.
    use_per_reliever_bullpen: bool = False
    cache_dir: Path = field(default_factory=lambda: Path("backtest_cache"))
    n_sims: int = 10_000
    rng_seed: int = 42
    # Cap games per day for fast smoke runs; None = all games.
    max_games_per_day: int | None = None
    verbose: bool = True
    # When True, re-run dates even if backtest.db already has
    # predictions for them. Default False → per-date resumption.
    force: bool = False

    def __post_init__(self):
        if self.start_date > self.end_date:
            raise ValueError(
                f"start_date {self.start_date} is after end_date {self.end_date}"
            )
        if self.use_per_reliever_bullpen:
            raise ValueError(
                "use_per_reliever_bullpen=True is not leak-safe in backtest mode: "
                "relievers.fetch_team_bullpen_roster does not accept an as_of "
                "parameter. Leave this False until the data layer is updated."
            )
        if self.n_sims < 100:
            raise ValueError(f"n_sims must be ≥ 100 for MC stability; got {self.n_sims}")


@dataclass
class BacktestResult:
    """Return value from run_backtest; consumable for analysis."""
    run_id: str
    n_games: int
    n_predictions: int
    n_skipped: int
    elapsed_seconds: float
    predictions_df: "object"   # pd.DataFrame — typed as object to avoid pandas import here
    actuals_df: "object"       # pd.DataFrame
