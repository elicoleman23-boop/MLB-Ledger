"""Backtest harness — replays the live model on historical slates with
a hard no-leakage invariant.

The package is a read-only consumer of hit_ledger.data, hit_ledger.sim,
and hit_ledger.config. It ships as a separate package (and a separate
SQLite DB) so backtest state never pollutes live caches.
"""
from hit_ledger.backtest.config import BacktestConfig, BacktestResult
from hit_ledger.backtest.leakage_check import LeakageError
from hit_ledger.backtest.runner import run_backtest

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "LeakageError",
    "run_backtest",
]
