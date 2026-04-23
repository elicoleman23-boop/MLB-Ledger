"""Top-level backtest orchestrator.

Walks a date range, replays each slate, persists predictions + actuals,
and prints a final summary block you can paste back for review. The
loop is re-entrant: if an earlier run populated some dates under the
same `run_id`, those dates are skipped with an explicit [RESUME]
message unless `config.force` is True.

Leakage errors are unrecoverable. Anything else (missing lineups,
profile scrape failures, game crashes) is tracked as a skip and
grouped by reason in the final summary.
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from hit_ledger.backtest.config import BacktestConfig, BacktestResult
from hit_ledger.backtest.leakage_check import LeakageError
from hit_ledger.backtest.outcomes import extract_batter_outcomes
from hit_ledger.backtest.replay import replay_slate
from hit_ledger.backtest.storage import (
    create_run,
    dates_already_completed,
    delete_date_rows,
    finalize_run,
    load_run,
    save_actuals,
    save_predictions,
)

logger = logging.getLogger(__name__)


def run_backtest(config: BacktestConfig) -> BacktestResult:
    """Execute a full backtest per `config` and return a BacktestResult.

    The heavy loop is intentionally sequential — pybaseball is
    rate-limited and parallel fetching causes throttling errors that
    cost more than they save.
    """
    db_path = Path(config.cache_dir) / "backtest.db"
    run_id = create_run(db_path, config)

    all_dates = _dates_between(config.start_date, config.end_date)

    # Per-date resumption: which dates already have predictions?
    done_dates = set() if config.force else dates_already_completed(db_path, run_id, all_dates)

    t_start = time.perf_counter()
    total_games = 0
    total_predictions = 0
    total_skipped = 0
    skip_reason_counter: Counter[str] = Counter()
    warnings_emitted: list[str] = []

    for d in all_dates:
        if d in done_dates:
            _log_resume(d, db_path, run_id, config)
            continue
        if config.force:
            # Wipe any previous rows for this date before re-running so
            # the `INSERT OR REPLACE` path stays consistent with --force.
            delete_date_rows(db_path, run_id, d)
            _log(config, f"[FORCE] {d.isoformat()}: re-running (existing rows deleted)")
        else:
            _log(config, f"[RUN]   {d.isoformat()}: fetching slate")

        try:
            predictions, skipped, n_games = replay_slate(d, config)
        except LeakageError as exc:
            logger.error("leakage_abort date=%s: %s", d, exc)
            raise

        # Extract actuals for every batter we produced a prediction for.
        actuals: list[dict] = []
        for pred in predictions:
            actual = extract_batter_outcomes(
                game_pk=int(pred["game_pk"]),
                batter_id=int(pred["batter_id"]),
            )
            actual["game_pk"] = pred["game_pk"]
            actual["batter_id"] = pred["batter_id"]
            actuals.append(actual)

        save_predictions(db_path, run_id, predictions)
        save_actuals(db_path, run_id, actuals)

        total_games += n_games
        total_predictions += len(predictions)
        total_skipped += len(skipped)
        for entry in skipped:
            skip_reason_counter[entry["reason"]] += 1

        _log(
            config,
            f"        {d.isoformat()}: {n_games} games, "
            f"{len(predictions)} batter predictions, {len(skipped)} skipped",
        )

    elapsed = time.perf_counter() - t_start
    finalize_run(
        db_path,
        run_id,
        n_games=total_games,
        n_batter_predictions=total_predictions,
    )

    # Load everything back into DataFrames for the caller.
    preds_df, actuals_df = load_run(db_path, run_id)

    _print_final_summary(
        config=config,
        run_id=run_id,
        elapsed_seconds=elapsed,
        total_games=total_games,
        total_predictions=total_predictions,
        total_skipped=total_skipped,
        skip_reason_counter=skip_reason_counter,
        warnings_emitted=warnings_emitted,
        preds_df=preds_df,
        actuals_df=actuals_df,
    )

    return BacktestResult(
        run_id=run_id,
        n_games=total_games,
        n_predictions=total_predictions,
        n_skipped=total_skipped,
        elapsed_seconds=elapsed,
        predictions_df=preds_df,
        actuals_df=actuals_df,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _dates_between(start: date, end: date) -> list[date]:
    out = []
    d = start
    while d <= end:
        out.append(d)
        d = d + timedelta(days=1)
    return out


def _log(config: BacktestConfig, msg: str) -> None:
    if config.verbose:
        print(msg)


def _log_resume(
    d: date, db_path: Path, run_id: str, config: BacktestConfig
) -> None:
    # Look up how many games the earlier run scored for this date, so
    # the resume line mirrors the spec's example message.
    from hit_ledger.backtest.storage import _connect  # re-use the single contextmanager
    with _connect(db_path) as conn:
        row = conn.execute(
            """SELECT COUNT(DISTINCT game_pk) AS n
               FROM batter_predictions
               WHERE run_id = ? AND game_date = ?""",
            (run_id, d.isoformat()),
        ).fetchone()
    n_games_done = row["n"] if row else 0
    _log(
        config,
        f"[RESUME] {d.isoformat()}: {n_games_done} games already in DB, skipping",
    )


def _print_final_summary(
    *,
    config: BacktestConfig,
    run_id: str,
    elapsed_seconds: float,
    total_games: int,
    total_predictions: int,
    total_skipped: int,
    skip_reason_counter: Counter[str],
    warnings_emitted: list[str],
    preds_df,
    actuals_df,
) -> None:
    """Print the end-of-run block requested in the spec — totals,
    grouped skip reasons, configuration flags, and a three-row
    prediction/actual eyeball sample."""
    bar = "=" * 70
    _log(config, "\n" + bar)
    _log(config, "BACKTEST SUMMARY")
    _log(config, bar)
    _log(config, f"  run_id            : {run_id}")
    _log(config, f"  date range        : {config.start_date} → {config.end_date}")
    _log(config, f"  total games       : {total_games}")
    _log(config, f"  total batter preds: {total_predictions}")
    _log(config, f"  skipped batters   : {total_skipped}")
    _log(config, f"  elapsed seconds   : {elapsed_seconds:.1f}")

    if skip_reason_counter:
        _log(config, "\n  skip reasons (grouped):")
        for reason, count in skip_reason_counter.most_common():
            _log(config, f"    {count:5d}  {reason}")
    else:
        _log(config, "\n  skip reasons (grouped): (none)")

    if warnings_emitted:
        _log(config, "\n  warnings:")
        for w in warnings_emitted:
            _log(config, f"    - {w}")

    # Configuration flags that materially affect interpretation
    _log(config, "\n  Umpire adjustment: disabled in backtest for leakage safety")
    _log(config, "  Bullpen mode    : team-level (per-reliever disabled pending")
    _log(config, "                    relievers.py as_of plumbing)")

    # Eyeball sample: first three joined prediction+actual rows
    if preds_df is not None and not preds_df.empty and actuals_df is not None and not actuals_df.empty:
        merged = preds_df.merge(
            actuals_df,
            on=["run_id", "game_pk", "batter_id"],
            how="inner",
            suffixes=("_pred", "_act"),
        )
        if not merged.empty:
            _log(config, "\n  First 3 prediction+actual rows (eyeball check):")
            cols = [
                "game_date", "batter_id", "team", "opp_team",
                "pred_p_1_hit", "pred_p_1_hr", "pred_expected_hits",
                "hits", "hrs", "total_bases", "played",
            ]
            available = [c for c in cols if c in merged.columns]
            for i, row in merged.head(3).iterrows():
                parts = [f"{c}={row[c]}" for c in available]
                _log(config, f"    · {' · '.join(parts)}")
    _log(config, bar + "\n")
