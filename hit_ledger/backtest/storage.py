"""SQLite persistence for backtest predictions and actuals.

Uses a SEPARATE database from hit_ledger.db (which stores the live
prediction cache + Statcast profile blobs). The split prevents a long
backtest run from pushing live cache entries out of their tables, and
keeps the analysis dataset reproducible independent of day-to-day
live-app state.

The schema follows the backtest spec — three tables:
  * backtest_runs:        one row per run_id (config + metadata)
  * batter_predictions:   one row per (run_id, game_pk, batter_id)
  * batter_actuals:       one row per (run_id, game_pk, batter_id)
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from hit_ledger.backtest.config import BacktestConfig


SCHEMA = """
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id          TEXT PRIMARY KEY,
    start_date      TEXT NOT NULL,
    end_date        TEXT NOT NULL,
    config_json     TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    n_games         INTEGER,
    n_batter_predictions INTEGER,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS batter_predictions (
    run_id          TEXT NOT NULL,
    game_pk         INTEGER NOT NULL,
    batter_id       INTEGER NOT NULL,
    game_date       TEXT NOT NULL,
    team            TEXT,
    opp_team        TEXT,
    starter_id      INTEGER,
    lineup_slot     INTEGER,
    venue           TEXT,
    umpire_k_dev    REAL,
    pred_p_1_hit    REAL,
    pred_p_2_hits   REAL,
    pred_p_1_hr     REAL,
    pred_expected_hits REAL,
    pred_expected_tb REAL,
    pred_p_tb_over_1_5 REAL,
    pred_p_tb_over_2_5 REAL,
    batter_profile_n_pa INTEGER,
    pitcher_profile_n_pa INTEGER,
    data_quality    TEXT,
    expected_pa_vs_starter REAL,
    expected_pa_vs_bullpen REAL,
    batter_stands   TEXT,
    pitcher_throws  TEXT,
    starter_arsenal_json TEXT,
    PRIMARY KEY (run_id, game_pk, batter_id)
);

CREATE TABLE IF NOT EXISTS batter_actuals (
    run_id          TEXT NOT NULL,
    game_pk         INTEGER NOT NULL,
    batter_id       INTEGER NOT NULL,
    pa_count        INTEGER,
    hits            INTEGER,
    hrs             INTEGER,
    total_bases     INTEGER,
    singles         INTEGER,
    doubles         INTEGER,
    triples         INTEGER,
    strikeouts      INTEGER,
    walks           INTEGER,
    hbp             INTEGER,
    actual_lineup_slot INTEGER,
    played          INTEGER,
    pa_sequence_json TEXT,
    PRIMARY KEY (run_id, game_pk, batter_id)
);

-- Used by the runner's per-date resumption check; indexed on
-- (run_id, game_date) so `SELECT 1 WHERE` lookups stay O(log n).
CREATE INDEX IF NOT EXISTS idx_predictions_run_date
    ON batter_predictions (run_id, game_date);
"""


# ---------------------------------------------------------------------------
# DB setup + connection
# ---------------------------------------------------------------------------
def init_db(db_path: Path) -> None:
    """Create the backtest DB at `db_path` if it doesn't exist. Safe to
    call repeatedly; `CREATE TABLE IF NOT EXISTS` is idempotent."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def create_run(
    db_path: Path,
    config: BacktestConfig,
    notes: str | None = None,
) -> str:
    """Insert a new `backtest_runs` row and return its `run_id`.

    The config is serialized to JSON (Path and set objects become
    strings/lists) so every run is self-describing without depending
    on the code that produced it.
    """
    init_db(db_path)
    run_id = f"bt_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO backtest_runs
            (run_id, start_date, end_date, config_json, created_at,
             n_games, n_batter_predictions, notes)
            VALUES (?, ?, ?, ?, ?, 0, 0, ?)
            """,
            (
                run_id,
                config.start_date.isoformat(),
                config.end_date.isoformat(),
                json.dumps(_config_to_jsonable(config)),
                datetime.utcnow().isoformat(),
                notes,
            ),
        )
    return run_id


def finalize_run(
    db_path: Path,
    run_id: str,
    n_games: int,
    n_batter_predictions: int,
) -> None:
    """Update totals on the run row after the orchestrator loop finishes."""
    with _connect(db_path) as conn:
        conn.execute(
            """UPDATE backtest_runs
               SET n_games = ?, n_batter_predictions = ?
               WHERE run_id = ?""",
            (n_games, n_batter_predictions, run_id),
        )


def save_predictions(
    db_path: Path,
    run_id: str,
    predictions: list[dict[str, Any]],
) -> None:
    """Append prediction rows for this run. Uses INSERT OR REPLACE so a
    second save for the same `(run_id, game_pk, batter_id)` overwrites,
    which matters for the `--force` re-run path."""
    if not predictions:
        return
    with _connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO batter_predictions (
              run_id, game_pk, batter_id, game_date, team, opp_team,
              starter_id, lineup_slot, venue, umpire_k_dev,
              pred_p_1_hit, pred_p_2_hits, pred_p_1_hr,
              pred_expected_hits, pred_expected_tb,
              pred_p_tb_over_1_5, pred_p_tb_over_2_5,
              batter_profile_n_pa, pitcher_profile_n_pa, data_quality,
              expected_pa_vs_starter, expected_pa_vs_bullpen,
              batter_stands, pitcher_throws, starter_arsenal_json
            ) VALUES (
              ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
              ?, ?, ?, ?, ?, ?, ?,
              ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                (
                    run_id,
                    int(p["game_pk"]),
                    int(p["batter_id"]),
                    _as_iso(p.get("game_date")),
                    p.get("team"),
                    p.get("opp_team"),
                    p.get("starter_id"),
                    p.get("lineup_slot"),
                    p.get("venue"),
                    p.get("umpire_k_dev"),
                    p.get("pred_p_1_hit"),
                    p.get("pred_p_2_hits"),
                    p.get("pred_p_1_hr"),
                    p.get("pred_expected_hits"),
                    p.get("pred_expected_tb"),
                    p.get("pred_p_tb_over_1_5"),
                    p.get("pred_p_tb_over_2_5"),
                    p.get("batter_profile_n_pa"),
                    p.get("pitcher_profile_n_pa"),
                    p.get("data_quality"),
                    p.get("expected_pa_vs_starter"),
                    p.get("expected_pa_vs_bullpen"),
                    p.get("batter_stands"),
                    p.get("pitcher_throws"),
                    json.dumps(p.get("starter_arsenal_summary") or {}),
                )
                for p in predictions
            ],
        )


def save_actuals(
    db_path: Path,
    run_id: str,
    actuals: list[dict[str, Any]],
) -> None:
    """Append actuals rows. Same INSERT OR REPLACE semantics."""
    if not actuals:
        return
    with _connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO batter_actuals (
              run_id, game_pk, batter_id,
              pa_count, hits, hrs, total_bases,
              singles, doubles, triples,
              strikeouts, walks, hbp,
              actual_lineup_slot, played, pa_sequence_json
            ) VALUES (
              ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                (
                    run_id,
                    int(a["game_pk"]),
                    int(a["batter_id"]),
                    a.get("pa_count", 0),
                    a.get("hits", 0),
                    a.get("hrs", 0),
                    a.get("total_bases", 0),
                    a.get("singles", 0),
                    a.get("doubles", 0),
                    a.get("triples", 0),
                    a.get("strikeouts", 0),
                    a.get("walks", 0),
                    a.get("hbp", 0),
                    a.get("lineup_slot"),
                    1 if a.get("played") else 0,
                    json.dumps(a.get("pa_sequence") or []),
                )
                for a in actuals
            ],
        )


# ---------------------------------------------------------------------------
# Resumption helper
# ---------------------------------------------------------------------------
def dates_already_completed(
    db_path: Path,
    run_id: str,
    dates: list[date],
) -> set[date]:
    """Return the subset of `dates` that already have ≥1 prediction row
    stored under `run_id`. Used by the orchestrator's resumption
    check — a previously-started run with some dates done will skip
    those dates on resume unless `force` is on."""
    if not dates:
        return set()
    init_db(db_path)
    done: set[date] = set()
    with _connect(db_path) as conn:
        for d in dates:
            row = conn.execute(
                "SELECT 1 FROM batter_predictions WHERE run_id = ? AND game_date = ? LIMIT 1",
                (run_id, d.isoformat()),
            ).fetchone()
            if row is not None:
                done.add(d)
    return done


def delete_date_rows(db_path: Path, run_id: str, game_date: date) -> None:
    """Remove all prediction and actual rows for a specific (run_id, date).
    Called before force-rerunning a date to avoid stale rows lingering."""
    with _connect(db_path) as conn:
        conn.execute(
            "DELETE FROM batter_predictions WHERE run_id = ? AND game_date = ?",
            (run_id, game_date.isoformat()),
        )
        # actuals don't carry game_date directly — join via predictions
        # (we deleted predictions above, so just match game_pks)
        conn.execute(
            """DELETE FROM batter_actuals
               WHERE run_id = ? AND game_pk NOT IN (
                 SELECT game_pk FROM batter_predictions WHERE run_id = ?
               )""",
            (run_id, run_id),
        )


# ---------------------------------------------------------------------------
# Read-back helpers for analysis
# ---------------------------------------------------------------------------
def load_run(db_path: Path, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return `(predictions_df, actuals_df)` for a run."""
    init_db(db_path)
    with _connect(db_path) as conn:
        preds = pd.read_sql_query(
            "SELECT * FROM batter_predictions WHERE run_id = ?",
            conn,
            params=(run_id,),
        )
        actuals = pd.read_sql_query(
            "SELECT * FROM batter_actuals WHERE run_id = ?",
            conn,
            params=(run_id,),
        )
    return preds, actuals


def list_runs(db_path: Path) -> pd.DataFrame:
    """Summary of every run in the DB — for the analysis-phase notebooks."""
    init_db(db_path)
    with _connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM backtest_runs ORDER BY created_at DESC", conn)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------
def _config_to_jsonable(config: BacktestConfig) -> dict:
    """Dataclass → JSON-safe dict (Path, date, set → primitives)."""
    d = asdict(config)
    d["start_date"] = config.start_date.isoformat()
    d["end_date"] = config.end_date.isoformat()
    d["cache_dir"] = str(config.cache_dir)
    return d


def _as_iso(v: Any) -> Any:
    if isinstance(v, date):
        return v.isoformat()
    return v
