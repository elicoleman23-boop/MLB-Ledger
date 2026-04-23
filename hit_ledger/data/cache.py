"""
SQLite-backed cache for daily data snapshots.

Statcast scrapes are slow and hit external servers — this layer stores
the day's inputs so re-runs are instant and we keep a historical record
for backtesting and ROI tracking.

Tables:
    games           — one row per scheduled game
    lineups         — one row per batter per game
    batter_profiles — pickled Statcast DataFrames per batter per day
    pitcher_arsenals— pitch-mix JSON per pitcher per day
    predictions     — model output per batter per day
"""
from __future__ import annotations

import json
import pickle
import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from hit_ledger.config import DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_date   TEXT NOT NULL,
    game_pk     INTEGER NOT NULL,
    home_team   TEXT,
    away_team   TEXT,
    venue       TEXT,
    game_time   TEXT,
    home_pitcher_id INTEGER,
    away_pitcher_id INTEGER,
    umpire_name TEXT,
    PRIMARY KEY (game_date, game_pk)
);

CREATE TABLE IF NOT EXISTS lineups (
    game_date   TEXT NOT NULL,
    game_pk     INTEGER NOT NULL,
    team        TEXT NOT NULL,
    batter_id   INTEGER NOT NULL,
    batter_name TEXT,
    lineup_slot INTEGER,
    bats        TEXT,
    PRIMARY KEY (game_date, game_pk, batter_id)
);

CREATE TABLE IF NOT EXISTS batter_profiles (
    game_date   TEXT NOT NULL,
    batter_id   INTEGER NOT NULL,
    profile_blob BLOB NOT NULL,
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (game_date, batter_id)
);

CREATE TABLE IF NOT EXISTS pitcher_profiles (
    game_date   TEXT NOT NULL,
    pitcher_id  INTEGER NOT NULL,
    profile_blob BLOB NOT NULL,
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (game_date, pitcher_id)
);

CREATE TABLE IF NOT EXISTS pitcher_arsenals (
    game_date    TEXT NOT NULL,
    pitcher_id   INTEGER NOT NULL,
    throws       TEXT,
    arsenal_json TEXT NOT NULL,
    fetched_at   TEXT NOT NULL,
    PRIMARY KEY (game_date, pitcher_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    game_date        TEXT NOT NULL,
    game_pk          INTEGER NOT NULL,
    batter_id        INTEGER NOT NULL,
    p_1_hit          REAL,
    p_2_hits         REAL,
    p_1_hr           REAL,
    p_tb_over_1_5    REAL,
    p_tb_over_2_5    REAL,
    expected_hits    REAL,
    expected_tb      REAL,
    computed_at      TEXT NOT NULL,
    PRIMARY KEY (game_date, game_pk, batter_id)
);

-- ---------------------------------------------------------------------
-- v2 tables
-- ---------------------------------------------------------------------

-- Team bullpen xBA-against by handedness; refreshed daily
CREATE TABLE IF NOT EXISTS bullpen_profiles (
    game_date   TEXT NOT NULL,
    team        TEXT NOT NULL,
    xba_vs_r    REAL,
    xba_vs_l    REAL,
    pa_vs_r     INTEGER,
    pa_vs_l     INTEGER,
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (game_date, team)
);

-- Fix F: per-team bullpen rosters with workload + leverage metadata.
-- Stored as a JSON list of reliever dicts because the schema varies
-- (recent_ip, LI, back_to_back, etc.) and the list is short.
CREATE TABLE IF NOT EXISTS team_bullpen_roster (
    game_date    TEXT NOT NULL,
    team         TEXT NOT NULL,
    roster_json  TEXT NOT NULL,
    fetched_at   TEXT NOT NULL,
    PRIMARY KEY (game_date, team)
);

-- Starter workload: avg IP/start in recent window
CREATE TABLE IF NOT EXISTS starter_workload (
    game_date        TEXT NOT NULL,
    pitcher_id       INTEGER NOT NULL,
    avg_ip_per_start REAL,
    starts_sampled   INTEGER,
    season_xba       REAL,
    fetched_at       TEXT NOT NULL,
    PRIMARY KEY (game_date, pitcher_id)
);

-- Pitcher-specific TTO splits (xBA per TTO)
CREATE TABLE IF NOT EXISTS tto_splits (
    game_date    TEXT NOT NULL,
    pitcher_id   INTEGER NOT NULL,
    tto_1_xba    REAL,
    tto_2_xba    REAL,
    tto_3_xba    REAL,
    tto_1_pa     INTEGER,
    tto_2_pa     INTEGER,
    tto_3_pa     INTEGER,
    fetched_at   TEXT NOT NULL,
    PRIMARY KEY (game_date, pitcher_id)
);

-- Today's umpire assignments + historical K% deviation
CREATE TABLE IF NOT EXISTS umpire_assignments (
    game_date      TEXT NOT NULL,
    game_pk        INTEGER NOT NULL,
    umpire_name    TEXT,
    k_pct          REAL,
    k_pct_dev      REAL,      -- umpire K% minus league K%
    games_sampled  INTEGER,
    fetched_at     TEXT NOT NULL,
    PRIMARY KEY (game_date, game_pk)
);

-- BvP history (batter vs pitcher career) - for annotation only
CREATE TABLE IF NOT EXISTS bvp_history (
    game_date   TEXT NOT NULL,
    batter_id   INTEGER NOT NULL,
    pitcher_id  INTEGER NOT NULL,
    pa          INTEGER,
    ab          INTEGER,
    hits        INTEGER,
    hr          INTEGER,
    so          INTEGER,
    bb          INTEGER,
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (game_date, batter_id, pitcher_id)
);
"""


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Yield a connection, ensuring schema exists."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Games & lineups
# ---------------------------------------------------------------------------
def save_games(game_date: date, games: list[dict[str, Any]]) -> None:
    with _connect() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO games
            (game_date, game_pk, home_team, away_team, venue, game_time,
             home_pitcher_id, away_pitcher_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    game_date.isoformat(),
                    g["game_pk"],
                    g.get("home_team"),
                    g.get("away_team"),
                    g.get("venue"),
                    g.get("game_time"),
                    g.get("home_pitcher_id"),
                    g.get("away_pitcher_id"),
                )
                for g in games
            ],
        )


def load_games(game_date: date) -> pd.DataFrame:
    with _connect() as conn:
        return pd.read_sql_query(
            "SELECT * FROM games WHERE game_date = ?",
            conn,
            params=(game_date.isoformat(),),
        )


def save_lineups(game_date: date, lineups: list[dict[str, Any]]) -> None:
    with _connect() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO lineups
            (game_date, game_pk, team, batter_id, batter_name, lineup_slot, bats)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    game_date.isoformat(),
                    lu["game_pk"],
                    lu["team"],
                    lu["batter_id"],
                    lu.get("batter_name"),
                    lu.get("lineup_slot"),
                    lu.get("bats"),
                )
                for lu in lineups
            ],
        )


def load_lineups(game_date: date) -> pd.DataFrame:
    with _connect() as conn:
        return pd.read_sql_query(
            "SELECT * FROM lineups WHERE game_date = ?",
            conn,
            params=(game_date.isoformat(),),
        )


# ---------------------------------------------------------------------------
# Batter profiles (pickled Statcast DataFrames)
# ---------------------------------------------------------------------------
def save_batter_profile(
    game_date: date, batter_id: int, profile_df: pd.DataFrame
) -> None:
    blob = pickle.dumps(profile_df)
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO batter_profiles
            (game_date, batter_id, profile_blob, fetched_at)
            VALUES (?, ?, ?, datetime('now'))
            """,
            (game_date.isoformat(), batter_id, blob),
        )


def load_batter_profile(game_date: date, batter_id: int) -> pd.DataFrame | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT profile_blob FROM batter_profiles "
            "WHERE game_date = ? AND batter_id = ?",
            (game_date.isoformat(), batter_id),
        ).fetchone()
    if row is None:
        return None
    return pickle.loads(row["profile_blob"])


# ---------------------------------------------------------------------------
# Pitcher profiles (pickled Statcast DataFrames — per-pitch view from the
# pitcher's POV, mirroring batter_profiles)
# ---------------------------------------------------------------------------
def save_pitcher_profile(
    game_date: date, pitcher_id: int, profile_df: pd.DataFrame
) -> None:
    blob = pickle.dumps(profile_df)
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO pitcher_profiles
            (game_date, pitcher_id, profile_blob, fetched_at)
            VALUES (?, ?, ?, datetime('now'))
            """,
            (game_date.isoformat(), pitcher_id, blob),
        )


def load_pitcher_profile(game_date: date, pitcher_id: int) -> pd.DataFrame | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT profile_blob FROM pitcher_profiles "
            "WHERE game_date = ? AND pitcher_id = ?",
            (game_date.isoformat(), pitcher_id),
        ).fetchone()
    if row is None:
        return None
    return pickle.loads(row["profile_blob"])


# ---------------------------------------------------------------------------
# Pitcher arsenals
# ---------------------------------------------------------------------------
def save_pitcher_arsenal(
    game_date: date,
    pitcher_id: int,
    throws: str,
    arsenal: dict[str, float],
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO pitcher_arsenals
            (game_date, pitcher_id, throws, arsenal_json, fetched_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            """,
            (
                game_date.isoformat(),
                pitcher_id,
                throws,
                json.dumps(arsenal),
            ),
        )


def load_pitcher_arsenal(
    game_date: date, pitcher_id: int
) -> tuple[str, dict[str, float]] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT throws, arsenal_json FROM pitcher_arsenals "
            "WHERE game_date = ? AND pitcher_id = ?",
            (game_date.isoformat(), pitcher_id),
        ).fetchone()
    if row is None:
        return None
    return row["throws"], json.loads(row["arsenal_json"])


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
def save_predictions(game_date: date, preds: pd.DataFrame) -> None:
    if preds.empty:
        return
    records = preds.to_dict(orient="records")
    with _connect() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO predictions
            (game_date, game_pk, batter_id, p_1_hit, p_2_hits, p_1_hr,
             p_tb_over_1_5, p_tb_over_2_5, expected_hits, expected_tb,
             computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            [
                (
                    game_date.isoformat(),
                    r["game_pk"],
                    r["batter_id"],
                    r.get("p_1_hit"),
                    r.get("p_2_hits"),
                    r.get("p_1_hr"),
                    r.get("p_tb_over_1_5"),
                    r.get("p_tb_over_2_5"),
                    r.get("expected_hits"),
                    r.get("expected_tb"),
                )
                for r in records
            ],
        )


def load_predictions(game_date: date) -> pd.DataFrame:
    with _connect() as conn:
        return pd.read_sql_query(
            "SELECT * FROM predictions WHERE game_date = ?",
            conn,
            params=(game_date.isoformat(),),
        )


# ---------------------------------------------------------------------------
# v2: Bullpen profiles
# ---------------------------------------------------------------------------
def save_bullpen_profile(
    game_date: date,
    team: str,
    xba_vs_r: float,
    xba_vs_l: float,
    pa_vs_r: int,
    pa_vs_l: int,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO bullpen_profiles
            (game_date, team, xba_vs_r, xba_vs_l, pa_vs_r, pa_vs_l, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (game_date.isoformat(), team, xba_vs_r, xba_vs_l, pa_vs_r, pa_vs_l),
        )


def load_bullpen_profile(game_date: date, team: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT xba_vs_r, xba_vs_l, pa_vs_r, pa_vs_l FROM bullpen_profiles "
            "WHERE game_date = ? AND team = ?",
            (game_date.isoformat(), team),
        ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Fix F: team bullpen rosters
# ---------------------------------------------------------------------------
def save_team_bullpen_roster(
    game_date: date,
    team: str,
    roster: list[dict[str, Any]],
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO team_bullpen_roster
            (game_date, team, roster_json, fetched_at)
            VALUES (?, ?, ?, datetime('now'))
            """,
            (game_date.isoformat(), team, json.dumps(roster)),
        )


def load_team_bullpen_roster(
    game_date: date, team: str
) -> list[dict[str, Any]] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT roster_json FROM team_bullpen_roster "
            "WHERE game_date = ? AND team = ?",
            (game_date.isoformat(), team),
        ).fetchone()
    if row is None:
        return None
    return json.loads(row["roster_json"])


# ---------------------------------------------------------------------------
# v2: Starter workload
# ---------------------------------------------------------------------------
def save_starter_workload(
    game_date: date,
    pitcher_id: int,
    avg_ip_per_start: float,
    starts_sampled: int,
    season_xba: float,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO starter_workload
            (game_date, pitcher_id, avg_ip_per_start, starts_sampled,
             season_xba, fetched_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                game_date.isoformat(),
                pitcher_id,
                avg_ip_per_start,
                starts_sampled,
                season_xba,
            ),
        )


def load_starter_workload(game_date: date, pitcher_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT avg_ip_per_start, starts_sampled, season_xba "
            "FROM starter_workload WHERE game_date = ? AND pitcher_id = ?",
            (game_date.isoformat(), pitcher_id),
        ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# v2: TTO splits
# ---------------------------------------------------------------------------
def save_tto_splits(
    game_date: date,
    pitcher_id: int,
    tto_xbas: dict[int, float],
    tto_pas: dict[int, int],
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO tto_splits
            (game_date, pitcher_id,
             tto_1_xba, tto_2_xba, tto_3_xba,
             tto_1_pa, tto_2_pa, tto_3_pa,
             fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                game_date.isoformat(),
                pitcher_id,
                tto_xbas.get(1), tto_xbas.get(2), tto_xbas.get(3),
                tto_pas.get(1, 0), tto_pas.get(2, 0), tto_pas.get(3, 0),
            ),
        )


def load_tto_splits(game_date: date, pitcher_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT tto_1_xba, tto_2_xba, tto_3_xba, tto_1_pa, tto_2_pa, tto_3_pa "
            "FROM tto_splits WHERE game_date = ? AND pitcher_id = ?",
            (game_date.isoformat(), pitcher_id),
        ).fetchone()
    if not row:
        return None
    return {
        "xba": {1: row["tto_1_xba"], 2: row["tto_2_xba"], 3: row["tto_3_xba"]},
        "pa": {1: row["tto_1_pa"], 2: row["tto_2_pa"], 3: row["tto_3_pa"]},
    }


# ---------------------------------------------------------------------------
# v2: Umpire assignments
# ---------------------------------------------------------------------------
def save_umpire_assignment(
    game_date: date,
    game_pk: int,
    umpire_name: str | None,
    k_pct: float | None,
    k_pct_dev: float | None,
    games_sampled: int | None,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO umpire_assignments
            (game_date, game_pk, umpire_name, k_pct, k_pct_dev,
             games_sampled, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                game_date.isoformat(),
                game_pk,
                umpire_name,
                k_pct,
                k_pct_dev,
                games_sampled,
            ),
        )


def load_umpire_assignment(game_date: date, game_pk: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT umpire_name, k_pct, k_pct_dev, games_sampled "
            "FROM umpire_assignments WHERE game_date = ? AND game_pk = ?",
            (game_date.isoformat(), game_pk),
        ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# v2: BvP history
# ---------------------------------------------------------------------------
def save_bvp(
    game_date: date,
    batter_id: int,
    pitcher_id: int,
    stats: dict,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO bvp_history
            (game_date, batter_id, pitcher_id, pa, ab, hits, hr, so, bb, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                game_date.isoformat(),
                batter_id,
                pitcher_id,
                stats.get("pa", 0),
                stats.get("ab", 0),
                stats.get("hits", 0),
                stats.get("hr", 0),
                stats.get("so", 0),
                stats.get("bb", 0),
            ),
        )


def load_bvp(game_date: date, batter_id: int, pitcher_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT pa, ab, hits, hr, so, bb FROM bvp_history "
            "WHERE game_date = ? AND batter_id = ? AND pitcher_id = ?",
            (game_date.isoformat(), batter_id, pitcher_id),
        ).fetchone()
    return dict(row) if row else None
