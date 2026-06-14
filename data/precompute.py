"""
data/precompute.py  (fixed)
───────────────────────────
Phase 1 Day 2 – Load cleaned data into PostgreSQL and pre-compute
all aggregate tables.

Fixes vs original:
  1. load_raw_tables now TRUNCATEs then APPENDs instead of replace=DROP,
     so PRIMARY KEY / FK constraints from SCHEMA_SQL are preserved and
     ON CONFLICT clauses in aggregate queries work correctly.
  2. compute_player_career_stats split into two separate execute() calls
     so SQLAlchemy runs both INSERT statements (not just the first).
  3. Removed unused `from sqlalchemy.orm import Session` import.
  4. Added `match_type TEXT` and all previously dropped CSV columns to
     both matches and deliveries tables in SCHEMA_SQL, so no data is lost.
  5. MATCHES_DB_COLS and DELIVERIES_DB_COLS updated to match the expanded
     schema — no more "Dropping N extra CSV columns" warnings.
  6. WHERE batter/bowler IS NOT NULL guards in aggregate queries prevent
     NOT NULL constraint violations from wide/no-ball deliveries.
  7. Wider NUMERIC precision on venue_stats and team_form columns to
     prevent numeric overflow errors.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


# Exact set of columns the matches table accepts.
MATCHES_DB_COLS = {
    "id", "season", "date", "city", "venue", "match_type",
    "team1", "team2", "toss_winner", "toss_decision", "winner",
    "win_by_runs", "win_by_wickets", "player_of_match",
    "dl_applied", "target_runs",
    "result", "result_margin", "target_overs", "super_over",
    "method", "umpire1", "umpire2",
}

# Exact set of columns the deliveries table accepts.
DELIVERIES_DB_COLS = {
    "match_id", "inning", "batting_team", "bowling_team", "over", "ball",
    "batter", "bowler", "non_striker", "batsman_runs", "extra_runs",
    "total_runs", "is_wicket", "dismissal_kind", "player_dismissed",
    "fielder", "intent", "phase",
    "extras_type", "wides", "noballs", "byes", "legbyes", "penalty",
    "other_wicket_type", "other_player_dismissed", "season", "start_date", "venue",
}

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS matches (
    id              INTEGER PRIMARY KEY,
    season          INTEGER,
    date            DATE,
    city            TEXT,
    venue           TEXT,
    match_type      TEXT,
    team1           TEXT,
    team2           TEXT,
    toss_winner     TEXT,
    toss_decision   TEXT,
    winner          TEXT,
    win_by_runs     INTEGER DEFAULT 0,
    win_by_wickets  INTEGER DEFAULT 0,
    player_of_match TEXT,
    dl_applied      BOOLEAN DEFAULT FALSE,
    target_runs     INTEGER DEFAULT 0,
    result          TEXT,
    result_margin   NUMERIC(10,2),
    target_overs    NUMERIC(6,2),
    super_over      TEXT,
    method          TEXT,
    umpire1         TEXT,
    umpire2         TEXT
);

CREATE TABLE IF NOT EXISTS deliveries (
    id                      BIGSERIAL PRIMARY KEY,
    match_id                INTEGER REFERENCES matches(id),
    inning                  SMALLINT,
    batting_team            TEXT,
    bowling_team            TEXT,
    over                    SMALLINT,
    ball                    SMALLINT,
    batter                  TEXT,
    bowler                  TEXT,
    non_striker             TEXT,
    batsman_runs            SMALLINT,
    extra_runs              SMALLINT,
    total_runs              SMALLINT,
    is_wicket               BOOLEAN DEFAULT FALSE,
    dismissal_kind          TEXT,
    player_dismissed        TEXT,
    fielder                 TEXT,
    intent                  TEXT,
    phase                   TEXT,
    extras_type             TEXT,
    wides                   SMALLINT,
    noballs                 SMALLINT,
    byes                    SMALLINT,
    legbyes                 SMALLINT,
    penalty                 SMALLINT,
    other_wicket_type       TEXT,
    other_player_dismissed  TEXT,
    season                  INTEGER,
    start_date              DATE,
    venue                   TEXT
);

CREATE TABLE IF NOT EXISTS player_career_stats (
    player          TEXT,
    season          INTEGER,
    role            TEXT,
    phase           TEXT,
    innings         INTEGER,
    runs            INTEGER,
    balls_faced     INTEGER,
    strike_rate     NUMERIC(8,2),
    average         NUMERIC(8,2),
    wickets         INTEGER,
    economy         NUMERIC(8,2),
    dot_pct         NUMERIC(8,2),
    PRIMARY KEY (player, season, role, phase)
);

CREATE TABLE IF NOT EXISTS h2h_records (
    batter          TEXT,
    bowler          TEXT,
    phase           TEXT,
    balls           INTEGER,
    runs            INTEGER,
    wickets         INTEGER,
    strike_rate     NUMERIC(8,2),
    dot_pct         NUMERIC(8,2),
    boundary_pct    NUMERIC(8,2),
    PRIMARY KEY (batter, bowler, phase)
);

CREATE TABLE IF NOT EXISTS venue_stats (
    venue               TEXT,
    season              INTEGER,
    avg_first_inn       NUMERIC(10,2),
    win_bat_first_pct   NUMERIC(8,2),
    avg_powerplay_rr    NUMERIC(8,2),
    avg_middle_rr       NUMERIC(8,2),
    avg_death_rr        NUMERIC(8,2),
    PRIMARY KEY (venue, season)
);

CREATE TABLE IF NOT EXISTS team_form (
    team            TEXT,
    season          INTEGER,
    last5_wins      INTEGER,
    last5_losses    INTEGER,
    avg_rr_diff     NUMERIC(10,2),
    form_rating     NUMERIC(8,2),
    PRIMARY KEY (team, season)
);

CREATE TABLE IF NOT EXISTS squad_strength (
    team            TEXT,
    season          INTEGER,
    strength_index  NUMERIC(8,2),
    PRIMARY KEY (team, season)
);

CREATE INDEX IF NOT EXISTS idx_deliveries_match   ON deliveries(match_id);
CREATE INDEX IF NOT EXISTS idx_deliveries_batter  ON deliveries(batter);
CREATE INDEX IF NOT EXISTS idx_deliveries_bowler  ON deliveries(bowler);
CREATE INDEX IF NOT EXISTS idx_deliveries_phase   ON deliveries(phase);
CREATE INDEX IF NOT EXISTS idx_matches_season     ON matches(season);
CREATE INDEX IF NOT EXISTS idx_matches_teams      ON matches(team1, team2);
CREATE INDEX IF NOT EXISTS idx_h2h_pair           ON h2h_records(batter, bowler);
"""


def get_engine(db_url: str):
    print(f"DEBUG CONNECTING TO: {db_url}")
    return create_engine(db_url, pool_pre_ping=True, pool_size=10)


def init_schema(engine) -> None:
    with engine.connect() as conn:
        conn.execute(text(SCHEMA_SQL))
        conn.commit()
    logger.info("Schema initialised.")


def assign_phase(over: int) -> str:
    if over < 6:
        return "powerplay"
    if over < 15:
        return "middle"
    return "death"


def load_raw_tables(engine, processed_dir: Path = Path("data/processed")) -> None:
    matches    = pd.read_csv(processed_dir / "matches_clean.csv")
    deliveries = pd.read_csv(
        processed_dir / "deliveries_labelled.csv", low_memory=False
    )

    deliveries["phase"]     = deliveries["over"].apply(assign_phase)
    deliveries["is_wicket"] = deliveries["dismissal_kind"].notna()

    # Coerce id to int so it matches INTEGER PRIMARY KEY
    matches["id"] = pd.to_numeric(matches["id"], errors="coerce").fillna(0).astype(int)
    deliveries["match_id"] = pd.to_numeric(
        deliveries["match_id"], errors="coerce"
    ).fillna(0).astype(int)

    # ── Filter matches columns ─────────────────────────────────────────────
    matches_cols_present = [c for c in matches.columns if c in MATCHES_DB_COLS]
    extra_match_cols = set(matches.columns) - MATCHES_DB_COLS
    if extra_match_cols:
        logger.info(
            "Dropping %d extra matches CSV columns not in DB schema: %s",
            len(extra_match_cols), sorted(extra_match_cols),
        )
    matches = matches[matches_cols_present]

    # ── Filter deliveries columns ──────────────────────────────────────────
    deliveries_cols_present = [c for c in deliveries.columns if c in DELIVERIES_DB_COLS]
    extra_delivery_cols = set(deliveries.columns) - DELIVERIES_DB_COLS
    if extra_delivery_cols:
        logger.info(
            "Dropping %d extra deliveries CSV columns not in DB schema: %s",
            len(extra_delivery_cols), sorted(extra_delivery_cols),
        )
    deliveries = deliveries[deliveries_cols_present]
    # ──────────────────────────────────────────────────────────────────────

    with engine.connect() as conn:
        # Truncate in FK-safe order (child first)
        conn.execute(text(
            "TRUNCATE TABLE deliveries, matches RESTART IDENTITY CASCADE;"
        ))
        conn.commit()

    matches.to_sql(
        "matches", engine, if_exists="append", index=False,
        method="multi", chunksize=500,
    )
    deliveries.to_sql(
        "deliveries", engine, if_exists="append", index=False,
        method="multi", chunksize=5000,
    )
    logger.info(
        "Loaded %d matches and %d deliveries into PostgreSQL.",
        len(matches), len(deliveries),
    )


def compute_player_career_stats(engine) -> None:
    batsman_sql = """
    INSERT INTO player_career_stats
    SELECT
        batter AS player, m.season, 'batsman' AS role, d.phase,
        COUNT(DISTINCT d.match_id) AS innings,
        SUM(d.batsman_runs)        AS runs,
        COUNT(*)                   AS balls_faced,
        ROUND(100.0 * SUM(d.batsman_runs) / NULLIF(COUNT(*), 0), 2) AS strike_rate,
        ROUND(
            SUM(d.batsman_runs)::NUMERIC /
            NULLIF(SUM(CASE WHEN d.dismissal_kind IS NOT NULL
                             AND d.player_dismissed = d.batter THEN 1 ELSE 0 END), 0),
        2) AS average,
        0 AS wickets, 0 AS economy,
        ROUND(100.0 * SUM(CASE WHEN d.batsman_runs=0 AND d.extra_runs=0
                               THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS dot_pct
    FROM deliveries d
    JOIN matches m ON d.match_id = m.id
    WHERE d.batter IS NOT NULL
    GROUP BY batter, m.season, d.phase
    ON CONFLICT (player, season, role, phase) DO UPDATE
        SET runs        = EXCLUDED.runs,
            balls_faced = EXCLUDED.balls_faced,
            strike_rate = EXCLUDED.strike_rate,
            average     = EXCLUDED.average,
            dot_pct     = EXCLUDED.dot_pct;
    """

    bowler_sql = """
    INSERT INTO player_career_stats
    SELECT
        bowler AS player, m.season, 'bowler' AS role, d.phase,
        COUNT(DISTINCT d.match_id) AS innings,
        0 AS runs, 0 AS balls_faced, 0 AS strike_rate, 0 AS average,
        SUM(CASE WHEN d.dismissal_kind IS NOT NULL
                  AND d.dismissal_kind NOT IN ('run out', 'retired hurt')
                 THEN 1 ELSE 0 END) AS wickets,
        ROUND(6.0 * SUM(d.total_runs) / NULLIF(COUNT(*), 0), 2) AS economy,
        ROUND(100.0 * SUM(CASE WHEN d.total_runs=0 THEN 1 ELSE 0 END) /
              NULLIF(COUNT(*), 0), 2) AS dot_pct
    FROM deliveries d
    JOIN matches m ON d.match_id = m.id
    WHERE d.bowler IS NOT NULL
    GROUP BY bowler, m.season, d.phase
    ON CONFLICT (player, season, role, phase) DO UPDATE
        SET wickets = EXCLUDED.wickets,
            economy = EXCLUDED.economy,
            dot_pct = EXCLUDED.dot_pct;
    """

    with engine.connect() as conn:
        conn.execute(text(batsman_sql))
        conn.execute(text(bowler_sql))
        conn.commit()
    logger.info("Player career stats computed.")


def compute_h2h_records(engine) -> None:
    sql = """
    INSERT INTO h2h_records
    SELECT
        batter, bowler, phase,
        COUNT(*) AS balls,
        SUM(batsman_runs) AS runs,
        SUM(CASE WHEN dismissal_kind IS NOT NULL
                  AND player_dismissed = batter THEN 1 ELSE 0 END) AS wickets,
        ROUND(100.0 * SUM(batsman_runs) / NULLIF(COUNT(*), 0), 2) AS strike_rate,
        ROUND(100.0 * SUM(CASE WHEN batsman_runs=0 AND extra_runs=0
                               THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS dot_pct,
        ROUND(100.0 * SUM(CASE WHEN batsman_runs >= 4
                               THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS boundary_pct
    FROM deliveries
    WHERE batter IS NOT NULL AND bowler IS NOT NULL
    GROUP BY batter, bowler, phase
    ON CONFLICT (batter, bowler, phase) DO UPDATE
        SET balls        = EXCLUDED.balls,
            runs         = EXCLUDED.runs,
            wickets      = EXCLUDED.wickets,
            strike_rate  = EXCLUDED.strike_rate,
            dot_pct      = EXCLUDED.dot_pct,
            boundary_pct = EXCLUDED.boundary_pct;
    """
    with engine.connect() as conn:
        conn.execute(text(sql))
        conn.commit()
    logger.info("H2H records computed.")


def compute_venue_stats(engine) -> None:
    sql = """
    INSERT INTO venue_stats
    SELECT
        m.venue, m.season,
        ROUND(AVG(CASE WHEN d.inning=1 THEN inn_totals.inn_runs END), 2) AS avg_first_inn,
        ROUND(100.0 * SUM(CASE WHEN m.winner=m.team1 AND m.toss_decision='bat'   THEN 1
                               WHEN m.winner=m.team2 AND m.toss_decision='field' THEN 1
                               ELSE 0 END) / NULLIF(COUNT(DISTINCT m.id), 0), 2) AS win_bat_first_pct,
        ROUND(AVG(CASE WHEN d.phase='powerplay' THEN d.total_runs END) * 6.0, 2) AS avg_powerplay_rr,
        ROUND(AVG(CASE WHEN d.phase='middle'    THEN d.total_runs END) * 6.0, 2) AS avg_middle_rr,
        ROUND(AVG(CASE WHEN d.phase='death'     THEN d.total_runs END) * 6.0, 2) AS avg_death_rr
    FROM matches m
    JOIN deliveries d ON m.id = d.match_id
    JOIN (
        SELECT match_id, inning, SUM(total_runs) AS inn_runs
        FROM deliveries GROUP BY match_id, inning
    ) inn_totals ON inn_totals.match_id = d.match_id AND inn_totals.inning = d.inning
    WHERE m.venue IS NOT NULL AND m.season IS NOT NULL
    GROUP BY m.venue, m.season
    ON CONFLICT (venue, season) DO UPDATE
        SET avg_first_inn     = EXCLUDED.avg_first_inn,
            win_bat_first_pct = EXCLUDED.win_bat_first_pct,
            avg_powerplay_rr  = EXCLUDED.avg_powerplay_rr,
            avg_middle_rr     = EXCLUDED.avg_middle_rr,
            avg_death_rr      = EXCLUDED.avg_death_rr;
    """
    with engine.connect() as conn:
        conn.execute(text(sql))
        conn.commit()
    logger.info("Venue stats computed.")


def compute_team_form(engine) -> None:
    sql = """
    INSERT INTO team_form
    WITH match_results AS (
        SELECT id, season, team1 AS team,
               CASE WHEN winner = team1 THEN 1 ELSE 0 END AS won,
               (win_by_runs + win_by_wickets * 10) AS margin
        FROM matches
        UNION ALL
        SELECT id, season, team2 AS team,
               CASE WHEN winner = team2 THEN 1 ELSE 0 END AS won,
               (win_by_runs + win_by_wickets * 10) AS margin
        FROM matches
    ),
    ranked AS (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY team, season ORDER BY id DESC) AS rn
        FROM match_results
    )
    SELECT team, season,
           SUM(CASE WHEN rn<=5 AND won=1 THEN 1 ELSE 0 END) AS last5_wins,
           SUM(CASE WHEN rn<=5 AND won=0 THEN 1 ELSE 0 END) AS last5_losses,
           ROUND(AVG(CASE WHEN rn<=5 THEN margin END), 2)    AS avg_rr_diff,
           ROUND(SUM(CASE WHEN rn<=5 AND won=1 THEN 1 ELSE 0 END)::NUMERIC / 5 * 100, 2) AS form_rating
    FROM ranked
    WHERE team IS NOT NULL AND season IS NOT NULL
    GROUP BY team, season
    ON CONFLICT (team, season) DO UPDATE
        SET last5_wins   = EXCLUDED.last5_wins,
            last5_losses = EXCLUDED.last5_losses,
            avg_rr_diff  = EXCLUDED.avg_rr_diff,
            form_rating  = EXCLUDED.form_rating;
    """
    with engine.connect() as conn:
        conn.execute(text(sql))
        conn.commit()
    logger.info("Team form computed.")


def run_all(db_url: str, processed_dir: Path = Path("data/processed")) -> None:
    engine = get_engine(db_url)
    init_schema(engine)
    load_raw_tables(engine, processed_dir)
    compute_player_career_stats(engine)
    compute_h2h_records(engine)
    compute_venue_stats(engine)
    compute_team_form(engine)
    logger.info("All pre-computed tables ready.")


if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)
    run_all(os.environ["DATABASE_URL"])