"""Load stage: write transformed matches into SQLite with an idempotent upsert.

Normal runs only touch rows at or after (MAX(date) in the table - a trailing
window), since the Kaggle source updates daily and can retroactively correct
recent results/rankings. `--full-refresh` reprocesses every row.

Natural key is (date, tournament, round, player_1, player_2) -- `round` is
required because round-robin events (e.g. the Masters Cup) can have the same
two players meet twice on the same date within the same tournament (round
robin, then final), with different results. Without `round` those two real
matches collide and one silently overwrites the other.
"""
import argparse
import logging
import sqlite3
from datetime import timedelta
from pathlib import Path

import pandas as pd

from pipeline.config import DB_PATH, RAW_CSV_PATH
from pipeline.transform import transform

logger = logging.getLogger(__name__)

RELOAD_TRAILING_DAYS = 7

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS matches (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT    NOT NULL,
    tournament  TEXT    NOT NULL,
    series      TEXT,
    court       TEXT,
    surface     TEXT,
    round       TEXT,
    best_of     INTEGER,
    player_1    TEXT    NOT NULL,
    player_2    TEXT    NOT NULL,
    winner      TEXT    NOT NULL,
    rank_1      INTEGER,
    rank_2      INTEGER,
    pts_1       INTEGER,
    pts_2       INTEGER,
    odd_1       REAL,
    odd_2       REAL,
    score       TEXT,
    rank_diff   INTEGER,
    pts_diff    INTEGER,
    form_1      REAL    NOT NULL,
    form_2      REAL    NOT NULL,
    UNIQUE (date, tournament, round, player_1, player_2)
)
"""
_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date)"

# DataFrame column -> matches table column
_COLUMN_MAP = {
    'Date': 'date', 'Tournament': 'tournament', 'Series': 'series',
    'Court': 'court', 'Surface': 'surface', 'Round': 'round',
    'Best of': 'best_of', 'Player_1': 'player_1', 'Player_2': 'player_2',
    'Winner': 'winner', 'Rank_1': 'rank_1', 'Rank_2': 'rank_2',
    'Pts_1': 'pts_1', 'Pts_2': 'pts_2', 'Odd_1': 'odd_1', 'Odd_2': 'odd_2',
    'Score': 'score', 'Rank_diff': 'rank_diff', 'Pts_diff': 'pts_diff',
    'Form_1': 'form_1', 'Form_2': 'form_2',
}
_COLUMNS = list(_COLUMN_MAP.values())
_KEY_COLUMNS = ('date', 'tournament', 'round', 'player_1', 'player_2')
_KEY_INDICES = [_COLUMNS.index(c) for c in _KEY_COLUMNS]
_INT_COLUMNS = {
    'best_of', 'rank_1', 'rank_2', 'pts_1', 'pts_2', 'rank_diff', 'pts_diff',
}
_UPDATE_COLUMNS = [c for c in _COLUMNS if c not in _KEY_COLUMNS]

_UPSERT_SQL = f"""
INSERT INTO matches ({', '.join(_COLUMNS)})
VALUES ({', '.join('?' for _ in _COLUMNS)})
ON CONFLICT({', '.join(_KEY_COLUMNS)}) DO UPDATE SET
    {', '.join(f'{c} = excluded.{c}' for c in _UPDATE_COLUMNS)}
"""


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_SCHEMA_SQL)
    conn.execute(_INDEX_SQL)
    conn.commit()


def _prepare_rows(df: pd.DataFrame) -> list[tuple]:
    out = df.rename(columns=_COLUMN_MAP)[_COLUMNS].copy()
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')

    def to_py(col: str, val):
        if pd.isna(val):
            return None
        return int(val) if col in _INT_COLUMNS else val

    return [
        tuple(to_py(col, val) for col, val in zip(_COLUMNS, row))
        for row in out.itertuples(index=False)
    ]


def load(
    df: pd.DataFrame, conn: sqlite3.Connection, full_refresh: bool = False
) -> tuple[int, int]:
    """Upsert df's rows into the matches table. Returns (inserted, updated)."""
    ensure_schema(conn)

    if not full_refresh:
        cutoff = conn.execute('SELECT MAX(date) FROM matches').fetchone()[0]
        if cutoff is not None:
            window_start = (
                pd.Timestamp(cutoff) - timedelta(days=RELOAD_TRAILING_DAYS)
            )
            df = df[df['Date'] >= window_start]

    rows = _prepare_rows(df)
    if not rows:
        logger.info('No rows in the incremental window; nothing to load')
        return 0, 0

    batch_min_date = min(r[0] for r in rows)
    existing = set(
        conn.execute(
            f"SELECT {', '.join(_KEY_COLUMNS)} FROM matches WHERE date >= ?",
            (batch_min_date,),
        ).fetchall()
    )
    batch_keys = {tuple(r[i] for i in _KEY_INDICES) for r in rows}
    updated = len(batch_keys & existing)
    inserted = len(batch_keys) - updated

    with conn:
        conn.executemany(_UPSERT_SQL, rows)

    logger.info(
        'Loaded %d rows: %d inserted, %d updated', len(rows), inserted, updated
    )
    return inserted, updated


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    parser = argparse.ArgumentParser(
        description='Transform + load the raw CSV into SQLite.'
    )
    parser.add_argument(
        '--full-refresh', action='store_true',
        help='Reload every row instead of just the incremental window.',
    )
    args = parser.parse_args()

    df = transform(RAW_CSV_PATH)
    conn = get_connection()
    try:
        load(df, conn, full_refresh=args.full_refresh)
    finally:
        conn.close()


if __name__ == '__main__':
    main()
