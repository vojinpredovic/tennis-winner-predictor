"""Load stage: write transformed matches into partitioned Parquet via DuckDB.

Storage is `data/matches/year=YYYY/*.parquet`, partitioned Hive-style by
match year and queried through DuckDB.

Parquet has no UNIQUE constraint, so idempotency comes from a per-partition
rewrite: for every year touched by the incoming batch, read that year's
existing partition, drop any row whose natural key is about to be replaced
(an anti-join), union in the incoming rows, and rewrite the whole
year=YYYY/ partition from the result. Untouched partitions are never read
or rewritten. Running the same batch through `load()` twice produces the
same union both times, so the row count stays stable across repeated runs.

Normal runs only touch rows at or after MAX(date) minus a trailing window,
since the Kaggle source updates daily and can retroactively correct recent
results and rankings. `--full-refresh` reprocesses every row.

Natural key is (date, tournament, round, player_1, player_2). `round` is
part of the key because round-robin events (e.g. the Masters Cup) can have
the same two players meet twice on the same date in the same tournament,
once in round robin and once in the final, with different results.
Dropping `round` would collide those two matches into one.
"""
import argparse
import logging
import shutil
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import duckdb
import pandas as pd

from pipeline.config import MATCHES_DIR, RAW_CSV_PATH
from pipeline.transform import transform

logger = logging.getLogger(__name__)

RELOAD_TRAILING_DAYS = 7

# DataFrame column -> matches column
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

# Explicit target type for every column written to Parquet. `year` is
# derived from `date` to route rows to a partition directory (year=YYYY/)
# and is projected away before the final write (see _write_year_partition),
# so it never lands in the on-disk schema below.
_COLUMN_TYPES = {
    'date': 'DATE', 'tournament': 'VARCHAR', 'series': 'VARCHAR',
    'court': 'VARCHAR', 'surface': 'VARCHAR', 'round': 'VARCHAR',
    'best_of': 'INTEGER', 'player_1': 'VARCHAR', 'player_2': 'VARCHAR',
    'winner': 'VARCHAR', 'rank_1': 'INTEGER', 'rank_2': 'INTEGER',
    'pts_1': 'INTEGER', 'pts_2': 'INTEGER', 'odd_1': 'DOUBLE',
    'odd_2': 'DOUBLE', 'score': 'VARCHAR', 'rank_diff': 'INTEGER',
    'pts_diff': 'INTEGER', 'form_1': 'DOUBLE', 'form_2': 'DOUBLE',
}
# SELECT list that casts every incoming column to its explicit type, plus
# `year`. `year` is kept here so the anti-join/union queries can filter and
# group on it; _write_year_partition drops it before the final write.
_TYPED_SELECT_SQL = ', '.join(
    f'CAST({col} AS {_COLUMN_TYPES[col]}) AS {col}' for col in _COLUMNS
) + ', CAST(year AS INTEGER) AS year'


def get_connection() -> duckdb.DuckDBPyConnection:
    """DuckDB acts as a query engine over the Parquet files under
    MATCHES_DIR; the connection itself is in-memory."""
    MATCHES_DIR.mkdir(parents=True, exist_ok=True)
    return duckdb.connect()


def ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Parquet has no schema DDL to create. Column types are enforced by
    the CAST in _TYPED_SELECT_SQL each time a partition is written."""
    MATCHES_DIR.mkdir(parents=True, exist_ok=True)


def _existing_max_date(conn: duckdb.DuckDBPyConnection) -> pd.Timestamp | None:
    pattern = str(MATCHES_DIR / '**' / '*.parquet')
    if not any(MATCHES_DIR.glob('year=*/*.parquet')):
        return None
    row = conn.execute(
        "SELECT MAX(date) FROM read_parquet("
        f"'{pattern}', hive_partitioning=true)"
    ).fetchone()
    return pd.Timestamp(row[0]) if row and row[0] is not None else None


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns=_COLUMN_MAP)[_COLUMNS].copy()
    out['year'] = out['date'].dt.year
    return out


def _assert_natural_key_unique(df: pd.DataFrame) -> None:
    dup_mask = df.duplicated(subset=list(_KEY_COLUMNS), keep=False)
    if dup_mask.any():
        dup_keys = df.loc[dup_mask, list(_KEY_COLUMNS)].drop_duplicates()
        raise ValueError(
            f'{len(dup_keys)} duplicate natural key(s) in incoming batch '
            f'(date, tournament, round, player_1, player_2):\n{dup_keys}'
        )


def _existing_partition(
    conn: duckdb.DuckDBPyConnection, year: int
) -> duckdb.DuckDBPyRelation:
    """Typed relation of every row currently stored in year=Y's partition,
    or an empty relation with the same schema if that partition doesn't
    exist yet (a new year in the incoming batch)."""
    partition_dir = MATCHES_DIR / f'year={year}'
    if not partition_dir.exists():
        return conn.sql(f'SELECT {_TYPED_SELECT_SQL} FROM incoming LIMIT 0')
    pattern = str(partition_dir / '*.parquet').replace("'", "''")
    value_cols = ', '.join(_COLUMNS)
    return conn.sql(f"""
        SELECT {value_cols}, {year} AS year
        FROM read_parquet('{pattern}')
    """)


def _write_year_partition(
    conn: duckdb.DuckDBPyConnection, year: int, rows: duckdb.DuckDBPyRelation
) -> None:
    """Rewrite year=Y/ from `rows`, the full replacement content for that
    year. Writes to a temp dir first and swaps it in, so a crash mid-write
    can't leave the partition half-overwritten."""
    tmp_root = Path(tempfile.mkdtemp(dir=MATCHES_DIR, prefix='.tmp-write-'))
    try:
        conn.execute(f"""
            COPY (SELECT * FROM rows) TO '{tmp_root}'
            (FORMAT PARQUET, PARTITION_BY (year), OVERWRITE_OR_IGNORE true)
        """)

        # DuckDB's Parquet writer leaves the PARTITION_BY column in the
        # files it writes (open bug as of 1.5.5:
        # github.com/duckdb/duckdb/issues/12147), storing `year` twice:
        # once in the directory name, once inside the file. Re-project it
        # away so the on-disk schema matches _COLUMN_TYPES exactly.
        value_cols = ', '.join(_COLUMNS)
        raw_glob = tmp_root / f'year={year}' / '*.parquet'
        pattern = str(raw_glob).replace("'", "''")
        cleaned = tmp_root / 'cleaned.parquet'
        conn.execute(f"""
            COPY (SELECT {value_cols} FROM read_parquet('{pattern}'))
            TO '{cleaned}' (FORMAT PARQUET)
        """)

        target = MATCHES_DIR / f'year={year}'
        if target.exists():
            shutil.rmtree(target)
        target.mkdir()
        shutil.move(str(cleaned), str(target / 'data_0.parquet'))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def load(
    df: pd.DataFrame,
    conn: duckdb.DuckDBPyConnection,
    full_refresh: bool = False,
) -> tuple[int, int]:
    """Upsert df's rows into the Parquet dataset.

    Returns (inserted, updated).
    """
    ensure_schema(conn)
    t0 = time.monotonic()

    if not full_refresh:
        cutoff = _existing_max_date(conn)
        if cutoff is not None:
            window_start = cutoff - timedelta(days=RELOAD_TRAILING_DAYS)
            df = df[df['Date'] >= window_start]

    if df.empty:
        logger.info('No rows in the incremental window; nothing to load')
        return 0, 0

    incoming = _prepare_df(df)
    _assert_natural_key_unique(incoming)

    conn.register('incoming', incoming)
    typed_incoming = conn.sql(f'SELECT {_TYPED_SELECT_SQL} FROM incoming')

    touched_years = sorted(incoming['year'].unique().tolist())
    inserted = updated = 0

    for year in touched_years:
        year_incoming = typed_incoming.filter(f'year = {year}')
        conn.register('year_incoming', year_incoming)
        existing = _existing_partition(conn, year)
        conn.register('existing', existing)

        key_cols = ' AND '.join(
            f'e.{c} IS NOT DISTINCT FROM i.{c}' for c in _KEY_COLUMNS
        )
        counts = conn.execute(f"""
            SELECT
                COUNT(*) FILTER (
                    WHERE EXISTS (SELECT 1 FROM existing e WHERE {key_cols})
                ) AS updated,
                COUNT(*) FILTER (
                    WHERE NOT EXISTS (
                        SELECT 1 FROM existing e WHERE {key_cols}
                    )
                ) AS inserted
            FROM year_incoming i
        """).fetchone()
        updated += counts[0]
        inserted += counts[1]

        kept_existing = conn.sql(f"""
            SELECT existing.* FROM existing
            WHERE NOT EXISTS (
                SELECT 1 FROM year_incoming i
                WHERE {key_cols.replace('e.', 'existing.')}
            )
        """)
        combined = kept_existing.union(year_incoming)
        _write_year_partition(conn, year, combined)

        conn.unregister('year_incoming')
        conn.unregister('existing')

    conn.unregister('incoming')

    logger.info(
        'Loaded %d rows across %d partition(s) (%s) in %.2fs: '
        '%d inserted, %d updated',
        len(incoming), len(touched_years), touched_years,
        time.monotonic() - t0, inserted, updated,
    )
    return inserted, updated


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    parser = argparse.ArgumentParser(
        description='Transform + load the raw CSV into partitioned Parquet.'
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
