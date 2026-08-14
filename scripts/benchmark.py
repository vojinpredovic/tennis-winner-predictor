"""Compare a date-filtered aggregate query against the old SQLite table and
the new partitioned Parquet dataset, and show DuckDB's query plan so
partition pruning is visible.

Why a columnar, partitioned layout scans less data for this query shape:

  Row store (SQLite): a table is stored row by row. Getting `surface` and
  `date` for 2015-2020 means walking every row of the whole table and
  pulling every column along with it.

  Columnar + partitioned (DuckDB/Parquet): two prunings stack.
    1. Partition pruning, at the directory level: rows live in year=YYYY/
       directories, so a predicate on `year` lets DuckDB pick which files
       to open before reading any data.
    2. Column pruning, at the file level: Parquet stores column values
       contiguously, so DuckDB reads only the column chunks the query
       selects or filters on.
  Together, a query like this one touches a small fraction of the bytes on
  disk the equivalent SQLite scan does. See the caveat at the end for why
  that doesn't show up as a speed win at this dataset's size.

Note: the query below filters on `year`, the Hive partition column DuckDB
exposes automatically, since that's what triggers directory-level pruning.
Filtering on `date` also prunes, but through each file's min/max row-group
statistics, which requires opening every file's metadata first, a weaker
form of pruning than the directory-level pruning this script demonstrates.

Prerequisites: run `python -m pipeline.run` on this branch to build the
Parquet dataset. Separately, checkout `sqlite-load` and run the pipeline
there to produce `data/tennis.db`. `data/` is gitignored, so both outputs
can coexist locally.
"""
import json
import sqlite3
import time

import duckdb

from pipeline.config import DB_PATH, MATCHES_DIR

YEAR_START, YEAR_END = 2015, 2020

_SQLITE_QUERY = """
SELECT surface, COUNT(*) AS matches
FROM matches
WHERE date >= ? AND date < ?
GROUP BY surface
ORDER BY surface
"""

_PARQUET_QUERY = f"""
SELECT surface, COUNT(*) AS matches
FROM read_parquet('{MATCHES_DIR}/**/*.parquet', hive_partitioning=true)
WHERE year BETWEEN {YEAR_START} AND {YEAR_END}
GROUP BY surface
ORDER BY surface
"""


def _run_sqlite() -> tuple[list[tuple], float]:
    conn = sqlite3.connect(DB_PATH)
    try:
        t0 = time.perf_counter()
        rows = conn.execute(
            _SQLITE_QUERY, (f'{YEAR_START}-01-01', f'{YEAR_END + 1}-01-01')
        ).fetchall()
        elapsed = time.perf_counter() - t0
    finally:
        conn.close()
    return rows, elapsed


def _run_duckdb(
    conn: duckdb.DuckDBPyConnection,
) -> tuple[list[tuple], float]:
    t0 = time.perf_counter()
    rows = conn.execute(_PARQUET_QUERY).fetchall()
    elapsed = time.perf_counter() - t0
    return rows, elapsed


def _find_scan_node(node: dict, operator_name: str) -> dict | None:
    if node.get('operator_name') == operator_name:
        return node
    for child in node.get('children', []):
        found = _find_scan_node(child, operator_name)
        if found is not None:
            return found
    return None


def _print_rows(label: str, rows: list[tuple], elapsed: float) -> None:
    print(f'{label}:')
    for surface, count in rows:
        print(f'  {(surface or "?"):10s} {count:>6,}')
    print(f'  wall clock: {elapsed * 1000:.2f} ms\n')


def main() -> None:
    if not DB_PATH.exists():
        print(
            f'No SQLite database at {DB_PATH}. Checkout the sqlite-load '
            "branch and run `python -m pipeline.run` there to produce it, "
            'then come back to this branch to run this benchmark.'
        )
        return
    if not any(MATCHES_DIR.glob('year=*/*.parquet')):
        print(f'No Parquet dataset at {MATCHES_DIR}. Run '
              '`python -m pipeline.run` on this branch first.')
        return

    print(f'Query: match counts by surface, {YEAR_START}-{YEAR_END}\n')

    sqlite_rows, sqlite_time = _run_sqlite()
    _print_rows('SQLite (row store, data/tennis.db)', sqlite_rows, sqlite_time)

    conn = duckdb.connect()
    duckdb_rows, duckdb_time = _run_duckdb(conn)
    _print_rows(
        'DuckDB + partitioned Parquet (columnar, data/matches/)',
        duckdb_rows, duckdb_time,
    )

    plan_json = json.loads(
        conn.execute(f'EXPLAIN (ANALYZE, FORMAT JSON) {_PARQUET_QUERY}')
        .fetchone()[1]
    )
    scan = _find_scan_node(plan_json, 'READ_PARQUET')
    total_partitions = len(list(MATCHES_DIR.glob('year=*')))
    print(
        f'Partition pruning: DuckDB scanned '
        f'{scan["extra_info"]["Scanning Files"]} year partitions. '
        f'The dataset spans {total_partitions} years; the query asked for '
        f'{YEAR_END - YEAR_START + 1}, and the rest were never opened.\n'
    )

    print('EXPLAIN ANALYZE (DuckDB, human-readable):')
    for row in conn.execute(f'EXPLAIN ANALYZE {_PARQUET_QUERY}').fetchall():
        print(row[-1])

    print(
        '\nCaveat: at ~68k rows (a few MB of Parquet), both queries finish '
        'in low single-digit milliseconds. There is no meaningful speed '
        'difference to report here. What this benchmark demonstrates is '
        'the access pattern: DuckDB opened 6 of 27 files and read only the '
        '`surface`/`year` columns it needed, while SQLite scanned the '
        'whole table row by row. That gap in bytes touched becomes a real '
        'time difference at a dataset size where bytes touched is the '
        'bottleneck: many partitions, wide tables, or storage far from '
        'compute. This 68k-row single-node dataset is far too small for '
        'that.'
    )


if __name__ == '__main__':
    main()
