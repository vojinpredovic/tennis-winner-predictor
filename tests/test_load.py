import duckdb
import pandas as pd
import pytest

import pipeline.load as load_module
from pipeline.load import load


def _sample_df(n=3, year=2024):
    dates = pd.date_range(f'{year}-01-01', periods=n, freq='D')
    return pd.DataFrame({
        'Date': dates,
        'Tournament': [f'Event {i}' for i in range(n)],
        'Series': ['ATP250'] * n,
        'Court': ['Outdoor'] * n,
        'Surface': ['Hard'] * n,
        'Round': ['1st Round'] * n,
        'Best of': [3] * n,
        'Player_1': [f'Player {i}A' for i in range(n)],
        'Player_2': [f'Player {i}B' for i in range(n)],
        'Winner': [f'Player {i}A' for i in range(n)],
        'Rank_1': [10.0] * n,
        'Rank_2': [20.0] * n,
        'Pts_1': [100.0] * n,
        'Pts_2': [200.0] * n,
        'Odd_1': [1.5] * n,
        'Odd_2': [2.5] * n,
        'Score': ['6-3 6-4'] * n,
        'Rank_diff': [10.0] * n,
        'Pts_diff': [-100.0] * n,
        'Form_1': [0.5] * n,
        'Form_2': [0.5] * n,
    })


@pytest.fixture
def matches_dir(tmp_path, monkeypatch):
    """Point pipeline.load at a scratch directory instead of data/matches."""
    d = tmp_path / 'matches'
    monkeypatch.setattr(load_module, 'MATCHES_DIR', d)
    return d


@pytest.fixture
def conn():
    connection = duckdb.connect()
    yield connection
    connection.close()


def _row_count(matches_dir) -> int:
    if not any(matches_dir.glob('year=*/*.parquet')):
        return 0
    pattern = str(matches_dir / '**' / '*.parquet')
    with duckdb.connect() as c:
        return c.execute(
            f"SELECT COUNT(*) FROM read_parquet('{pattern}', "
            'hive_partitioning=true)'
        ).fetchone()[0]


def _partition_rows(matches_dir, year) -> int:
    pattern = str(matches_dir / f'year={year}' / '*.parquet')
    with duckdb.connect() as c:
        return c.execute(
            f"SELECT COUNT(*) FROM read_parquet('{pattern}')"
        ).fetchone()[0]


def test_load_is_idempotent(matches_dir, conn):
    df = _sample_df()

    inserted, updated = load(df, conn, full_refresh=True)
    assert (inserted, updated) == (3, 0)
    assert _row_count(matches_dir) == 3

    inserted, updated = load(df, conn, full_refresh=True)
    assert (inserted, updated) == (0, 3)
    assert _row_count(matches_dir) == 3


def test_partitions_created_per_year_with_expected_row_counts(
    matches_dir, conn
):
    df = pd.concat(
        [_sample_df(n=2, year=2022), _sample_df(n=3, year=2023)],
        ignore_index=True,
    )

    load(df, conn, full_refresh=True)

    assert (matches_dir / 'year=2022').is_dir()
    assert (matches_dir / 'year=2023').is_dir()
    assert not (matches_dir / 'year=2024').exists()
    assert _partition_rows(matches_dir, 2022) == 2
    assert _partition_rows(matches_dir, 2023) == 3
    assert _row_count(matches_dir) == 5


def test_load_upsert_updates_changed_values_in_place(matches_dir, conn):
    df = _sample_df()
    load(df, conn, full_refresh=True)

    changed = df.copy()
    changed.loc[0, 'Winner'] = changed.loc[0, 'Player_2']
    changed.loc[0, 'Score'] = '3-6 4-6'
    inserted, updated = load(changed, conn, full_refresh=True)

    assert (inserted, updated) == (0, 3)
    assert _row_count(matches_dir) == 3

    pattern = str(matches_dir / '**' / '*.parquet')
    row = conn.execute(
        f"SELECT winner, score FROM read_parquet('{pattern}', "
        "hive_partitioning=true) WHERE tournament = 'Event 0'"
    ).fetchone()
    assert row == (changed.loc[0, 'Player_2'], '3-6 4-6')


def test_natural_key_uniqueness_assertion_fires_on_duplicate_input(
    matches_dir, conn
):
    df = _sample_df(n=2)
    # Force a natural-key collision: same date/tournament/round/players.
    df.loc[1, ['Date', 'Tournament', 'Round', 'Player_1', 'Player_2']] = (
        df.loc[0, ['Date', 'Tournament', 'Round', 'Player_1', 'Player_2']]
    )

    with pytest.raises(ValueError, match='duplicate natural key'):
        load(df, conn, full_refresh=True)

    assert _row_count(matches_dir) == 0


def test_load_treats_same_day_rematch_as_two_distinct_rows(
    matches_dir, conn
):
    # Same two players, same date, same tournament, different round: a
    # round-robin-then-final scenario. Without `round` in the natural key,
    # these would collide and one match would silently overwrite the other.
    df = pd.DataFrame({
        'Date': pd.to_datetime(['2024-01-01', '2024-01-01']),
        'Tournament': ['Masters Cup', 'Masters Cup'],
        'Series': ['Masters Cup'] * 2,
        'Court': ['Indoor'] * 2,
        'Surface': ['Hard'] * 2,
        'Round': ['Round Robin', 'The Final'],
        'Best of': [3, 5],
        'Player_1': ['A', 'A'],
        'Player_2': ['B', 'B'],
        'Winner': ['A', 'B'],
        'Rank_1': [1.0, 1.0],
        'Rank_2': [2.0, 2.0],
        'Pts_1': [100.0, 100.0],
        'Pts_2': [90.0, 90.0],
        'Odd_1': [1.5, 1.5],
        'Odd_2': [2.5, 2.5],
        'Score': ['6-3 6-4', '4-6 4-6'],
        'Rank_diff': [1.0, 1.0],
        'Pts_diff': [10.0, 10.0],
        'Form_1': [0.5, 0.5],
        'Form_2': [0.5, 0.5],
    })

    inserted, updated = load(df, conn, full_refresh=True)
    assert (inserted, updated) == (2, 0)
    assert _row_count(matches_dir) == 2
