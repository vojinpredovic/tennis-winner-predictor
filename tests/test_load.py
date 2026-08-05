import sqlite3

import pandas as pd
import pytest

from pipeline.load import load


def _sample_df(n=3):
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
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
def conn():
    connection = sqlite3.connect(':memory:')
    yield connection
    connection.close()


def test_load_is_idempotent(conn):
    df = _sample_df()

    inserted, updated = load(df, conn, full_refresh=True)
    assert (inserted, updated) == (3, 0)
    assert conn.execute('SELECT COUNT(*) FROM matches').fetchone()[0] == 3

    inserted, updated = load(df, conn, full_refresh=True)
    assert (inserted, updated) == (0, 3)
    assert conn.execute('SELECT COUNT(*) FROM matches').fetchone()[0] == 3


def test_load_upsert_updates_changed_values_in_place(conn):
    df = _sample_df()
    load(df, conn, full_refresh=True)

    changed = df.copy()
    changed.loc[0, 'Winner'] = changed.loc[0, 'Player_2']
    changed.loc[0, 'Score'] = '3-6 4-6'
    inserted, updated = load(changed, conn, full_refresh=True)

    assert (inserted, updated) == (0, 3)
    assert conn.execute('SELECT COUNT(*) FROM matches').fetchone()[0] == 3
    row = conn.execute(
        'SELECT winner, score FROM matches WHERE tournament = ?', ('Event 0',)
    ).fetchone()
    assert row == (changed.loc[0, 'Player_2'], '3-6 4-6')


def test_load_treats_same_day_rematch_as_two_distinct_rows(conn):
    # Same two players, same date, same tournament, different round -- a
    # round-robin-then-final scenario. Without `round` in the natural key
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
    assert conn.execute('SELECT COUNT(*) FROM matches').fetchone()[0] == 2
