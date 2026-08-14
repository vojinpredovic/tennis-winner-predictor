import duckdb
import pandas as pd
import pytest

import pipeline.load as load_module
from analysis import load_features
from pipeline.load import load

# Dtypes the old SQLite-backed query produced, verified against
# data/tennis.db before the migration. sqlite3 has no integer-with-nulls
# storage class, so anything nullable came back as float64/NaN.
# analysis.py's DuckDB query casts to match this contract even though the
# underlying Parquet storage now uses genuine nullable integers. See the
# comment above build_query() in analysis.py.
_EXPECTED_DTYPES = {
    'Surface': 'str',
    'Court': 'str',
    'Best of': 'int32',
    'Player_1': 'str',
    'Player_2': 'str',
    'Winner': 'str',
    'Rank_1': 'float64',
    'Rank_2': 'float64',
    'Pts_1': 'float64',
    'Pts_2': 'float64',
    'Rank_diff': 'float64',
    'Pts_diff': 'float64',
    'Form_1': 'float64',
    'Form_2': 'float64',
}


def _sample_df(n=5, year=2024):
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
        # Include a null rank/points to exercise the nullable-int -> DOUBLE
        # cast path, matching real data (unranked players).
        'Rank_1': [10.0, None, 30.0, 40.0, 50.0][:n],
        'Rank_2': [20.0] * n,
        'Pts_1': [100.0] * n,
        'Pts_2': [None, 200.0, 200.0, 200.0, 200.0][:n],
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
    d = tmp_path / 'matches'
    monkeypatch.setattr(load_module, 'MATCHES_DIR', d)
    return d


def test_feature_query_shape_and_dtypes_match_sqlite_version(matches_dir):
    df = _sample_df(n=5)
    with duckdb.connect() as conn:
        load(df, conn, full_refresh=True)

    features = load_features(matches_dir)

    assert len(features) == len(df)
    assert list(features.columns) == [
        'Date', 'Surface', 'Court', 'Best of', 'Player_1', 'Player_2',
        'Winner', 'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Rank_diff',
        'Pts_diff', 'Form_1', 'Form_2',
    ]
    for col, expected_dtype in _EXPECTED_DTYPES.items():
        assert str(features[col].dtype) == str(expected_dtype), (
            f'{col} dtype {features[col].dtype} != expected {expected_dtype}'
        )

    # Nulls survive the nullable-INTEGER -> DOUBLE cast as real NaNs (not
    # dropped or coerced to 0), same as the old float64/NaN SQLite columns.
    assert features['Rank_1'].isna().sum() == 1
    assert features['Pts_2'].isna().sum() == 1


def test_feature_query_is_sorted_chronologically(matches_dir):
    df = _sample_df(n=5)
    shuffled = df.sample(frac=1, random_state=0)
    with duckdb.connect() as conn:
        load(shuffled, conn, full_refresh=True)

    features = load_features(matches_dir)
    assert features['Date'].is_monotonic_increasing
