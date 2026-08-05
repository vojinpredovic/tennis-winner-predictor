import numpy as np
import pandas as pd
import pytest

from pipeline.transform import (
    _add_recent_form,
    _recode_missing_sentinel,
    _validate_date_monotonic,
    _validate_form_range,
    _validate_no_nulls,
    _validate_ranks_positive,
    _validate_row_count,
    _validate_winner_consistency,
)


def _matches(dates, player_2, winner):
    return pd.DataFrame({
        'Date': pd.to_datetime(dates),
        'Player_1': ['A'] * len(dates),
        'Player_2': player_2,
        'Winner': winner,
    })


def test_first_match_has_neutral_form():
    df = _matches(['2024-01-01'], ['B'], ['A'])
    out = _add_recent_form(df)
    assert out.loc[0, 'Form_1'] == 0.5


def test_form_uses_only_prior_matches_not_the_current_result():
    # A's results in matches 1-5: win, loss, win, win, loss -> rolling mean 0.6
    dates = pd.date_range('2024-01-01', periods=6, freq='D')
    df = _matches(
        dates,
        player_2=['B', 'C', 'D', 'E', 'F', 'G'],
        winner=['A', 'C', 'A', 'A', 'F', 'A'],
    )
    out = _add_recent_form(df)
    assert out.loc[5, 'Form_1'] == pytest.approx(0.6)

    # Flip match 6's own outcome; the earlier matches are unchanged, so if
    # Form_1 depended on the match's own result this would differ from 0.6.
    flipped = df.copy()
    flipped.loc[5, 'Winner'] = 'G'
    out_flipped = _add_recent_form(flipped)
    assert out_flipped.loc[5, 'Form_1'] == out.loc[5, 'Form_1']


def test_recode_missing_sentinel_produces_nulls_not_zero_or_negative():
    df = pd.DataFrame({
        'Rank_1': [-1, 5, 100],
        'Rank_2': [10, -1, 50],
        'Pts_1': [-1, 200, 300],
        'Pts_2': [400, -1, 500],
    })
    out = _recode_missing_sentinel(df)

    assert out['Rank_1'].isna().tolist() == [True, False, False]
    assert out['Rank_2'].isna().tolist() == [False, True, False]
    for col in ['Rank_1', 'Rank_2', 'Pts_1', 'Pts_2']:
        non_null = out[col].dropna()
        assert (non_null > 0).all()
        assert -1 not in non_null.tolist()


def _valid_df(n=10):
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    return pd.DataFrame({
        'Date': dates,
        'Player_1': [f'P{i}' for i in range(n)],
        'Player_2': [f'Q{i}' for i in range(n)],
        'Winner': [f'P{i}' for i in range(n)],
        'Form_1': [0.5] * n,
        'Form_2': [0.5] * n,
        'Rank_1': [10.0] * n,
        'Rank_2': [20.0] * n,
        'Pts_1': [100.0] * n,
        'Pts_2': [200.0] * n,
    })


def test_validate_row_count_fires_when_too_low():
    with pytest.raises(ValueError, match='outside expected range'):
        _validate_row_count(_valid_df(10))


def test_validate_no_nulls_fires_on_missing_winner():
    df = _valid_df()
    df.loc[0, 'Winner'] = np.nan
    with pytest.raises(ValueError, match='contains nulls'):
        _validate_no_nulls(df)


def test_validate_winner_consistency_fires_on_unknown_winner():
    df = _valid_df()
    df.loc[0, 'Winner'] = 'Someone Else'
    with pytest.raises(ValueError, match='neither Player_1 nor Player_2'):
        _validate_winner_consistency(df)


def test_validate_date_monotonic_fires_on_out_of_order_dates():
    df = _valid_df()
    df.loc[0, 'Date'], df.loc[1, 'Date'] = df.loc[1, 'Date'], df.loc[0, 'Date']
    with pytest.raises(ValueError, match='monotonically non-decreasing'):
        _validate_date_monotonic(df)


def test_validate_form_range_fires_on_out_of_range_value():
    df = _valid_df()
    df.loc[0, 'Form_1'] = 1.5
    with pytest.raises(ValueError, match=r'outside \[0, 1\]'):
        _validate_form_range(df)


def test_validate_ranks_positive_fires_on_negative_rank():
    df = _valid_df()
    df.loc[0, 'Rank_1'] = -5.0
    with pytest.raises(ValueError, match='non-positive, non-null values'):
        _validate_ranks_positive(df)


def test_validate_ranks_positive_allows_null():
    df = _valid_df()
    df.loc[0, 'Rank_1'] = np.nan
    _validate_ranks_positive(df)  # should not raise
