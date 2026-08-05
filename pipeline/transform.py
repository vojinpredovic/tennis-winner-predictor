"""Transform stage: clean the raw ATP matches CSV and engineer features.

Ported from the original analysis.py, kept behavior-identical. The recent-form
computation is leakage-free by construction (shift(1) before the rolling
window, so a match's own outcome never affects its own form value) — that
property must not change.

This always operates on the full raw file (not just new rows), since rolling
form needs each player's complete match history to be correct. `load.py`
decides which of the resulting rows actually get written to the database.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_EXPECTED_ROWS = 60_000
MAX_EXPECTED_ROWS = 200_000


def transform(raw_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date', kind='stable').reset_index(drop=True)

    df = _add_recent_form(df)
    df = _recode_missing_sentinel(df)

    df['Rank_diff'] = df['Rank_2'] - df['Rank_1']
    df['Pts_diff'] = df['Pts_1'] - df['Pts_2']

    _validate(df)
    logger.info(
        'Transformed %d rows (%s to %s)',
        len(df), df['Date'].min().date(), df['Date'].max().date(),
    )
    return df


def _add_recent_form(df: pd.DataFrame) -> pd.DataFrame:
    """Each player's win rate over their last 5 matches, using only results
    strictly before the current match (shift(1))."""
    p1_view = df[['Date', 'Player_1', 'Player_2', 'Winner']].rename(
        columns={'Player_1': 'Player', 'Player_2': 'Opponent'})
    p2_view = df[['Date', 'Player_1', 'Player_2', 'Winner']].rename(
        columns={'Player_1': 'Opponent', 'Player_2': 'Player'})
    p1_view['is_p1'] = True
    p2_view['is_p1'] = False

    p1_view['Won'] = (p1_view['Player'] == p1_view['Winner']).astype(int)
    p2_view['Won'] = (p2_view['Player'] == p2_view['Winner']).astype(int)

    # Keep the original df index on both halves (no ignore_index) so form can
    # be assigned straight back by index below, instead of merging on
    # Player+Date, which duplicates rows whenever a player has two matches
    # on the same date.
    long_df = pd.concat([p1_view, p2_view], axis=0)
    long_df = long_df.sort_values(['Player', 'Date'], kind='stable')
    long_df['Won_prev'] = long_df.groupby('Player')['Won'].shift(1)
    long_df['recent_form'] = (
        long_df.groupby('Player')['Won_prev']
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    form_1 = long_df.loc[long_df['is_p1'], 'recent_form']
    form_2 = long_df.loc[~long_df['is_p1'], 'recent_form']
    # A player's very first career match has no prior results,
    # treat that as neutral form.
    df['Form_1'] = form_1.reindex(df.index).fillna(0.5)
    df['Form_2'] = form_2.reindex(df.index).fillna(0.5)
    return df


def _recode_missing_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    """The source uses -1 as a missing-value sentinel for ranks/points; make
    that an explicit null rather than a bogus number a model could split on."""
    for col in ['Rank_1', 'Rank_2', 'Pts_1', 'Pts_2']:
        df[col] = df[col].replace(-1, np.nan)
    return df


def _validate(df: pd.DataFrame) -> None:
    _validate_row_count(df)
    _validate_no_nulls(df)
    _validate_winner_consistency(df)
    _validate_date_monotonic(df)
    _validate_form_range(df)
    _validate_ranks_positive(df)


def _validate_row_count(df: pd.DataFrame) -> None:
    row_count = len(df)
    if not (MIN_EXPECTED_ROWS <= row_count <= MAX_EXPECTED_ROWS):
        raise ValueError(
            f'Row count {row_count} outside expected range '
            f'[{MIN_EXPECTED_ROWS}, {MAX_EXPECTED_ROWS}]'
        )


def _validate_no_nulls(df: pd.DataFrame) -> None:
    for col in ['Date', 'Player_1', 'Player_2', 'Winner']:
        if df[col].isna().any():
            raise ValueError(f'Column {col!r} contains nulls')


def _validate_winner_consistency(df: pd.DataFrame) -> None:
    bad_winner = (
        (df['Winner'] != df['Player_1']) & (df['Winner'] != df['Player_2'])
    )
    if bad_winner.any():
        raise ValueError(
            f'{bad_winner.sum()} rows have a Winner that is neither '
            'Player_1 nor Player_2'
        )


def _validate_date_monotonic(df: pd.DataFrame) -> None:
    if not df['Date'].is_monotonic_increasing:
        raise ValueError('Date is not monotonically non-decreasing after sort')


def _validate_form_range(df: pd.DataFrame) -> None:
    for col in ['Form_1', 'Form_2']:
        if not df[col].between(0, 1).all():
            raise ValueError(f'{col} has values outside [0, 1]')


def _validate_ranks_positive(df: pd.DataFrame) -> None:
    for col in ['Rank_1', 'Rank_2', 'Pts_1', 'Pts_2']:
        valid = df[col].isna() | (df[col] > 0)
        if not valid.all():
            raise ValueError(f'{col} has non-positive, non-null values')
