import duckdb
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from pipeline.config import MATCHES_DIR

# Only the columns the model reads below -- not odd_1/odd_2, score,
# tournament, series, or round. Parquet is columnar, so a column that isn't
# named here is never read off disk at all: DuckDB only opens the column
# chunks for the columns listed in the SELECT, unlike a row store (SQLite)
# where every row is read whole regardless of which columns you keep.
#
# rank_1/rank_2/pts_1/pts_2/rank_diff/pts_diff are stored as nullable
# INTEGER (see pipeline/load.py) -- correct for storage, but read straight
# through, DuckDB hands nullable integers back to pandas as nullable Int32
# (pd.NA) rather than the float64/NaN the old SQLite-backed query produced.
# Cast them to DOUBLE here so the DataFrame handed to sklearn has the same
# dtypes as before; the nullable-int storage type is unaffected.
def build_query(matches_dir=MATCHES_DIR) -> str:
    return f"""
    SELECT
        date               AS "Date",
        surface            AS "Surface",
        court              AS "Court",
        best_of            AS "Best of",
        player_1           AS "Player_1",
        player_2           AS "Player_2",
        winner             AS "Winner",
        CAST(rank_1     AS DOUBLE) AS "Rank_1",
        CAST(rank_2     AS DOUBLE) AS "Rank_2",
        CAST(pts_1      AS DOUBLE) AS "Pts_1",
        CAST(pts_2      AS DOUBLE) AS "Pts_2",
        CAST(rank_diff  AS DOUBLE) AS "Rank_diff",
        CAST(pts_diff   AS DOUBLE) AS "Pts_diff",
        form_1             AS "Form_1",
        form_2             AS "Form_2"
    FROM read_parquet('{matches_dir}/**/*.parquet', hive_partitioning=true)
    ORDER BY "Date"
    """


def load_features(matches_dir=MATCHES_DIR) -> pd.DataFrame:
    with duckdb.connect() as conn:
        return conn.execute(build_query(matches_dir)).df()


def main() -> None:
    df = load_features()
    df = pd.get_dummies(df, columns=['Surface', 'Court'])

    target = (df['Winner'] == df['Player_1']).astype(int)

    dummy_cols = [
        c for c in df.columns
        if c.startswith('Surface_') or c.startswith('Court_')
    ]
    feature_cols = dummy_cols + [
        'Best of', 'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2',
        'Rank_diff', 'Pts_diff', 'Form_1', 'Form_2',
    ]
    X = df[feature_cols]
    y = target

    # Chronological split (df is already sorted by Date): train on the
    # past, evaluate on the future, matching how the model would actually
    # be used.
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=20, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f'Test accuracy: {accuracy_score(y_test, preds):.3f}')

    # Baseline: naively pick the better-ranked (lower Rank number) player.
    rank_baseline = (X_test['Rank_1'] < X_test['Rank_2']).astype(int)
    print(
        'Baseline (better rank wins) accuracy: '
        f'{accuracy_score(y_test, rank_baseline):.3f}'
    )

    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print('\nFeature importances:')
    print(importances)


if __name__ == '__main__':
    main()
