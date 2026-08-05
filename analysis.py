import sqlite3

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from pipeline.config import DB_PATH

QUERY = """
SELECT
    date       AS "Date",
    surface    AS "Surface",
    court      AS "Court",
    best_of    AS "Best of",
    player_1   AS "Player_1",
    player_2   AS "Player_2",
    winner     AS "Winner",
    rank_1     AS "Rank_1",
    rank_2     AS "Rank_2",
    pts_1      AS "Pts_1",
    pts_2      AS "Pts_2",
    rank_diff  AS "Rank_diff",
    pts_diff   AS "Pts_diff",
    form_1     AS "Form_1",
    form_2     AS "Form_2"
FROM matches
ORDER BY date
"""

with sqlite3.connect(DB_PATH) as conn:
    df = pd.read_sql(QUERY, conn)

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

# Chronological split (df is already sorted by Date): train on the past,
# evaluate on the future, matching how the model would actually be used.
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
