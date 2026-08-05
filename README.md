# Tennis Analysis

An ETL pipeline that pulls ATP match history from Kaggle, cleans it and
engineers features, loads it into SQLite, and trains a decision tree that
predicts match winners from pre-match information: player rankings, ranking
points, recent form, and match conditions (surface, court, best-of format).

Clone the repo, run one command, and everything reproduces from scratch —
no manual downloads.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

### Kaggle credentials

The pipeline downloads
[`dissfya/atp-tennis-2000-2023daily-pull`](https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull)
(daily-updated ATP match history) via the Kaggle API. You need a Kaggle
account and API token:

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → API →
   "Create New Token". This downloads `kaggle.json`.
2. Place it at `~/.kaggle/kaggle.json` (`chmod 600` it), **or** export
   `KAGGLE_USERNAME` and `KAGGLE_KEY` as environment variables instead.

Credentials are never read, logged, or committed by this project's code —
they're picked up directly by the `kaggle` client.

## Running the pipeline

```bash
python -m pipeline.run
```

This runs **extract → transform → load** in order:

- If `data/raw/atp_tennis.csv` already exists, extract skips the download
  (pass `--force` to re-download anyway).
- Transform always re-derives features from the full raw file (rolling form
  needs each player's complete history to be correct).
- Load normally only touches rows at or after the database's latest date
  minus a 7-day trailing window — new matches plus any recent corrections
  the source republishes. Pass `--full-refresh` to reprocess every row.

Since the source updates daily, re-running `python -m pipeline.run`
periodically is the normal way to keep `data/tennis.db` current.

Then train and evaluate the model:

```bash
python analysis.py
```

```
Test accuracy: 0.643
Baseline (better rank wins) accuracy: 0.641

Feature importances:
...
```

## Pipeline stages

### Extract (`pipeline/extract.py`)

Downloads the dataset via the Kaggle API into `data/raw/atp_tennis.csv`,
skipping the download if the file is already present. Logs the file path,
size, and row count pulled.

### Transform (`pipeline/transform.py`)

Cleans the raw CSV and engineers features:

- Parses `Date` and sorts chronologically (stable sort).
- Computes each player's **recent form**: win rate over their last 5
  matches, using only results *strictly before* the current match
  (`shift(1)` before the rolling window) — no leakage from the match being
  predicted. A player's first career match defaults to a neutral `0.5`.
- Recodes the dataset's `-1` missing-value sentinel to a real null in
  `Rank_1`, `Rank_2`, `Pts_1`, `Pts_2`, so the model treats it as genuinely
  missing rather than a real value.
- Derives `Rank_diff` (`Rank_2 - Rank_1`) and `Pts_diff` (`Pts_1 - Pts_2`).
- Validates the result and raises if anything looks wrong: row count out of
  a sane range, nulls in key columns, a `Winner` that isn't `Player_1` or
  `Player_2`, non-monotonic dates, `Form_*` outside `[0, 1]`, or ranks/points
  that are neither null nor positive.

### Load (`pipeline/load.py`)

Writes to a single `matches` table in SQLite (`data/tennis.db`) via an
idempotent upsert, so re-running the pipeline never duplicates rows:

| Column | Type | Notes |
| --- | --- | --- |
| `id` | INTEGER | primary key |
| `date` | TEXT | ISO `YYYY-MM-DD` |
| `tournament` | TEXT | |
| `series` | TEXT | |
| `court` | TEXT | Indoor / Outdoor |
| `surface` | TEXT | Hard / Clay / Grass / Carpet |
| `round` | TEXT | |
| `best_of` | INTEGER | 3 or 5 |
| `player_1`, `player_2` | TEXT | |
| `winner` | TEXT | |
| `rank_1`, `rank_2` | INTEGER | null if unranked |
| `pts_1`, `pts_2` | INTEGER | null if missing |
| `odd_1`, `odd_2` | REAL | bookmaker odds |
| `score` | TEXT | |
| `rank_diff`, `pts_diff` | INTEGER | derived |
| `form_1`, `form_2` | REAL | derived, leakage-free |

Natural key: `(date, tournament, round, player_1, player_2)`, enforced with
a `UNIQUE` constraint. `round` is part of the key because round-robin events
(e.g. the Masters Cup) can have the same two players meet twice on the same
date within the same tournament — once in round robin, once in the final —
with different results; without `round` those two real matches collide.
`date` is indexed since queries filter/order by it.

### Entry point (`pipeline/run.py`)

`python -m pipeline.run [--force] [--full-refresh]` ties the three stages
together, logging row counts and elapsed time per stage, and exits non-zero
if any stage fails validation.

### Modeling (`analysis.py`)

Reads the `matches` table with a single SQL query (ordered by date), one-hot
encodes `Surface`/`Court`, and does a chronological 80/20 train/test split
(train on the past, evaluate on the future — matching how the model would
actually be used). Trains a shallow `DecisionTreeClassifier` (`max_depth=6`,
`min_samples_leaf=20`) and reports test accuracy, a naive baseline (always
pick the better-ranked player), and feature importances.

## Tests

```bash
python -m pytest tests/
```

Covers the parts that can silently break: that recent form never leaks the
current match's own result, that the `-1` sentinel recodes to null (not
zero or a negative number), that each transform validation actually fires
on corrupted input, and that loading is idempotent (including the
same-day-rematch case above).

## Notes / limitations

- The tree is intentionally shallow and simple — it's a baseline, not a
  tuned production model.
- `Odd_1` / `Odd_2` (bookmaker odds) are stored in the database but not used
  as model features, since they're themselves a strong prediction signal
  that could dominate/mask the other features; excluding them keeps the
  model focused on player/match stats.
- No cross-validation or hyperparameter search is performed — the split is
  a single chronological train/test cut.
