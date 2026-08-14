# Tennis Analysis

An ETL pipeline that pulls ATP match history from Kaggle, cleans it and
engineers features, loads it into a year-partitioned Parquet dataset queried
through DuckDB, and trains a decision tree that predicts match winners from
pre-match information: player rankings, ranking points, recent form, and
match conditions (surface, court, best-of format).

Clone the repo, run one command, and everything reproduces from scratch,
with no manual downloads.

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

Credentials are never read, logged, or committed by this project's code.
They're picked up directly by the `kaggle` client.

## Running the pipeline

```bash
python -m pipeline.run
```

This runs **extract → transform → load** in order:

- If `data/raw/atp_tennis.csv` already exists, extract skips the download
  (pass `--force` to re-download anyway).
- Transform always re-derives features from the full raw file (rolling form
  needs each player's complete history to be correct).
- Load normally only touches rows at or after the dataset's latest date
  minus a 7-day trailing window: new matches plus any recent corrections
  the source republishes. Pass `--full-refresh` to reprocess every row.

Since the source updates daily, re-running `python -m pipeline.run`
periodically is the normal way to keep `data/matches/` current.

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
  (`shift(1)` before the rolling window), so there's no leakage from the
  match being predicted. A player's first career match defaults to a
  neutral `0.5`.
- Recodes the dataset's `-1` missing-value sentinel to a real null in
  `Rank_1`, `Rank_2`, `Pts_1`, `Pts_2`, so the model treats it as genuinely
  missing rather than a real value.
- Derives `Rank_diff` (`Rank_2 - Rank_1`) and `Pts_diff` (`Pts_1 - Pts_2`).
- Validates the result and raises if anything looks wrong: row count out of
  a sane range, nulls in key columns, a `Winner` that isn't `Player_1` or
  `Player_2`, non-monotonic dates, `Form_*` outside `[0, 1]`, or ranks/points
  that are neither null nor positive.

### Load (`pipeline/load.py`)

Writes to a Parquet dataset at `data/matches/`, partitioned Hive-style by
the match's year, via DuckDB:

```
data/matches/
  year=2000/data_0.parquet
  year=2001/data_0.parquet
  ...
  year=2026/data_0.parquet
```

| Column | Type | Notes |
| --- | --- | --- |
| `date` | DATE | |
| `tournament` | VARCHAR | |
| `series` | VARCHAR | |
| `court` | VARCHAR | Indoor / Outdoor |
| `surface` | VARCHAR | Hard / Clay / Grass / Carpet |
| `round` | VARCHAR | |
| `best_of` | INTEGER | 3 or 5 |
| `player_1`, `player_2` | VARCHAR | |
| `winner` | VARCHAR | |
| `rank_1`, `rank_2` | INTEGER | nullable, null if unranked |
| `pts_1`, `pts_2` | INTEGER | nullable, null if missing |
| `odd_1`, `odd_2` | DOUBLE | bookmaker odds |
| `score` | VARCHAR | |
| `rank_diff`, `pts_diff` | INTEGER | nullable, derived |
| `form_1`, `form_2` | DOUBLE | derived, leakage-free |

Types are explicit, cast in SQL when each partition is written. `year`
routes rows to a partition directory but is dropped before the final
write, so it isn't itself a stored column. This takes an extra step
because DuckDB's Parquet writer leaves the `PARTITION_BY` column in the
files it writes, unlike its CSV writer (open upstream issue as of DuckDB
1.5.5: [duckdb/duckdb#12147](https://github.com/duckdb/duckdb/issues/12147)).
Left alone, that would store the year twice: once in the file, once in the
directory name. `load.py` re-projects it away after the partitioned write
to avoid that.

**Idempotency without a database.** SQLite gave idempotency through a
`UNIQUE` constraint and `ON CONFLICT`. Parquet has no such constraint, so
`load()` gets the same guarantee through a per-partition rewrite. For every
year touched by an incremental batch, it:

1. Reads that year's existing partition, or treats it as empty for a new
   year.
2. Anti-joins it against the incoming batch on the natural key, dropping
   any existing row about to be replaced.
3. Unions the result with the incoming rows for that year.
4. Rewrites `year=YYYY/` from that union, using a temp-directory-then-swap
   so a crash mid-write can't leave a partition half-overwritten.

Partitions the batch doesn't touch are never read or rewritten. Running the
same batch through `load()` twice produces the same union both times, so
the row count stays stable across repeated runs. Before any of this,
`load()` asserts the natural key is unique within the incoming batch and
raises loudly if it isn't, since a bad transform run could otherwise
silently corrupt a partition.

Natural key: `(date, tournament, round, player_1, player_2)`. `round` is
part of the key because round-robin events (e.g. the Masters Cup) can have
the same two players meet twice on the same date in the same tournament,
once in round robin and once in the final, with different results.
Dropping `round` would collide those two matches into one.

### Modeling (`analysis.py`)

Reads the Parquet dataset through DuckDB with
`read_parquet('data/matches/**/*.parquet', hive_partitioning=true)`,
selecting only the 15 columns the model actually uses. It skips `odd_1`/
`odd_2`, `score`, `tournament`, `series`, and `round`. Because Parquet is
columnar, DuckDB only reads the column chunks named in the `SELECT`; a
row-oriented store like SQLite reads every column of every row regardless.
One-hot encodes `Surface`/`Court`, then does a chronological 80/20
train/test split, training on the past and evaluating on the future to
match how the model would actually be used. Trains a shallow
`DecisionTreeClassifier` (`max_depth=6`, `min_samples_leaf=20`) and reports
test accuracy, a naive baseline (always pick the better-ranked player), and
feature importances.

### Benchmark (`scripts/benchmark.py`)

```bash
python -m scripts.benchmark
```

Runs the same date-filtered aggregate query (match counts by surface,
2015-2020) against the SQLite table (from the `sqlite-load` branch's
output, at `data/tennis.db`) and the partitioned Parquet dataset, and
prints:

- Both result sets side by side, with wall-clock timings. They agree,
  since it's the same underlying data.
- DuckDB's `EXPLAIN ANALYZE` plan for the Parquet query, showing how many
  of the 27 year partitions were actually opened for a 6-year filter. This
  makes partition pruning visible directly, rather than just describing
  it.
- A caveat that at 68k rows both queries finish in a few milliseconds, so
  there's no real performance win to report. The point is the access
  pattern (files and columns touched), which only becomes a real time
  difference at a scale this dataset doesn't reach. See "Why columnar"
  below.

### Entry point (`pipeline/run.py`)

`python -m pipeline.run [--force] [--full-refresh]` ties the three stages
together, logging row counts and elapsed time per stage, and exits non-zero
if any stage fails validation.

## Why columnar

**Row-oriented (SQLite).** A table is stored row by row on disk. Reading
`surface` and `date` for matches in 2015-2020 means scanning every row of
the whole table and pulling every column along with it. Fetching fewer
columns than you store, or skipping years, requires a secondary index.

**Columnar (Parquet).** Column values are stored contiguously, so a query
that only selects `surface` and `date` reads only those two columns'
storage. `analysis.py`'s explicit column list relies on exactly this.

**Partitioning by year.** Segregating rows into `year=YYYY/` directories
lets a query filtered by year skip entire files before opening them, with
no metadata read and no row-group statistics check needed.
`scripts/benchmark.py`'s `EXPLAIN ANALYZE` output shows this directly: a
2015-2020 filter opens 6 of 27 partition files. This is also why the load
stage's idempotency mechanism rewrites whole year partitions: it's the
same unit the query engine reasons about.

This doesn't show up as a speed win in this repo (see the benchmark's
caveat), since at 68k rows and a few MB, everything fits comfortably in
memory either way. The benefit shows up at analytical scale: many
partitions, wide tables with dozens of rarely-read columns, or storage far
from compute, where the bytes not transferred are the entire point.

## Tests

```bash
python -m pytest tests/
```

Covers the parts that can silently break: that recent form never leaks the
current match's own result, that the `-1` sentinel recodes to null (not
zero or a negative number), that each transform validation actually fires
on corrupted input, that loading is idempotent (including the
same-day-rematch and corrected-result-upsert cases), that year partitions
land in the right directories with the right row counts, that the
natural-key uniqueness assertion fires on deliberately duplicated input,
and that `analysis.py`'s feature query returns the same shape and dtypes
the old SQLite-backed query did.

## Notes / limitations

- The tree is intentionally shallow and simple, a baseline rather than a
  tuned production model.
- `Odd_1` / `Odd_2` (bookmaker odds) are stored in the dataset but excluded
  from model features, since they're themselves a strong prediction signal
  that could dominate the other features. Excluding them keeps the model
  focused on player and match stats.
- No cross-validation or hyperparameter search is performed. The split is
  a single chronological train/test cut.
- This is a single-node dataset of 68k rows read by one process on one
  machine. The DuckDB and partitioned Parquet setup here demonstrates the
  columnar and partition-pruning access pattern that matters at real
  analytical scale (millions of rows, many partitions, distributed
  storage), not distributed scale itself. At this size, a single SQLite
  file would perform perfectly adequately too.
