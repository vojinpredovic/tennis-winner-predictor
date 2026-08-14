# Tennis Analysis

An ETL pipeline that pulls ATP match history from Kaggle, cleans it and
engineers features, loads it into a year-partitioned Parquet dataset queried
through DuckDB, and trains a decision tree that predicts match winners from
pre-match information: player rankings, ranking points, recent form, and
match conditions (surface, court, best-of format).

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
- Load normally only touches rows at or after the dataset's latest date
  minus a 7-day trailing window — new matches plus any recent corrections
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

Types are explicit (cast in SQL when each partition is written), not
inferred from the data. `year` is used only to route rows to a partition
directory and is not itself a stored column — DuckDB's Parquet writer
doesn't strip the `PARTITION_BY` column from the files it writes the way it
does for CSV (open upstream issue as of DuckDB 1.5.5:
[duckdb/duckdb#12147](https://github.com/duckdb/duckdb/issues/12147)), so
`load.py` re-projects it away after the partitioned write, rather than
storing the year both in the file and in the directory name.

**Idempotency without a database.** SQLite gave idempotency for free via a
`UNIQUE` constraint and `ON CONFLICT`; Parquet files have no such
constraint, so re-running the pipeline can't rely on the storage layer to
reject or merge duplicates. Instead, for every year touched by an
incremental batch, `load()`:

1. Reads that year's existing partition (or treats it as empty, for a new
   year).
2. Anti-joins it against the incoming batch on the natural key, dropping
   any existing row about to be replaced.
3. Unions the result with the incoming rows for that year.
4. Rewrites `year=YYYY/` from that union, via a temp-directory-then-swap so
   a crash mid-write can't leave a partition half-overwritten.

Partitions the batch doesn't touch are never read or rewritten. Running the
same batch through `load()` twice produces the same union both times, so
the row count is stable across repeated runs — the same guarantee the old
`UNIQUE`/`ON CONFLICT` upsert gave, achieved by rewriting at the partition
grain instead of the row grain. Before any of this, `load()` asserts the
natural key is unique within the incoming batch itself and raises loudly if
it isn't (a bad transform run could otherwise silently corrupt a
partition).

Natural key: `(date, tournament, round, player_1, player_2)`. `round` is
part of the key because round-robin events (e.g. the Masters Cup) can have
the same two players meet twice on the same date within the same
tournament — once in round robin, once in the final — with different
results; without `round` those two real matches collide.

### Modeling (`analysis.py`)

Reads the Parquet dataset through DuckDB with
`read_parquet('data/matches/**/*.parquet', hive_partitioning=true)`,
selecting only the 15 columns the model actually uses — not `odd_1`/
`odd_2`, `score`, `tournament`, `series`, or `round`. Because Parquet is
columnar, a column left out of the `SELECT` is never read off disk at all;
a row-oriented store like SQLite has to read every row whole regardless of
which columns you keep. One-hot encodes `Surface`/`Court`, then does a
chronological 80/20 train/test split (train on the past, evaluate on the
future — matching how the model would actually be used). Trains a shallow
`DecisionTreeClassifier` (`max_depth=6`, `min_samples_leaf=20`) and reports
test accuracy, a naive baseline (always pick the better-ranked player), and
feature importances.

### Benchmark (`scripts/benchmark.py`)

```bash
python -m scripts.benchmark
```

Runs the same date-filtered aggregate query (match counts by surface,
2015–2020) against the SQLite table (from the `sqlite-load` branch's
output, if present at `data/tennis.db`) and the partitioned Parquet
dataset, and prints:

- Both result sets side by side (they agree, since it's the same
  underlying data), with wall-clock timings.
- DuckDB's `EXPLAIN ANALYZE` plan for the Parquet query, including how many
  of the 27 year partitions were actually opened for a 6-year filter —
  partition pruning made visible, not just asserted.
- An explicit caveat that at 68k rows both queries finish in a few
  milliseconds and there's no real performance win to report; the point is
  the access pattern (files/columns touched), which only becomes a real
  time difference at a scale this dataset doesn't reach. See "Why
  columnar" below.

### Entry point (`pipeline/run.py`)

`python -m pipeline.run [--force] [--full-refresh]` ties the three stages
together, logging row counts and elapsed time per stage, and exits non-zero
if any stage fails validation.

## Why columnar

**Row-oriented (SQLite):** a table is stored row by row on disk. Reading
`surface` and `date` for matches in 2015–2020 still means scanning every
row of the whole table and pulling every column along with it — there's no
way to read fewer columns than you store, and no way to skip years you
don't want without a secondary index.

**Columnar (Parquet):** column values are stored contiguously, so a query
that only selects `surface` and `date` reads only those columns' storage —
the other 18+ columns in the schema are never touched. `analysis.py`'s
explicit column list exploits exactly this.

**Partitioning by year:** on top of columnar storage, physically
segregating rows into `year=YYYY/` directories lets a query filtered by
year skip entire files before opening them — no metadata read, no row-group
statistics check, nothing. `scripts/benchmark.py`'s `EXPLAIN ANALYZE`
output shows this directly: a 2015–2020 filter opens 6 of 27 partition
files. This is why the load stage's idempotency mechanism is also built
around rewriting whole year partitions — it's the same unit the query
engine reasons about.

Neither of these shows up as a speed win in this repo (see the benchmark's
caveat) — at 68k rows and a few MB, everything fits comfortably in memory
either way. The benefit is real at analytical scale: many partitions, wide
tables with dozens of rarely-read columns, or storage far from compute
(cold storage, object stores) where bytes not transferred is the entire
point.

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

- The tree is intentionally shallow and simple — it's a baseline, not a
  tuned production model.
- `Odd_1` / `Odd_2` (bookmaker odds) are stored in the dataset but not used
  as model features, since they're themselves a strong prediction signal
  that could dominate/mask the other features; excluding them keeps the
  model focused on player/match stats.
- No cross-validation or hyperparameter search is performed — the split is
  a single chronological train/test cut.
- This is a single-node dataset of 68k rows read by one process on one
  machine. The DuckDB + partitioned Parquet setup here demonstrates the
  columnar/partition-pruning *access pattern* that matters at real
  analytical scale (millions of rows, many partitions, distributed
  storage) — it does not demonstrate distributed scale itself, and at this
  size a single SQLite file would perform perfectly adequately too.
