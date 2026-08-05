"""Entry point: python -m pipeline.run [--force] [--full-refresh]

Runs extract -> transform -> load in order, with per-stage timing and row
counts logged. Exits non-zero if any stage fails (e.g. a validation check
in transform.py).
"""
import argparse
import logging
import sys
import time

from pipeline.extract import extract
from pipeline.load import get_connection, load
from pipeline.transform import transform

logger = logging.getLogger(__name__)


def run(force: bool = False, full_refresh: bool = False) -> None:
    t0 = time.monotonic()
    raw_path = extract(force=force)
    logger.info('Extract finished in %.2fs', time.monotonic() - t0)

    t1 = time.monotonic()
    df = transform(raw_path)
    logger.info(
        'Transform finished in %.2fs (%d rows)',
        time.monotonic() - t1, len(df),
    )

    t2 = time.monotonic()
    conn = get_connection()
    try:
        inserted, updated = load(df, conn, full_refresh=full_refresh)
    finally:
        conn.close()
    logger.info(
        'Load finished in %.2fs (%d inserted, %d updated)',
        time.monotonic() - t2, inserted, updated,
    )

    logger.info('Pipeline finished in %.2fs total', time.monotonic() - t0)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    parser = argparse.ArgumentParser(
        description='Run the ATP matches ETL pipeline end to end.'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Re-download the raw CSV even if it already exists.',
    )
    parser.add_argument(
        '--full-refresh', action='store_true',
        help='Reload every row instead of just the incremental window.',
    )
    args = parser.parse_args()

    try:
        run(force=args.force, full_refresh=args.full_refresh)
    except Exception:
        logger.exception('Pipeline failed')
        sys.exit(1)


if __name__ == '__main__':
    main()
