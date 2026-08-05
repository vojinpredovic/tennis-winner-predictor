"""Extract stage: ensure the raw ATP matches CSV is present in data/raw/.

Downloads via the Kaggle API when the file isn't already there. Credentials are
resolved by the `kaggle` client itself from ~/.kaggle/kaggle.json or the
KAGGLE_USERNAME/KAGGLE_KEY environment variables — this module never reads,
logs, or touches them directly.
"""
import argparse
import logging
from pathlib import Path

from pipeline.config import KAGGLE_DATASET, RAW_CSV_PATH, RAW_DIR

logger = logging.getLogger(__name__)


def extract(force: bool = False) -> Path:
    """Ensure the raw CSV is present in data/raw/, downloading it if needed."""
    if RAW_CSV_PATH.exists() and not force:
        logger.info(
            'Raw file already present, skipping download: %s', RAW_CSV_PATH
        )
        return _log_summary(RAW_CSV_PATH)

    _download_from_kaggle()
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(
            f'Kaggle download completed but {RAW_CSV_PATH} was not found'
        )
    return _log_summary(RAW_CSV_PATH)


def _download_from_kaggle() -> None:
    # Importing `kaggle` triggers an authentication attempt as a side effect,
    # so this import is deferred until a download is actually needed.
    from kaggle.api.kaggle_api_extended import KaggleApi

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    logger.info('Downloading %s from Kaggle into %s', KAGGLE_DATASET, RAW_DIR)
    api.dataset_download_files(
        KAGGLE_DATASET, path=str(RAW_DIR), unzip=True, quiet=False
    )


def _log_summary(path: Path) -> Path:
    size_mb = path.stat().st_size / 1_000_000
    with path.open('r', encoding='utf-8', errors='replace') as f:
        row_count = sum(1 for _ in f) - 1  # exclude header
    logger.info(
        'Raw file ready: %s (%.1f MB, %d data rows)', path, size_mb, row_count
    )
    return path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    parser = argparse.ArgumentParser(
        description='Download the ATP matches dataset from Kaggle.'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Re-download even if the raw file already exists.',
    )
    args = parser.parse_args()
    extract(force=args.force)


if __name__ == '__main__':
    main()
