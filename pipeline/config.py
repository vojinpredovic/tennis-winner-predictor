"""Shared path constants, resolved relative to the project root."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
RAW_CSV_PATH = RAW_DIR / 'atp_tennis.csv'
DB_PATH = DATA_DIR / 'tennis.db'

KAGGLE_DATASET = 'dissfya/atp-tennis-2000-2023daily-pull'
KAGGLE_CSV_MEMBER = 'atp_tennis.csv'
