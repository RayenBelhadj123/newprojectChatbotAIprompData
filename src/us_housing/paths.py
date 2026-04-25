"""Shared project paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "01_raw"
DEFAULT_DATASET = RAW_DATA_DIR / "us_home_price_analysis_2004_2024.csv"
LEGACY_DATASET = DATA_DIR / "us_home_price_analysis_2004_2024.csv"


def resolve_default_dataset() -> Path:
    """Return the default dataset path, with a legacy fallback during migration."""
    if DEFAULT_DATASET.exists():
        return DEFAULT_DATASET
    return LEGACY_DATASET

