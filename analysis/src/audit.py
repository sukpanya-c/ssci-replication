from __future__ import annotations

from typing import Any

import pandas as pd

try:
    from .config import GENRE_COLUMN, TRACK_ID_COLUMN, get_missing_required_columns
except ImportError:  # pragma: no cover - compatibility for direct module execution
    from config import GENRE_COLUMN, TRACK_ID_COLUMN, get_missing_required_columns


def audit_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize schema quality, missingness, duplicates, and genre conflicts."""

    _validate_required_columns(df)
    track_counts = df.groupby(TRACK_ID_COLUMN, sort=False).size()
    genre_counts = df.groupby(TRACK_ID_COLUMN, sort=False)[GENRE_COLUMN].nunique(dropna=True)

    return {
        "row_count": int(len(df)),
        "unique_track_id_count": int(df[TRACK_ID_COLUMN].nunique(dropna=True)),
        "duplicate_track_id_count": int((track_counts > 1).sum()),
        "multi_genre_track_id_count": int((genre_counts > 1).sum()),
        "missingness_by_column": df.isna().sum().astype(int).to_dict(),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
    }


def _validate_required_columns(df: pd.DataFrame) -> None:
    """Raise a clear error when the required columns are missing."""

    missing_columns = get_missing_required_columns(df.columns)
    if missing_columns:
        missing_list = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing_list}")
