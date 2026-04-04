from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .config import (
        GENRE_COLUMN,
        JUNK_COLUMNS,
        MANUSCRIPT_SHARING_COLUMNS,
        REQUIRED_COLUMNS,
        ROBUSTNESS_METADATA_COLUMNS,
        TRACK_ID_COLUMN,
        get_missing_required_columns,
    )
except ImportError:  # pragma: no cover - compatibility for direct module execution
    from config import (
        GENRE_COLUMN,
        JUNK_COLUMNS,
        MANUSCRIPT_SHARING_COLUMNS,
        REQUIRED_COLUMNS,
        ROBUSTNESS_METADATA_COLUMNS,
        TRACK_ID_COLUMN,
        get_missing_required_columns,
    )


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the raw Spotify tracks CSV from disk."""

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    return pd.read_csv(data_path)


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop junk index columns and validate the required schema."""

    cleaned = df.drop(columns=[column for column in JUNK_COLUMNS if column in df.columns]).copy()
    _validate_required_columns(cleaned)
    return cleaned.reset_index(drop=True)


def build_primary_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Build the main sample with one non-ambiguous genre label per track."""

    prepared = _prepare_for_sampling(df)
    genre_counts = prepared.groupby(TRACK_ID_COLUMN, sort=False)[GENRE_COLUMN].nunique(dropna=True)
    ambiguous_track_ids = genre_counts[genre_counts > 1].index
    primary = prepared.loc[~prepared[TRACK_ID_COLUMN].isin(ambiguous_track_ids)].copy()
    primary = primary.drop_duplicates(subset=TRACK_ID_COLUMN, keep="first")
    return primary.reset_index(drop=True)


def build_robustness_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Build a one-row-per-track sample with explicit genre ambiguity metadata."""

    prepared = _prepare_for_sampling(df)
    robustness = prepared.drop_duplicates(subset=TRACK_ID_COLUMN, keep="first")
    genre_counts = (
        prepared.groupby(TRACK_ID_COLUMN, sort=False)[GENRE_COLUMN]
        .nunique(dropna=True)
        .rename("genre_count")
        .astype(int)
    )
    robustness = robustness.merge(genre_counts, on=TRACK_ID_COLUMN, how="left", validate="one_to_one")
    robustness["had_multiple_genres"] = robustness["genre_count"].gt(1)
    return robustness.reset_index(drop=True)


def select_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the required analysis columns in a stable order."""

    _validate_required_columns(df)
    selected_columns = list(REQUIRED_COLUMNS) + [
        column for column in ROBUSTNESS_METADATA_COLUMNS if column in df.columns
    ]
    return df.loc[:, selected_columns].copy()


def select_manuscript_sharing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the manuscript-facing subset of processed analysis columns."""

    missing_columns = [column for column in MANUSCRIPT_SHARING_COLUMNS if column not in df.columns]
    if missing_columns:
        missing_list = ", ".join(missing_columns)
        raise ValueError(f"Missing manuscript-sharing columns: {missing_list}")

    selected_columns = list(MANUSCRIPT_SHARING_COLUMNS) + [
        column for column in ROBUSTNESS_METADATA_COLUMNS if column in df.columns
    ]
    return df.loc[:, selected_columns].copy()


def export_dataset(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a prepared dataset to CSV and return the destination path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def _prepare_for_sampling(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize input before sample construction."""

    prepared = clean_raw_data(df)
    if prepared[TRACK_ID_COLUMN].isna().any():
        raise ValueError("Column 'track_id' contains missing values.")
    return prepared


def _validate_required_columns(df: pd.DataFrame) -> None:
    """Raise a clear error when the required columns are missing."""

    missing_columns = get_missing_required_columns(df.columns)
    if missing_columns:
        missing_list = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing_list}")
