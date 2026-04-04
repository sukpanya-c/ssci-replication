from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.src.config import MANUSCRIPT_SHARING_COLUMNS, REQUIRED_COLUMNS
from analysis.src.data_prep import (
    build_primary_sample,
    build_robustness_sample,
    clean_raw_data,
    export_dataset,
    load_raw_data,
    select_analysis_columns,
    select_manuscript_sharing_columns,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Unnamed: 0.1": [0, 1, 2, 3, 4],
            "Unnamed: 0": [10, 11, 12, 13, 14],
            "track_id": ["a", "a", "b", "c", "c"],
            "popularity": [50, 50, 40, 60, 61],
            "duration_ms": [1000, 1000, 1200, 900, 905],
            "explicit": [False, False, True, False, False],
            "danceability": [0.1, 0.1, 0.2, 0.3, 0.31],
            "energy": [0.2, 0.2, 0.3, 0.4, 0.41],
            "key": [1, 1, 2, 3, 3],
            "loudness": [-5.0, -5.0, -6.0, -7.0, -7.1],
            "mode": [1, 1, 0, 1, 1],
            "speechiness": [0.05, 0.05, 0.06, 0.07, 0.08],
            "acousticness": [0.2, 0.2, 0.3, 0.4, 0.5],
            "instrumentalness": [0.0, 0.0, 0.1, 0.2, 0.3],
            "liveness": [0.1, 0.1, 0.2, 0.3, 0.4],
            "valence": [0.2, 0.2, 0.3, 0.4, 0.5],
            "tempo": [120.0, 120.0, 121.0, 122.0, 123.0],
            "time_signature": [4, 4, 4, 4, 4],
            "track_genre": ["pop", "pop", "rock", "jazz", "blues"],
        }
    )


def test_raw_load_succeeds(tmp_path: Path) -> None:
    path = tmp_path / "tracks.csv"
    pd.DataFrame({"track_id": ["x"], "track_genre": ["pop"]}).to_csv(path, index=False)

    loaded = load_raw_data(path)

    assert isinstance(loaded, pd.DataFrame)
    assert loaded.loc[0, "track_id"] == "x"


def test_load_raw_data_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Data file not found"):
        load_raw_data(tmp_path / "missing.csv")


def test_junk_columns_are_dropped() -> None:
    cleaned = clean_raw_data(_sample_df())

    assert "Unnamed: 0.1" not in cleaned.columns
    assert "Unnamed: 0" not in cleaned.columns


def test_missing_required_columns_raise_error() -> None:
    df = _sample_df().drop(columns=["valence"])

    with pytest.raises(ValueError, match="Missing required columns"):
        clean_raw_data(df)


def test_primary_sample_is_unique_on_track_id() -> None:
    primary = build_primary_sample(clean_raw_data(_sample_df()))

    assert primary["track_id"].is_unique


def test_multi_genre_tracks_are_excluded_from_primary_sample() -> None:
    primary = build_primary_sample(clean_raw_data(_sample_df()))

    assert "c" not in set(primary["track_id"])
    assert set(primary["track_id"]) == {"a", "b"}


def test_robustness_sample_remains_unique_on_track_id() -> None:
    robustness = build_robustness_sample(clean_raw_data(_sample_df()))

    assert robustness["track_id"].is_unique
    assert set(robustness["track_id"]) == {"a", "b", "c"}


def test_select_analysis_columns_keeps_required_columns_only_in_order() -> None:
    selected = select_analysis_columns(clean_raw_data(_sample_df()))

    assert list(selected.columns) == list(REQUIRED_COLUMNS)


def test_robustness_sample_contains_all_raw_unique_track_ids() -> None:
    cleaned = clean_raw_data(_sample_df())
    robustness = build_robustness_sample(cleaned)

    assert set(robustness["track_id"]) == set(cleaned["track_id"].unique())


def test_robustness_sample_flags_multi_genre_tracks() -> None:
    robustness = build_robustness_sample(clean_raw_data(_sample_df())).set_index("track_id")

    assert bool(robustness.loc["c", "had_multiple_genres"]) is True
    assert int(robustness.loc["c", "genre_count"]) == 2


def test_robustness_sample_flags_single_genre_tracks() -> None:
    robustness = build_robustness_sample(clean_raw_data(_sample_df())).set_index("track_id")

    assert bool(robustness.loc["a", "had_multiple_genres"]) is False
    assert int(robustness.loc["a", "genre_count"]) == 1
    assert bool(robustness.loc["b", "had_multiple_genres"]) is False
    assert int(robustness.loc["b", "genre_count"]) == 1


def test_select_analysis_columns_keeps_robustness_metadata_when_present() -> None:
    selected = select_analysis_columns(build_robustness_sample(clean_raw_data(_sample_df())))

    assert list(selected.columns) == list(REQUIRED_COLUMNS) + ["genre_count", "had_multiple_genres"]


def test_select_manuscript_sharing_columns_keeps_core_columns_only() -> None:
    selected = select_manuscript_sharing_columns(clean_raw_data(_sample_df()))

    assert list(selected.columns) == list(MANUSCRIPT_SHARING_COLUMNS)


def test_select_manuscript_sharing_columns_keeps_robustness_metadata_when_present() -> None:
    selected = select_manuscript_sharing_columns(build_robustness_sample(clean_raw_data(_sample_df())))

    assert list(selected.columns) == list(MANUSCRIPT_SHARING_COLUMNS) + [
        "genre_count",
        "had_multiple_genres",
    ]


def test_select_manuscript_sharing_columns_raises_for_missing_required_core_column() -> None:
    df = clean_raw_data(_sample_df()).drop(columns=["valence"])

    with pytest.raises(ValueError, match="Missing manuscript-sharing columns"):
        select_manuscript_sharing_columns(df)


def test_export_dataset_writes_csv(tmp_path: Path) -> None:
    export_path = tmp_path / "primary.csv"
    dataset = select_analysis_columns(build_primary_sample(clean_raw_data(_sample_df())))

    written_path = export_dataset(dataset, export_path)

    assert written_path == export_path
    assert export_path.exists()
    reloaded = pd.read_csv(export_path)
    assert list(reloaded.columns) == list(REQUIRED_COLUMNS)
