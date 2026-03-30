from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.src.audit import audit_dataset
from analysis.src.data_prep import clean_raw_data


def test_audit_dictionary_contains_required_summary_keys() -> None:
    df = pd.DataFrame(
        {
            "Unnamed: 0.1": [0, 1, 2],
            "Unnamed: 0": [10, 11, 12],
            "track_id": ["a", "a", "b"],
            "popularity": [50, 51, 52],
            "duration_ms": [1000, 1001, 1002],
            "explicit": [False, False, True],
            "danceability": [0.1, 0.2, 0.3],
            "energy": [0.2, 0.3, 0.4],
            "key": [1, 1, 2],
            "loudness": [-5.0, -5.1, -6.0],
            "mode": [1, 1, 0],
            "speechiness": [0.05, 0.06, 0.07],
            "acousticness": [0.2, 0.3, 0.4],
            "instrumentalness": [0.0, 0.1, 0.2],
            "liveness": [0.1, 0.2, 0.3],
            "valence": [0.2, 0.3, 0.4],
            "tempo": [120.0, 121.0, 122.0],
            "time_signature": [4, 4, 4],
            "track_genre": ["pop", "rock", "jazz"],
        }
    )

    summary = audit_dataset(clean_raw_data(df))

    assert {
        "row_count",
        "unique_track_id_count",
        "duplicate_track_id_count",
        "multi_genre_track_id_count",
        "missingness_by_column",
        "dtypes",
    } <= set(summary)

    assert summary["row_count"] == 3
    assert summary["unique_track_id_count"] == 2
    assert summary["duplicate_track_id_count"] == 1
    assert summary["multi_genre_track_id_count"] == 1
