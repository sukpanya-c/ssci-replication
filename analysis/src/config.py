from __future__ import annotations

from pathlib import Path
from typing import Iterable

RAW_DATA_PATH = Path("dataset/spotify_popularity_audio_dataset.csv")
OUTPUT_DATA_DIR = Path("analysis/output/data")
OUTPUT_TABLE_DIR = Path("analysis/output/tables")
OUTPUT_FIGURE_DIR = Path("analysis/output/figures")
PRIMARY_EXPORT_PATH = OUTPUT_DATA_DIR / "spotify_tracks_primary_sample.csv"
ROBUSTNESS_EXPORT_PATH = OUTPUT_DATA_DIR / "spotify_tracks_robustness_sample.csv"
DESCRIPTIVE_TABLE_PATH = OUTPUT_TABLE_DIR / "descriptive_statistics.csv"
CORRELATION_MATRIX_PATH = OUTPUT_TABLE_DIR / "correlation_matrix.csv"
MAIN_REGRESSION_TABLE_PATH = OUTPUT_TABLE_DIR / "main_regression_table.csv"
PREDICTIVE_CHECK_TABLE_PATH = OUTPUT_TABLE_DIR / "predictive_check_summary.csv"
ROBUSTNESS_SUMMARY_TABLE_PATH = OUTPUT_TABLE_DIR / "robustness_summary_table.csv"
MAIN_COEFFICIENT_FIGURE_PATH = OUTPUT_FIGURE_DIR / "main_coefficient_plot.png"
GENRE_SELECTION_TABLE_PATH = OUTPUT_TABLE_DIR / "genre_selection_table.csv"
WITHIN_GENRE_INTERACTION_TABLE_PATH = OUTPUT_TABLE_DIR / "within_genre_interaction_table.csv"
WITHIN_GENRE_FOLLOWUP_TABLE_PATH = OUTPUT_TABLE_DIR / "within_genre_followup_table.csv"
WITHIN_GENRE_ROBUSTNESS_TABLE_PATH = OUTPUT_TABLE_DIR / "within_genre_robustness_table.csv"
WITHIN_GENRE_FIGURE_PATH = OUTPUT_FIGURE_DIR / "within_genre_feature_comparison.png"
WITHIN_GENRE_JOINT_TEST_TABLE_PATH = OUTPUT_TABLE_DIR / "within_genre_joint_test_table.csv"
WITHIN_GENRE_PREDICTIVE_CHECK_TABLE_PATH = OUTPUT_TABLE_DIR / "within_genre_predictive_check_table.csv"
WITHIN_GENRE_SELECTION_ROBUSTNESS_TABLE_PATH = OUTPUT_TABLE_DIR / "within_genre_selection_rule_robustness.csv"
WITHIN_GENRE_SELECTION_ROBUSTNESS_FIGURE_PATH = OUTPUT_FIGURE_DIR / "within_genre_selection_rule_robustness.png"
WITHIN_GENRE_REPEATED_HOLDOUT_SUMMARY_TABLE_PATH = (
    OUTPUT_TABLE_DIR / "within_genre_repeated_holdout_summary.csv"
)
WITHIN_GENRE_REPEATED_HOLDOUT_RAW_TABLE_PATH = OUTPUT_TABLE_DIR / "within_genre_repeated_holdout_raw.csv"
WITHIN_GENRE_ALL_ELIGIBLE_COMPARISON_TABLE_PATH = OUTPUT_TABLE_DIR / "within_genre_all_eligible_comparison.csv"
GENRE_PROFILE_SUMMARY_TABLE_PATH = OUTPUT_TABLE_DIR / "genre_profile_summary_table.csv"
GENRE_DEVIATION_MODEL_TABLE_PATH = OUTPUT_TABLE_DIR / "genre_deviation_model_table.csv"
GENRE_DEVIATION_ROBUSTNESS_TABLE_PATH = OUTPUT_TABLE_DIR / "genre_deviation_robustness_table.csv"
GENRE_DEVIATION_FIGURE_PATH = OUTPUT_FIGURE_DIR / "genre_deviation_effects.png"
JUNK_COLUMNS: tuple[str, ...] = ("Unnamed: 0.1", "Unnamed: 0")
REQUIRED_COLUMNS: tuple[str, ...] = (
    "track_id",
    "popularity",
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    "track_genre",
)
ROBUSTNESS_METADATA_COLUMNS: tuple[str, ...] = ("genre_count", "had_multiple_genres")
TRACK_ID_COLUMN = "track_id"
GENRE_COLUMN = "track_genre"
OUTCOME_COLUMN = "popularity"
CONTROL_COLUMNS: tuple[str, ...] = ("duration_ms", "explicit", "key", "mode", "time_signature")
AUDIO_FEATURE_COLUMNS: tuple[str, ...] = (
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
)
CONTINUOUS_MAIN_COLUMNS: tuple[str, ...] = ("duration_ms",) + AUDIO_FEATURE_COLUMNS
RANDOM_STATE = 42
WITHIN_GENRE_MIN_COUNT = 400
WITHIN_GENRE_THRESHOLD_GRID: tuple[int, ...] = (250, 300, 350, 400)
WITHIN_GENRE_HOLDOUT_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)
WITHIN_GENRE_FOCAL_FEATURES: tuple[str, ...] = ("danceability", "energy", "valence")
WITHIN_GENRE_TARGET_GENRES: tuple[str, ...] = ("pop", "rock", "hip-hop", "jazz", "electronic")
WITHIN_GENRE_GENRE_SUBSTITUTES: dict[str, tuple[str, ...]] = {
    "pop": ("pop", "synth-pop", "power-pop", "k-pop", "cantopop", "mandopop", "j-pop"),
    "rock": ("rock", "rock-n-roll", "psych-rock", "hard-rock", "rockabilly", "j-rock", "punk-rock"),
    "hip-hop": ("hip-hop",),
    "jazz": ("jazz",),
    "electronic": ("electronic", "dance", "dancehall", "j-dance", "club", "chicago-house", "detroit-techno"),
}
WITHIN_GENRE_EXCLUDED_GENRES: tuple[str, ...] = ("study", "sleep", "comedy")


def get_missing_required_columns(columns: Iterable[str]) -> list[str]:
    """Return required columns that are absent from a column collection."""

    column_set = set(columns)
    return [column for column in REQUIRED_COLUMNS if column not in column_set]
