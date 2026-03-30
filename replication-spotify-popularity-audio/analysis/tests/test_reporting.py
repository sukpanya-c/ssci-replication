from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.src.modeling import fit_main_regression_models, fit_robustness_models
from analysis.src.modeling import (
    compute_within_genre_joint_test,
    fit_all_eligible_within_genre_analysis,
    fit_genre_deviation_analysis,
    fit_within_genre_analysis,
    run_within_genre_repeated_holdout_validation,
    run_within_genre_predictive_check,
    run_within_genre_selection_rule_robustness,
)
from analysis.src.reporting import (
    build_genre_selection_summary_table,
    build_genre_deviation_model_table,
    build_genre_deviation_profile_summary_table,
    build_genre_deviation_robustness_table,
    build_main_regression_table,
    build_robustness_summary_table,
    build_within_genre_joint_test_table,
    build_within_genre_followup_table,
    build_within_genre_interaction_table,
    build_within_genre_predictive_check_table,
    build_within_genre_repeated_holdout_summary_table,
    build_within_genre_scope_comparison_table,
    build_within_genre_selection_robustness_table,
    build_within_genre_robustness_table,
    export_table,
    plot_genre_deviation_effects,
    plot_main_coefficients,
    plot_within_genre_selection_robustness,
    plot_within_genre_feature_comparison,
)


def _analysis_df(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    genres = np.array(["acoustic", "rock", "pop", "jazz"])
    genre_effects = {"acoustic": 0.0, "rock": 5.0, "pop": 8.0, "jazz": 2.0}

    frame = pd.DataFrame(
        {
            "track_id": [f"track_{idx}" for idx in range(n)],
            "duration_ms": rng.integers(120_000, 300_000, size=n),
            "explicit": rng.integers(0, 2, size=n).astype(bool),
            "key": rng.integers(0, 6, size=n),
            "mode": rng.integers(0, 2, size=n),
            "time_signature": rng.choice([3, 4, 5], size=n),
            "danceability": rng.uniform(0.1, 0.9, size=n),
            "energy": rng.uniform(0.1, 0.95, size=n),
            "loudness": rng.uniform(-20.0, -3.0, size=n),
            "speechiness": rng.uniform(0.02, 0.35, size=n),
            "acousticness": rng.uniform(0.0, 0.95, size=n),
            "instrumentalness": rng.uniform(0.0, 0.8, size=n),
            "liveness": rng.uniform(0.05, 0.8, size=n),
            "valence": rng.uniform(0.05, 0.95, size=n),
            "tempo": rng.uniform(70.0, 180.0, size=n),
            "track_genre": rng.choice(genres, size=n),
        }
    )
    frame["genre_count"] = rng.integers(1, 4, size=n)
    frame["had_multiple_genres"] = frame["genre_count"] > 1

    popularity = (
        18.0
        + 9.0 * frame["danceability"]
        + 8.0 * frame["energy"]
        + 6.0 * frame["valence"]
        - 5.0 * frame["instrumentalness"]
        + 0.00004 * frame["duration_ms"]
        + 2.0 * frame["explicit"].astype(int)
        + frame["track_genre"].map(genre_effects)
        + rng.normal(0.0, 2.5, size=n)
    )
    frame["popularity"] = np.clip(popularity, 0.0, 100.0)
    return frame


def _within_genre_df() -> pd.DataFrame:
    rng = np.random.default_rng(29)
    genre_counts = [
        ("pop", 40),
        ("rock", 35),
        ("hip-hop", 30),
        ("jazz", 25),
        ("electronic", 22),
    ]
    rows: list[dict[str, float | int | bool | str]] = []
    track_index = 0
    for genre, count in genre_counts:
        for _ in range(count):
            danceability = float(rng.uniform(0.15, 0.9))
            energy = float(rng.uniform(0.1, 0.95))
            valence = float(rng.uniform(0.05, 0.95))
            duration_ms = int(rng.integers(120_000, 300_000))
            explicit = bool(rng.integers(0, 2))
            popularity = (
                25.0
                + 6.0 * danceability
                + 4.0 * energy
                + 3.0 * valence
                + (2.0 if genre == "pop" else 0.0)
                + (1.5 if genre == "rock" else 0.0)
                + rng.normal(0.0, 4.0)
            )
            rows.append(
                {
                    "track_id": f"report_{track_index}",
                    "popularity": float(np.clip(popularity, 0.0, 100.0)),
                    "duration_ms": duration_ms,
                    "explicit": explicit,
                    "danceability": danceability,
                    "energy": energy,
                    "key": int(rng.integers(0, 6)),
                    "loudness": float(rng.uniform(-18.0, -3.0)),
                    "mode": int(rng.integers(0, 2)),
                    "speechiness": float(rng.uniform(0.02, 0.35)),
                    "acousticness": float(rng.uniform(0.0, 0.9)),
                    "instrumentalness": float(rng.uniform(0.0, 0.8)),
                    "liveness": float(rng.uniform(0.05, 0.8)),
                    "valence": valence,
                    "tempo": float(rng.uniform(70.0, 180.0)),
                    "time_signature": int(rng.choice([3, 4, 5])),
                    "track_genre": genre,
                }
            )
            track_index += 1
    return pd.DataFrame(rows)


def _within_genre_df_with_extra_eligible() -> pd.DataFrame:
    frame = _within_genre_df().copy()
    extra = frame.loc[frame["track_genre"] == "pop"].head(24).copy()
    extra["track_id"] = [f"report_extra_{idx}" for idx in range(len(extra))]
    extra["track_genre"] = "study"
    return pd.concat([frame, extra], ignore_index=True)


def test_build_main_regression_table_returns_expected_columns() -> None:
    results = fit_main_regression_models(_analysis_df())
    table = build_main_regression_table(results)

    assert {"term", "model_1_coef", "model_2_coef", "model_3_coef"} <= set(table.columns)
    assert "model_3_p_value" in table.columns


def test_build_main_regression_table_keeps_numeric_estimates_and_separate_flags() -> None:
    results = fit_main_regression_models(_analysis_df())
    table = build_main_regression_table(results)

    numeric_columns = [
        "model_1_coef",
        "model_1_se",
        "model_1_p_value",
        "model_2_coef",
        "model_2_se",
        "model_2_p_value",
        "model_3_coef",
        "model_3_se",
        "model_3_p_value",
    ]

    assert "row_type" in table.columns
    assert {"model_1_flag", "model_2_flag", "model_3_flag"} <= set(table.columns)
    assert all(is_numeric_dtype(table[column]) for column in numeric_columns)

    flag_rows = table.loc[table["row_type"] == "flag"]
    assert flag_rows["model_1_coef"].isna().all()
    assert flag_rows["model_2_coef"].isna().all()
    assert flag_rows["model_3_coef"].isna().all()
    assert set(flag_rows["model_1_flag"].dropna()) == {"No"}
    assert set(flag_rows["model_2_flag"].dropna()) == {"Yes", "No"}
    assert set(flag_rows["model_3_flag"].dropna()) == {"Yes"}


def test_build_robustness_summary_table_returns_expected_columns() -> None:
    results = fit_robustness_models(_analysis_df())
    table = build_robustness_summary_table(results)

    assert {"model_name", "model_type", "term", "coefficient", "std_error", "p_value"} <= set(table.columns)
    assert table.loc[table["model_name"] == "binary_top_quartile", "model_type"].eq("Logit").all()


def test_export_table_writes_non_empty_csv(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    table = pd.DataFrame({"a": [1], "b": [2]})

    written_path = export_table(table, path)

    assert written_path.exists()
    assert written_path.stat().st_size > 0


def test_plot_main_coefficients_writes_non_empty_figure(tmp_path: Path) -> None:
    results = fit_main_regression_models(_analysis_df())
    figure_path = tmp_path / "coefficients.png"

    written_path = plot_main_coefficients(results, figure_path)

    assert written_path.exists()
    assert written_path.stat().st_size > 0


def test_reporting_module_supports_package_import() -> None:
    module = importlib.import_module("analysis.src.reporting")

    assert hasattr(module, "build_main_regression_table")


def test_within_genre_reporting_tables_return_expected_columns() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", PerfectSeparationWarning)
        results = fit_within_genre_analysis(_within_genre_df(), min_count=20)
    interaction_table = build_within_genre_interaction_table(results)
    followup_table = build_within_genre_followup_table(results)
    robustness_table = build_within_genre_robustness_table(results)

    assert {"term_group", "term", "coefficient", "std_error", "p_value"} <= set(interaction_table.columns)
    assert {"genre", "term", "coefficient", "std_error", "p_value", "q_value"} <= set(followup_table.columns)
    assert {"model_type", "term_group", "term", "coefficient", "std_error", "p_value"} <= set(
        robustness_table.columns
    )
    assert interaction_table["term_group"].eq("feature_genre_interaction").any()
    assert robustness_table["term_group"].eq("feature_genre_interaction").any()
    assert robustness_table["model_type"].eq("Logit").all()
    assert followup_table["q_value"].between(0.0, 1.0).all()


def test_plot_within_genre_feature_comparison_writes_non_empty_figure(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", PerfectSeparationWarning)
        results = fit_within_genre_analysis(_within_genre_df(), min_count=20)
    figure_path = tmp_path / "within_genre_coefficients.png"

    written_path = plot_within_genre_feature_comparison(results, figure_path)

    assert written_path.exists()
    assert written_path.stat().st_size > 0


def test_genre_deviation_reporting_outputs_expected_columns() -> None:
    results = fit_genre_deviation_analysis(_within_genre_df(), min_count=20)
    profile_table = build_genre_deviation_profile_summary_table(results)
    model_table = build_genre_deviation_model_table(results)
    robustness_table = build_genre_deviation_robustness_table(results)

    assert {"genre", "feature", "genre_mean", "genre_std", "n_obs"} <= set(profile_table.columns)
    assert {"term_group", "term", "coefficient", "std_error", "p_value"} <= set(model_table.columns)
    assert {"term_group", "term", "coefficient", "std_error", "p_value"} <= set(robustness_table.columns)
    assert model_table["term_group"].eq("absolute_deviation").any()
    assert robustness_table["term_group"].eq("signed_deviation").any()


def test_within_genre_support_tables_return_expected_columns() -> None:
    selection = pd.DataFrame(
        {
            "target_genre": ["pop", "rock"],
            "selected_genre": ["pop", "rock-n-roll"],
            "n_tracks": [40, 35],
            "selection_type": ["exact", "substitute"],
            "selection_rank": [1, 2],
        }
    )
    joint_test = {
        "test_type": "Wald chi-square",
        "statistic": 12.3,
        "df": 4,
        "p_value": 0.015,
        "term_count": 4,
    }
    predictive = pd.DataFrame(
        {
            "model_name": ["no_interaction", "interaction_aware"],
            "test_r2": [0.11, 0.13],
            "test_rmse": [10.0, 9.8],
            "group_overlap_count": [0, 0],
        }
    )

    selection_table = build_genre_selection_summary_table(selection, min_count=20)
    joint_table = build_within_genre_joint_test_table(joint_test)
    predictive_table = build_within_genre_predictive_check_table(predictive)

    assert {
        "target_genre",
        "manuscript_genre_label",
        "selected_genre",
        "threshold_rule",
        "market_rule",
    } <= set(selection_table.columns)
    assert selection_table.loc[selection_table["target_genre"] == "rock", "manuscript_genre_label"].item() == (
        "rock (dataset proxy: rock-n-roll)"
    )
    assert {"test_type", "statistic", "df", "p_value", "term_count"} <= set(joint_table.columns)
    assert {"model_name", "test_r2", "test_rmse", "group_overlap_count"} <= set(predictive_table.columns)


def test_within_genre_support_helpers_run_on_current_results() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", PerfectSeparationWarning)
        results = fit_within_genre_analysis(_within_genre_df(), min_count=20)
    joint = compute_within_genre_joint_test(results)
    predictive = run_within_genre_predictive_check(_within_genre_df(), min_count=20)

    joint_table = build_within_genre_joint_test_table(joint)
    predictive_table = build_within_genre_predictive_check_table(predictive)

    assert joint_table.loc[0, "term_count"] == 12
    assert set(predictive_table["model_name"]) == {"no_interaction", "interaction_aware"}


def test_within_genre_robustness_reporting_helpers_return_expected_columns() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", PerfectSeparationWarning)
        threshold_summary = run_within_genre_selection_rule_robustness(_within_genre_df(), thresholds=(15, 20))
        repeated = run_within_genre_repeated_holdout_validation(
            _within_genre_df(),
            min_count=20,
            random_states=(2, 4, 6),
        )
        selected_results = fit_within_genre_analysis(_within_genre_df_with_extra_eligible(), min_count=20)
        eligible_results = fit_all_eligible_within_genre_analysis(_within_genre_df_with_extra_eligible(), min_count=20)

    robustness_table = build_within_genre_selection_robustness_table(threshold_summary)
    repeated_summary = build_within_genre_repeated_holdout_summary_table(repeated["summary"])
    scope_table = build_within_genre_scope_comparison_table(selected_results, eligible_results)

    assert {"min_count", "sample_size", "adj_r2_gain", "wald_p_value"} <= set(robustness_table.columns)
    assert {"mean_test_r2_gain", "mean_rmse_reduction", "n_repetitions"} <= set(repeated_summary.columns)
    assert {"analysis_scope", "n_genres", "sample_size", "adj_r2_gain", "wald_p_value"} <= set(scope_table.columns)
    assert set(scope_table["analysis_scope"]) == {"market_facing_subset", "all_eligible_genres"}


def test_plot_within_genre_selection_robustness_writes_non_empty_figure(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", PerfectSeparationWarning)
        threshold_summary = run_within_genre_selection_rule_robustness(_within_genre_df(), thresholds=(15, 20))

    figure_path = tmp_path / "within_genre_selection_robustness.png"
    written_path = plot_within_genre_selection_robustness(threshold_summary, figure_path)

    assert written_path.exists()
    assert written_path.stat().st_size > 0


def test_plot_genre_deviation_effects_writes_non_empty_figure(tmp_path: Path) -> None:
    results = fit_genre_deviation_analysis(_within_genre_df(), min_count=20)
    figure_path = tmp_path / "genre_deviation_effects.png"

    written_path = plot_genre_deviation_effects(results, figure_path)

    assert written_path.exists()
    assert written_path.stat().st_size > 0
