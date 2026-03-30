from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.src.modeling import (
    build_genre_deviation_features,
    compute_within_genre_joint_test,
    create_descriptive_outputs,
    fit_all_eligible_within_genre_analysis,
    fit_genre_deviation_analysis,
    fit_main_regression_models,
    fit_robustness_models,
    fit_within_genre_analysis,
    run_grouped_predictive_check,
    run_within_genre_repeated_holdout_validation,
    run_within_genre_predictive_check,
    run_within_genre_selection_rule_robustness,
    select_market_relevant_genres,
)


def _analysis_df(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    genres = np.array(["acoustic", "rock", "pop", "jazz"])
    genre_effects = {"acoustic": 0.0, "rock": 4.5, "pop": 7.0, "jazz": 2.5}

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
            "loudness": rng.uniform(-22.0, -2.0, size=n),
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
        20.0
        + 10.0 * frame["danceability"]
        + 7.0 * frame["energy"]
        + 6.0 * frame["valence"]
        - 4.0 * frame["instrumentalness"]
        + 0.00003 * frame["duration_ms"]
        + 1.8 * frame["explicit"].astype(int)
        + frame["track_genre"].map(genre_effects)
        + rng.normal(0.0, 2.0, size=n)
    )
    frame["popularity"] = np.clip(popularity, 0.0, 100.0)
    return frame


def _within_genre_df() -> pd.DataFrame:
    rng = np.random.default_rng(23)
    genre_counts = [
        ("pop", 40),
        ("rock", 35),
        ("hip-hop", 30),
        ("jazz", 25),
        ("electronic", 22),
        ("study", 10),
    ]
    genre_intercepts = {
        "pop": 7.5,
        "rock": 6.0,
        "hip-hop": 5.5,
        "jazz": 3.0,
        "electronic": 4.5,
        "study": 1.0,
    }
    dance_slopes = {
        "pop": 7.0,
        "rock": 6.0,
        "hip-hop": 5.5,
        "jazz": 4.0,
        "electronic": 5.0,
        "study": 1.5,
    }
    energy_slopes = {
        "pop": 3.5,
        "rock": 4.5,
        "hip-hop": 5.0,
        "jazz": 2.0,
        "electronic": 4.0,
        "study": -1.0,
    }
    valence_slopes = {
        "pop": 5.0,
        "rock": 2.5,
        "hip-hop": 1.5,
        "jazz": 3.5,
        "electronic": 2.0,
        "study": 1.0,
    }

    rows: list[dict[str, float | int | bool | str]] = []
    track_index = 0
    for genre, count in genre_counts:
        for _ in range(count):
            danceability = float(rng.uniform(0.15, 0.9))
            energy = float(rng.uniform(0.1, 0.95))
            valence = float(rng.uniform(0.05, 0.95))
            acousticness = float(rng.uniform(0.0, 0.9))
            duration_ms = int(rng.integers(120_000, 300_000))
            explicit = bool(rng.integers(0, 2))
            key = int(rng.integers(0, 6))
            mode = int(rng.integers(0, 2))
            time_signature = int(rng.choice([3, 4, 5]))
            loudness = float(rng.uniform(-18.0, -3.0))
            speechiness = float(rng.uniform(0.02, 0.35))
            instrumentalness = float(rng.uniform(0.0, 0.8))
            liveness = float(rng.uniform(0.05, 0.8))
            tempo = float(rng.uniform(70.0, 180.0))
            popularity = (
                18.0
                + genre_intercepts[genre]
                + dance_slopes[genre] * danceability
                + energy_slopes[genre] * energy
                + valence_slopes[genre] * valence
                - 1.5 * acousticness
                + 0.00003 * duration_ms
                + 1.3 * int(explicit)
                + rng.normal(0.0, 4.0)
            )
            rows.append(
                {
                    "track_id": f"wg_{track_index}",
                    "popularity": float(np.clip(popularity, 0.0, 100.0)),
                    "duration_ms": duration_ms,
                    "explicit": explicit,
                    "danceability": danceability,
                    "energy": energy,
                    "key": key,
                    "loudness": loudness,
                    "mode": mode,
                    "speechiness": speechiness,
                    "acousticness": acousticness,
                    "instrumentalness": instrumentalness,
                    "liveness": liveness,
                    "valence": valence,
                    "tempo": tempo,
                    "time_signature": time_signature,
                    "track_genre": genre,
                }
            )
            track_index += 1
    return pd.DataFrame(rows)


def _within_genre_df_with_extra_eligible() -> pd.DataFrame:
    frame = _within_genre_df().copy()
    extra = frame.loc[frame["track_genre"] == "pop"].head(24).copy()
    extra["track_id"] = [f"extra_{idx}" for idx in range(len(extra))]
    extra["track_genre"] = "study"
    return pd.concat([frame, extra], ignore_index=True)


def test_create_descriptive_outputs_returns_summary_and_correlation() -> None:
    summary, correlation = create_descriptive_outputs(_analysis_df())

    assert {"variable", "mean", "std", "n_obs"} <= set(summary.columns)
    assert "popularity" in set(correlation.columns)
    assert "popularity" in set(correlation.index)


def test_fit_main_regression_models_returns_three_models() -> None:
    results = fit_main_regression_models(_analysis_df())

    assert set(results["models"]) == {"model_1", "model_2", "model_3"}
    assert results["comparison"].loc["model_3", "adj_r2"] >= results["comparison"].loc["model_2", "adj_r2"]


def test_grouped_predictive_check_has_no_track_id_leakage() -> None:
    summary = run_grouped_predictive_check(_analysis_df())

    assert set(summary["model_name"]) == {"genre_control", "genre_control_plus_affective"}
    assert summary["group_overlap_count"].eq(0).all()


def test_fit_robustness_models_returns_expected_specs() -> None:
    results = fit_robustness_models(_analysis_df())

    assert set(results["models"]) == {
        "binary_top_quartile",
        "log_duration_spec",
        "no_genre_transparency",
    }
    assert "top_quartile_popularity" in results["data"].columns
    assert "log_duration" in results["data"].columns
    assert results["comparison"].loc["binary_top_quartile", "model_type"] == "Logit"
    assert results["comparison"].loc["binary_top_quartile", "fit_metric_name"] == "mcfadden_pseudo_r2"


def test_fit_robustness_models_handles_zero_duration() -> None:
    df = _analysis_df()
    df.loc[0, "duration_ms"] = 0

    results = fit_robustness_models(df)

    assert np.isfinite(results["data"]["log_duration"]).all()


def test_modeling_module_supports_package_import() -> None:
    module = importlib.import_module("analysis.src.modeling")

    assert hasattr(module, "fit_main_regression_models")


def test_select_market_relevant_genres_prefers_pre_specified_exact_genres() -> None:
    selection = select_market_relevant_genres(_within_genre_df(), min_count=20)

    assert selection["selected_genre"].tolist() == ["pop", "rock", "hip-hop", "jazz", "electronic"]
    assert selection["selection_type"].tolist() == ["exact", "exact", "exact", "exact", "exact"]
    assert selection["n_tracks"].tolist() == [40, 35, 30, 25, 22]


def test_select_market_relevant_genres_uses_documented_substitute_when_needed() -> None:
    df = _within_genre_df().copy()
    df.loc[df["track_genre"] == "rock", "track_genre"] = "rock-n-roll"
    selection = select_market_relevant_genres(df, min_count=20)

    rock_row = selection.loc[selection["target_genre"] == "rock"].iloc[0]

    assert rock_row["selected_genre"] == "rock-n-roll"
    assert rock_row["selection_type"] == "substitute"


def test_fit_within_genre_analysis_returns_expected_models() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", PerfectSeparationWarning)
        results = fit_within_genre_analysis(_within_genre_df(), min_count=20)

    assert results["selected_genres"] == ["pop", "rock", "hip-hop", "jazz", "electronic"]
    assert int(results["selection_table"]["n_tracks"].sum()) == 152
    assert "C(track_genre) * (danceability_z + energy_z + valence_z)" in results["pooled_formula"]
    assert int(results["pooled_model"].nobs) == 152
    assert set(results["followup_models"]) == {"pop", "rock", "hip-hop", "jazz", "electronic"}
    assert results["robustness_comparison"]["model_type"] == "Logit"


def test_compute_within_genre_joint_test_returns_valid_summary() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", PerfectSeparationWarning)
        results = fit_within_genre_analysis(_within_genre_df(), min_count=20)

    summary = compute_within_genre_joint_test(results)

    assert summary["term_count"] == 12
    assert summary["test_type"] == "Wald chi-square"
    assert summary["p_value"] >= 0.0


def test_run_within_genre_predictive_check_returns_two_models_without_leakage() -> None:
    summary = run_within_genre_predictive_check(_within_genre_df(), min_count=20)

    assert set(summary["model_name"]) == {"no_interaction", "interaction_aware"}
    assert summary["group_overlap_count"].eq(0).all()


def test_run_within_genre_selection_rule_robustness_returns_threshold_rows() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", PerfectSeparationWarning)
        summary = run_within_genre_selection_rule_robustness(_within_genre_df(), thresholds=(15, 20))

    assert summary["min_count"].tolist() == [15, 20]
    assert {"sample_size", "adj_r2_gain", "wald_statistic", "wald_p_value"} <= set(summary.columns)
    assert summary["selected_genres"].str.contains("pop").all()


def test_run_within_genre_repeated_holdout_validation_returns_raw_and_summary() -> None:
    results = run_within_genre_repeated_holdout_validation(
        _within_genre_df(),
        min_count=20,
        random_states=(3, 5, 7),
    )

    raw = results["raw_results"]
    summary = results["summary"]

    assert set(raw["random_state"]) == {3, 5, 7}
    assert set(raw["model_name"]) == {"no_interaction", "interaction_aware"}
    assert raw["group_overlap_count"].eq(0).all()
    assert summary.loc[0, "n_repetitions"] == 3
    assert {"mean_test_r2_gain", "mean_rmse_reduction"} <= set(summary.columns)


def test_fit_all_eligible_within_genre_analysis_includes_non_target_eligible_genres() -> None:
    frame = _within_genre_df_with_extra_eligible()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", PerfectSeparationWarning)
        results = fit_all_eligible_within_genre_analysis(frame, min_count=20)

    assert "study" in results["selected_genres"]
    assert len(results["selected_genres"]) == 6
    assert int(results["pooled_model"].nobs) == len(frame)


def test_build_genre_deviation_features_centers_within_genre() -> None:
    df = _within_genre_df()
    selected = df.loc[df["track_genre"].isin(["pop", "rock", "hip-hop", "jazz", "electronic"])].copy()

    enriched, summary = build_genre_deviation_features(selected, ("danceability", "energy", "valence"))

    assert {"genre", "feature", "genre_mean", "genre_std", "n_obs"} <= set(summary.columns)
    for feature in ("danceability", "energy", "valence"):
        assert enriched.groupby("track_genre")[f"{feature}_dev"].mean().abs().max() < 1e-10
        assert np.allclose(enriched[f"{feature}_abs_dev"], enriched[f"{feature}_dev"].abs())


def test_fit_genre_deviation_analysis_returns_expected_models() -> None:
    results = fit_genre_deviation_analysis(_within_genre_df(), min_count=20)

    assert results["selected_genres"] == ["pop", "rock", "hip-hop", "jazz", "electronic"]
    assert "danceability_abs_dev_z" in results["main_formula"]
    assert "danceability_dev_z" in results["robustness_formula"]
    assert "danceability_z" in results["main_model"].params.index
    assert "danceability_abs_dev_z" in results["main_model"].params.index
    assert "danceability_dev_z" in results["robustness_model"].params.index
