from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .config import (
        AUDIO_FEATURE_COLUMNS,
        CONTROL_COLUMNS,
        CONTINUOUS_MAIN_COLUMNS,
        GENRE_COLUMN,
        OUTCOME_COLUMN,
        RANDOM_STATE,
        TRACK_ID_COLUMN,
        WITHIN_GENRE_EXCLUDED_GENRES,
        WITHIN_GENRE_FOCAL_FEATURES,
        WITHIN_GENRE_GENRE_SUBSTITUTES,
        WITHIN_GENRE_HOLDOUT_SEEDS,
        WITHIN_GENRE_MIN_COUNT,
        WITHIN_GENRE_THRESHOLD_GRID,
        WITHIN_GENRE_TARGET_GENRES,
    )
except ImportError:  # pragma: no cover - compatibility for direct module execution
    from config import (
        AUDIO_FEATURE_COLUMNS,
        CONTROL_COLUMNS,
        CONTINUOUS_MAIN_COLUMNS,
        GENRE_COLUMN,
        OUTCOME_COLUMN,
        RANDOM_STATE,
        TRACK_ID_COLUMN,
        WITHIN_GENRE_EXCLUDED_GENRES,
        WITHIN_GENRE_FOCAL_FEATURES,
        WITHIN_GENRE_GENRE_SUBSTITUTES,
        WITHIN_GENRE_HOLDOUT_SEEDS,
        WITHIN_GENRE_MIN_COUNT,
        WITHIN_GENRE_THRESHOLD_GRID,
        WITHIN_GENRE_TARGET_GENRES,
    )


def load_analysis_dataset(path: str | Path) -> pd.DataFrame:
    """Load a prepared analysis dataset from disk."""

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Analysis dataset not found: {dataset_path}")
    return pd.read_csv(dataset_path)


def standardize_continuous_predictors(
    df: pd.DataFrame, continuous_columns: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Add z-scored columns for continuous predictors and return their name map."""

    prepared = df.copy()
    standardized_map: dict[str, str] = {}
    for column in continuous_columns:
        standardized_name = f"{column}_z"
        column_std = prepared[column].std(ddof=0)
        if np.isclose(column_std, 0.0):
            prepared[standardized_name] = 0.0
        else:
            prepared[standardized_name] = (prepared[column] - prepared[column].mean()) / column_std
        standardized_map[column] = standardized_name
    return prepared, standardized_map


def create_descriptive_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create descriptive statistics and a correlation matrix for the main analysis variables."""

    descriptive_columns = [OUTCOME_COLUMN, *CONTROL_COLUMNS, *AUDIO_FEATURE_COLUMNS]
    summary = (
        df.loc[:, descriptive_columns]
        .agg(["mean", "std"])
        .T.reset_index()
        .rename(columns={"index": "variable", "std": "std"})
    )
    summary.insert(1, "n_obs", int(len(df)))
    sample_n = pd.DataFrame(
        [{"variable": "sample_n", "n_obs": int(len(df)), "mean": float(len(df)), "std": np.nan}]
    )
    summary = pd.concat([sample_n, summary], ignore_index=True)

    correlation_columns = [OUTCOME_COLUMN, *CONTINUOUS_MAIN_COLUMNS]
    correlation = df.loc[:, correlation_columns].corr()
    return summary, correlation


def fit_main_regression_models(df: pd.DataFrame) -> dict[str, Any]:
    """Fit the three nested OLS models for the main manuscript analysis."""

    prepared = _prepare_analysis_frame(df)
    prepared, standardized_map = standardize_continuous_predictors(prepared, CONTINUOUS_MAIN_COLUMNS)

    controls_formula = _build_controls_formula(standardized_map["duration_ms"])
    audio_formula = " + ".join(standardized_map[column] for column in AUDIO_FEATURE_COLUMNS)
    formulas = {
        "model_1": f"{OUTCOME_COLUMN} ~ {controls_formula}",
        "model_2": f"{OUTCOME_COLUMN} ~ {controls_formula} + C({GENRE_COLUMN})",
        "model_3": f"{OUTCOME_COLUMN} ~ {controls_formula} + C({GENRE_COLUMN}) + {audio_formula}",
    }
    models = {name: smf.ols(formula, data=prepared).fit(cov_type="HC1") for name, formula in formulas.items()}
    comparison = _build_ols_comparison(models)

    return {
        "data": prepared,
        "models": models,
        "formulas": formulas,
        "comparison": comparison,
        "standardized_columns": standardized_map,
    }


def run_grouped_predictive_check(df: pd.DataFrame) -> pd.DataFrame:
    """Run a compact grouped train/test predictive comparison without track leakage."""

    prepared = _prepare_analysis_frame(df)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_index, test_index = next(
        splitter.split(prepared, groups=prepared[TRACK_ID_COLUMN].astype(str))
    )
    train_df = prepared.iloc[train_index].copy()
    test_df = prepared.iloc[test_index].copy()

    train_groups = set(train_df[TRACK_ID_COLUMN].astype(str))
    test_groups = set(test_df[TRACK_ID_COLUMN].astype(str))
    group_overlap_count = len(train_groups & test_groups)

    model_specs = {
        "genre_control": {
            "numeric": ["duration_ms"],
            "binary": ["explicit"],
            "categorical": ["key", "mode", "time_signature", GENRE_COLUMN],
        },
        "genre_control_plus_affective": {
            "numeric": ["duration_ms", *AUDIO_FEATURE_COLUMNS],
            "binary": ["explicit"],
            "categorical": ["key", "mode", "time_signature", GENRE_COLUMN],
        },
    }

    rows: list[dict[str, Any]] = []
    baseline_metrics: dict[str, float] | None = None
    for model_name, spec in model_specs.items():
        pipeline = _build_predictive_pipeline(
            numeric_columns=spec["numeric"],
            binary_columns=spec["binary"],
            categorical_columns=spec["categorical"],
        )
        pipeline.fit(train_df, train_df[OUTCOME_COLUMN])
        predictions = pipeline.predict(test_df)
        rmse = float(np.sqrt(mean_squared_error(test_df[OUTCOME_COLUMN], predictions)))
        model_metrics = {
            "model_name": model_name,
            "test_r2": float(r2_score(test_df[OUTCOME_COLUMN], predictions)),
            "test_rmse": rmse,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "train_group_count": int(len(train_groups)),
            "test_group_count": int(len(test_groups)),
            "group_overlap_count": int(group_overlap_count),
            "r2_gain_vs_genre_control": 0.0,
            "rmse_reduction_vs_genre_control": 0.0,
        }
        if baseline_metrics is None:
            baseline_metrics = model_metrics
        else:
            model_metrics["r2_gain_vs_genre_control"] = model_metrics["test_r2"] - baseline_metrics["test_r2"]
            model_metrics["rmse_reduction_vs_genre_control"] = (
                baseline_metrics["test_rmse"] - model_metrics["test_rmse"]
            )
        rows.append(model_metrics)

    return pd.DataFrame(rows)


def fit_robustness_models(df: pd.DataFrame) -> dict[str, Any]:
    """Fit the narrow robustness model set for the appendix workflow."""

    prepared = _prepare_analysis_frame(df)
    prepared["top_quartile_popularity"] = (
        prepared[OUTCOME_COLUMN] >= prepared[OUTCOME_COLUMN].quantile(0.75)
    ).astype(int)
    prepared["log_duration"] = np.log(prepared["duration_ms"].clip(lower=1))

    robustness_continuous = ("duration_ms", "log_duration", "genre_count", *AUDIO_FEATURE_COLUMNS)
    prepared, standardized_map = standardize_continuous_predictors(prepared, robustness_continuous)

    controls_duration = _build_controls_formula(standardized_map["duration_ms"])
    controls_log_duration = _build_controls_formula(standardized_map["log_duration"])
    audio_formula = " + ".join(standardized_map[column] for column in AUDIO_FEATURE_COLUMNS)

    formulas = {
        "binary_top_quartile": (
            f"top_quartile_popularity ~ {controls_duration} + {audio_formula}"
        ),
        "log_duration_spec": (
            f"{OUTCOME_COLUMN} ~ {controls_log_duration} + {audio_formula}"
        ),
        "no_genre_transparency": (
            f"{OUTCOME_COLUMN} ~ {controls_duration} + {standardized_map['genre_count']} + "
            f"had_multiple_genres + {audio_formula}"
        ),
    }

    models: dict[str, Any] = {
        "binary_top_quartile": smf.logit(formulas["binary_top_quartile"], data=prepared).fit(
            disp=False, cov_type="HC1"
        ),
        "log_duration_spec": smf.ols(formulas["log_duration_spec"], data=prepared).fit(cov_type="HC1"),
        "no_genre_transparency": smf.ols(formulas["no_genre_transparency"], data=prepared).fit(cov_type="HC1"),
    }

    comparison = _build_mixed_model_comparison(models)
    return {
        "data": prepared,
        "models": models,
        "formulas": formulas,
        "comparison": comparison,
        "standardized_columns": standardized_map,
    }


def select_market_relevant_genres(
    df: pd.DataFrame,
    min_count: int = WITHIN_GENRE_MIN_COUNT,
) -> pd.DataFrame:
    """Select a fixed market-relevant genre set using thresholded exact-or-substitute rules."""

    if min_count < 1:
        raise ValueError("min_count must be at least 1.")

    prepared = _prepare_analysis_frame(df)
    counts = prepared[GENRE_COLUMN].value_counts()
    excluded = set(WITHIN_GENRE_EXCLUDED_GENRES)

    rows: list[dict[str, Any]] = []
    used_genres: set[str] = set()
    for selection_rank, target_genre in enumerate(WITHIN_GENRE_TARGET_GENRES, start=1):
        selected_genre = None
        for candidate in WITHIN_GENRE_GENRE_SUBSTITUTES[target_genre]:
            if candidate in excluded or candidate in used_genres:
                continue
            if int(counts.get(candidate, 0)) >= min_count:
                selected_genre = candidate
                break
        if selected_genre is None:
            raise ValueError(
                f"No eligible market-relevant genre found for target '{target_genre}' with min_count={min_count}."
            )
        used_genres.add(selected_genre)
        rows.append(
            {
                "target_genre": target_genre,
                "selected_genre": selected_genre,
                "n_tracks": int(counts[selected_genre]),
                "selection_type": "exact" if selected_genre == target_genre else "substitute",
                "selection_rank": selection_rank,
            }
        )
    return pd.DataFrame(rows)


def select_all_eligible_genres(
    df: pd.DataFrame,
    min_count: int = WITHIN_GENRE_MIN_COUNT,
) -> pd.DataFrame:
    """Select all genres meeting the minimum count threshold."""

    if min_count < 1:
        raise ValueError("min_count must be at least 1.")

    prepared = _prepare_analysis_frame(df)
    counts = prepared[GENRE_COLUMN].value_counts()
    eligible = counts.loc[counts >= min_count].sort_values(ascending=False, kind="mergesort")
    if eligible.empty:
        raise ValueError(f"No eligible genres found with min_count={min_count}.")

    return pd.DataFrame(
        {
            "selected_genre": eligible.index.tolist(),
            "n_tracks": eligible.astype(int).tolist(),
            "selection_type": "eligible",
            "selection_rank": list(range(1, len(eligible) + 1)),
        }
    )


def fit_within_genre_analysis(
    df: pd.DataFrame,
    min_count: int = WITHIN_GENRE_MIN_COUNT,
    focal_features: Sequence[str] = WITHIN_GENRE_FOCAL_FEATURES,
) -> dict[str, Any]:
    """Fit the within-genre heterogeneity design on the primary sample."""

    prepared = _prepare_analysis_frame(df)
    selection = select_market_relevant_genres(prepared, min_count=min_count)
    return _fit_within_genre_design(
        prepared=prepared,
        selection=selection,
        focal_features=focal_features,
        selection_rule={
            "selection_mode": "market_facing_subset",
            "min_count": min_count,
            "target_genres": WITHIN_GENRE_TARGET_GENRES,
            "excluded_genres": WITHIN_GENRE_EXCLUDED_GENRES,
        },
    )


def fit_all_eligible_within_genre_analysis(
    df: pd.DataFrame,
    min_count: int = WITHIN_GENRE_MIN_COUNT,
    focal_features: Sequence[str] = WITHIN_GENRE_FOCAL_FEATURES,
) -> dict[str, Any]:
    """Fit the within-genre heterogeneity design on all eligible genres."""

    prepared = _prepare_analysis_frame(df)
    selection = select_all_eligible_genres(prepared, min_count=min_count)
    return _fit_within_genre_design(
        prepared=prepared,
        selection=selection,
        focal_features=focal_features,
        selection_rule={
            "selection_mode": "all_eligible_genres",
            "min_count": min_count,
        },
    )


def compute_within_genre_joint_test(results: dict[str, Any]) -> dict[str, Any]:
    """Run an omnibus Wald test for the pooled within-genre interaction block."""

    model = results["pooled_model"]
    interaction_terms = [
        term
        for term in model.params.index
        if ":" in term and any(f"{feature}_z" in term for feature in results["focal_features"])
    ]
    param_names = list(model.params.index)
    restrictions = np.zeros((len(interaction_terms), len(param_names)))
    for row_index, term in enumerate(interaction_terms):
        restrictions[row_index, param_names.index(term)] = 1.0

    wald_result = model.wald_test(restrictions, use_f=False, scalar=True)
    statistic = float(np.asarray(wald_result.statistic).squeeze())
    p_value = float(np.asarray(wald_result.pvalue).squeeze())
    return {
        "test_type": "Wald chi-square",
        "statistic": statistic,
        "df": int(len(interaction_terms)),
        "p_value": p_value,
        "term_count": int(len(interaction_terms)),
    }


def run_within_genre_predictive_check(
    df: pd.DataFrame,
    min_count: int = WITHIN_GENRE_MIN_COUNT,
    focal_features: Sequence[str] = WITHIN_GENRE_FOCAL_FEATURES,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Run a compact selected-sample predictive comparison for the within-genre design."""

    prepared = _prepare_analysis_frame(df)
    selection = select_market_relevant_genres(prepared, min_count=min_count)
    selected_genres = selection["selected_genre"].tolist()
    selected = prepared.loc[prepared[GENRE_COLUMN].isin(selected_genres)].copy()
    selected = _set_formula_categories(selected, selected_genres)

    focal_features = tuple(focal_features)
    focal_formula = " + ".join(focal_features)
    no_interaction_formula = (
        f"{OUTCOME_COLUMN} ~ duration_ms + explicit + C(key) + C(mode) + "
        f"C(time_signature) + C({GENRE_COLUMN}) + {focal_formula}"
    )
    interaction_formula = (
        f"{OUTCOME_COLUMN} ~ duration_ms + explicit + C(key) + C(mode) + "
        f"C(time_signature) + C({GENRE_COLUMN}) * ({focal_formula})"
    )

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_index, test_index = next(splitter.split(selected, groups=selected[TRACK_ID_COLUMN].astype(str)))
    train_df = selected.iloc[train_index].copy()
    test_df = selected.iloc[test_index].copy()
    group_overlap_count = len(
        set(train_df[TRACK_ID_COLUMN].astype(str)) & set(test_df[TRACK_ID_COLUMN].astype(str))
    )

    rows: list[dict[str, Any]] = []
    baseline_metrics: dict[str, float] | None = None
    for model_name, formula in {
        "no_interaction": no_interaction_formula,
        "interaction_aware": interaction_formula,
    }.items():
        fitted = smf.ols(formula, data=train_df).fit()
        predictions = fitted.predict(test_df)
        rmse = float(np.sqrt(mean_squared_error(test_df[OUTCOME_COLUMN], predictions)))
        row = {
            "model_name": model_name,
            "test_r2": float(r2_score(test_df[OUTCOME_COLUMN], predictions)),
            "test_rmse": rmse,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "group_overlap_count": int(group_overlap_count),
            "r2_gain_vs_no_interaction": 0.0,
            "rmse_reduction_vs_no_interaction": 0.0,
        }
        if baseline_metrics is None:
            baseline_metrics = row
        else:
            row["r2_gain_vs_no_interaction"] = row["test_r2"] - baseline_metrics["test_r2"]
            row["rmse_reduction_vs_no_interaction"] = baseline_metrics["test_rmse"] - row["test_rmse"]
        rows.append(row)

    return pd.DataFrame(rows)


def run_within_genre_selection_rule_robustness(
    df: pd.DataFrame,
    thresholds: Sequence[int] = WITHIN_GENRE_THRESHOLD_GRID,
    focal_features: Sequence[str] = WITHIN_GENRE_FOCAL_FEATURES,
) -> pd.DataFrame:
    """Summarize how the market-facing within-genre results behave across thresholds."""

    rows: list[dict[str, Any]] = []
    for min_count in thresholds:
        results = fit_within_genre_analysis(df, min_count=min_count, focal_features=focal_features)
        joint_test = compute_within_genre_joint_test(results)
        rows.append(
            {
                "min_count": int(min_count),
                "selected_genres": " | ".join(results["selected_genres"]),
                "n_genres": int(len(results["selected_genres"])),
                "sample_size": int(results["pooled_model"].nobs),
                "no_interaction_adj_r2": float(results["no_interaction_model"].rsquared_adj),
                "interaction_adj_r2": float(results["pooled_model"].rsquared_adj),
                "adj_r2_gain": float(
                    results["pooled_model"].rsquared_adj - results["no_interaction_model"].rsquared_adj
                ),
                "wald_test_type": joint_test["test_type"],
                "wald_statistic": joint_test["statistic"],
                "wald_df": joint_test["df"],
                "wald_p_value": joint_test["p_value"],
            }
        )
    return pd.DataFrame(rows)


def run_within_genre_repeated_holdout_validation(
    df: pd.DataFrame,
    min_count: int = WITHIN_GENRE_MIN_COUNT,
    focal_features: Sequence[str] = WITHIN_GENRE_FOCAL_FEATURES,
    random_states: Sequence[int] = WITHIN_GENRE_HOLDOUT_SEEDS,
) -> dict[str, pd.DataFrame]:
    """Run repeated selected-sample holdouts for the interaction-aware predictive check."""

    raw_frames: list[pd.DataFrame] = []
    for random_state in random_states:
        summary = run_within_genre_predictive_check(
            df,
            min_count=min_count,
            focal_features=focal_features,
            random_state=random_state,
        ).copy()
        summary.insert(0, "random_state", int(random_state))
        raw_frames.append(summary)

    raw_results = pd.concat(raw_frames, ignore_index=True)
    interaction_rows = raw_results.loc[raw_results["model_name"] == "interaction_aware"].copy()
    summary = pd.DataFrame(
        [
            {
                "n_repetitions": int(len(interaction_rows)),
                "mean_test_r2_gain": float(interaction_rows["r2_gain_vs_no_interaction"].mean()),
                "std_test_r2_gain": float(interaction_rows["r2_gain_vs_no_interaction"].std(ddof=0)),
                "min_test_r2_gain": float(interaction_rows["r2_gain_vs_no_interaction"].min()),
                "max_test_r2_gain": float(interaction_rows["r2_gain_vs_no_interaction"].max()),
                "mean_rmse_reduction": float(interaction_rows["rmse_reduction_vs_no_interaction"].mean()),
                "std_rmse_reduction": float(interaction_rows["rmse_reduction_vs_no_interaction"].std(ddof=0)),
                "min_rmse_reduction": float(interaction_rows["rmse_reduction_vs_no_interaction"].min()),
                "max_rmse_reduction": float(interaction_rows["rmse_reduction_vs_no_interaction"].max()),
            }
        ]
    )
    return {"raw_results": raw_results, "summary": summary}


def build_genre_deviation_features(
    df: pd.DataFrame,
    focal_features: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add genre-profile means plus signed and absolute deviation variables."""

    prepared = _prepare_analysis_frame(df)
    enriched = prepared.copy()
    summary_frames: list[pd.DataFrame] = []
    for feature in focal_features:
        genre_profile = (
            enriched.groupby(GENRE_COLUMN, observed=False)[feature]
            .agg(genre_mean="mean", genre_std="std", n_obs="size")
            .reset_index()
            .rename(columns={GENRE_COLUMN: "genre"})
        )
        genre_profile["feature"] = feature
        summary_frames.append(genre_profile.loc[:, ["genre", "feature", "genre_mean", "genre_std", "n_obs"]])

        genre_means = enriched.groupby(GENRE_COLUMN, observed=False)[feature].transform("mean")
        enriched[f"{feature}_genre_mean"] = genre_means
        enriched[f"{feature}_dev"] = enriched[feature] - genre_means
        enriched[f"{feature}_abs_dev"] = enriched[f"{feature}_dev"].abs()

    return enriched, pd.concat(summary_frames, ignore_index=True)


def fit_genre_deviation_analysis(
    df: pd.DataFrame,
    min_count: int = WITHIN_GENRE_MIN_COUNT,
    focal_features: Sequence[str] = WITHIN_GENRE_FOCAL_FEATURES,
) -> dict[str, Any]:
    """Test whether genre-relative affective deviations are associated with popularity."""

    prepared = _prepare_analysis_frame(df)
    selection = select_market_relevant_genres(prepared, min_count=min_count)
    selected_genres = selection["selected_genre"].tolist()
    selected = prepared.loc[prepared[GENRE_COLUMN].isin(selected_genres)].copy()
    selected = _set_formula_categories(selected, selected_genres)

    focal_features = tuple(focal_features)
    selected, profile_summary = build_genre_deviation_features(selected, focal_features)
    selected, raw_standardized_map = standardize_continuous_predictors(selected, focal_features)

    abs_deviation_columns = tuple(f"{feature}_abs_dev" for feature in focal_features)
    signed_deviation_columns = tuple(f"{feature}_dev" for feature in focal_features)
    selected, abs_standardized_map = standardize_continuous_predictors(selected, abs_deviation_columns)
    selected, signed_standardized_map = standardize_continuous_predictors(selected, signed_deviation_columns)

    raw_feature_formula = " + ".join(raw_standardized_map[feature] for feature in focal_features)
    abs_deviation_formula = " + ".join(abs_standardized_map[column] for column in abs_deviation_columns)
    signed_deviation_formula = " + ".join(
        signed_standardized_map[column] for column in signed_deviation_columns
    )

    main_formula = (
        f"{OUTCOME_COLUMN} ~ duration_ms + explicit + C(key) + C(mode) + "
        f"C(time_signature) + C({GENRE_COLUMN}) + {raw_feature_formula} + {abs_deviation_formula}"
    )
    main_model = smf.ols(main_formula, data=selected).fit(cov_type="HC1")
    main_comparison = _build_mixed_model_comparison({"genre_deviation_main": main_model}).loc[
        "genre_deviation_main"
    ]

    robustness_formula = (
        f"{OUTCOME_COLUMN} ~ duration_ms + explicit + C(key) + C(mode) + "
        f"C(time_signature) + C({GENRE_COLUMN}) + {signed_deviation_formula}"
    )
    robustness_model = smf.ols(robustness_formula, data=selected).fit(cov_type="HC1")
    robustness_comparison = _build_mixed_model_comparison({"genre_deviation_robustness": robustness_model}).loc[
        "genre_deviation_robustness"
    ]

    return {
        "selection_table": selection,
        "selected_genres": selected_genres,
        "focal_features": focal_features,
        "analysis_data": selected,
        "profile_summary": profile_summary,
        "main_model": main_model,
        "main_formula": main_formula,
        "main_comparison": main_comparison,
        "robustness_model": robustness_model,
        "robustness_formula": robustness_formula,
        "robustness_comparison": robustness_comparison,
        "raw_standardized_columns": raw_standardized_map,
        "abs_deviation_standardized_columns": abs_standardized_map,
        "signed_deviation_standardized_columns": signed_standardized_map,
    }


def _fit_within_genre_design(
    prepared: pd.DataFrame,
    selection: pd.DataFrame,
    focal_features: Sequence[str],
    selection_rule: dict[str, Any],
) -> dict[str, Any]:
    """Fit the shared within-genre design for a supplied genre selection."""

    selected_genres = selection["selected_genre"].tolist()
    selected = prepared.loc[prepared[GENRE_COLUMN].isin(selected_genres)].copy()
    selected = _set_formula_categories(selected, selected_genres)

    focal_features = tuple(focal_features)
    selected, pooled_standardized_map = standardize_continuous_predictors(selected, focal_features)
    focal_formula = " + ".join(pooled_standardized_map[feature] for feature in focal_features)
    no_interaction_formula = (
        f"{OUTCOME_COLUMN} ~ duration_ms + explicit + C(key) + C(mode) + "
        f"C(time_signature) + C({GENRE_COLUMN}) + {focal_formula}"
    )
    pooled_formula = (
        f"{OUTCOME_COLUMN} ~ duration_ms + explicit + C(key) + C(mode) + C(time_signature) + "
        f"C({GENRE_COLUMN}) * ({focal_formula})"
    )
    pooled_model = smf.ols(pooled_formula, data=selected).fit(cov_type="HC1")
    no_interaction_model = smf.ols(no_interaction_formula, data=selected).fit(cov_type="HC1")

    followup_models: dict[str, Any] = {}
    followup_formulas: dict[str, str] = {}
    followup_summaries: list[dict[str, Any]] = []
    followup_standardized_columns: dict[str, dict[str, str]] = {}
    for genre in selected_genres:
        genre_df = selected.loc[selected[GENRE_COLUMN] == genre].copy()
        genre_df, genre_standardized_map = standardize_continuous_predictors(genre_df, focal_features)
        feature_formula = " + ".join(genre_standardized_map[feature] for feature in focal_features)
        formula = (
            f"{OUTCOME_COLUMN} ~ duration_ms + explicit + C(key) + C(mode) + "
            f"C(time_signature) + {feature_formula}"
        )
        model = smf.ols(formula, data=genre_df).fit(cov_type="HC1")
        followup_models[genre] = model
        followup_formulas[genre] = formula
        followup_standardized_columns[genre] = genre_standardized_map
        followup_summaries.append(
            {
                "genre": genre,
                "n_obs": int(model.nobs),
                "adj_r2": float(model.rsquared_adj),
                "aic": float(model.aic),
                "bic": float(model.bic),
            }
        )

    selected["top_quartile_popularity"] = (
        selected[OUTCOME_COLUMN] >= selected[OUTCOME_COLUMN].quantile(0.75)
    ).astype(int)
    robustness_formula = (
        f"top_quartile_popularity ~ duration_ms + explicit + C(key) + C(mode) + "
        f"C(time_signature) + C({GENRE_COLUMN}) * ({focal_formula})"
    )
    robustness_model = smf.glm(
        robustness_formula,
        data=selected,
        family=sm.families.Binomial(),
    ).fit(cov_type="HC1")
    robustness_comparison = _build_mixed_model_comparison(
        {"binary_top_quartile_interaction": robustness_model}
    ).loc["binary_top_quartile_interaction"]

    pooled_comparison = {
        "n_obs": int(pooled_model.nobs),
        "adj_r2": float(pooled_model.rsquared_adj),
        "aic": float(pooled_model.aic),
        "bic": float(pooled_model.bic),
    }

    return {
        "selection_table": selection.reset_index(drop=True),
        "selected_genres": selected_genres,
        "selection_rule": selection_rule,
        "focal_features": focal_features,
        "analysis_data": selected,
        "pooled_model": pooled_model,
        "pooled_formula": pooled_formula,
        "no_interaction_model": no_interaction_model,
        "no_interaction_formula": no_interaction_formula,
        "pooled_comparison": pooled_comparison,
        "pooled_standardized_columns": pooled_standardized_map,
        "followup_models": followup_models,
        "followup_formulas": followup_formulas,
        "followup_comparison": pd.DataFrame(followup_summaries),
        "followup_standardized_columns": followup_standardized_columns,
        "robustness_model": robustness_model,
        "robustness_formula": robustness_formula,
        "robustness_comparison": robustness_comparison,
    }


def _prepare_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize booleans and validate a frame before modeling."""

    prepared = df.copy()
    for column in ("explicit", "had_multiple_genres"):
        if column in prepared.columns:
            prepared[column] = prepared[column].astype(int)
    return prepared


def _set_formula_categories(df: pd.DataFrame, genre_categories: Sequence[str]) -> pd.DataFrame:
    """Freeze categorical levels so formula-based train/test predictions are stable."""

    prepared = df.copy()
    prepared[GENRE_COLUMN] = pd.Categorical(prepared[GENRE_COLUMN], categories=list(genre_categories), ordered=True)
    for column in ("key", "mode", "time_signature"):
        prepared[column] = pd.Categorical(
            prepared[column],
            categories=sorted(pd.Series(prepared[column]).dropna().unique().tolist()),
            ordered=True,
        )
    return prepared


def _build_controls_formula(duration_term: str) -> str:
    """Build the shared control block used in the main models."""

    return (
        f"{duration_term} + explicit + C(key) + C(mode) + C(time_signature)"
    )


def _build_ols_comparison(models: dict[str, Any]) -> pd.DataFrame:
    """Collect fit statistics for nested OLS models."""

    rows: list[dict[str, Any]] = []
    model_2_adj_r2 = models["model_2"].rsquared_adj
    for model_name, model in models.items():
        rows.append(
            {
                "model_name": model_name,
                "n_obs": int(model.nobs),
                "r2": float(model.rsquared),
                "adj_r2": float(model.rsquared_adj),
                "aic": float(model.aic),
                "bic": float(model.bic),
                "adj_r2_gain_vs_model_2": float(model.rsquared_adj - model_2_adj_r2)
                if model_name == "model_3"
                else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("model_name")


def _build_mixed_model_comparison(models: dict[str, Any]) -> pd.DataFrame:
    """Collect compact fit statistics for OLS and binomial robustness models."""

    rows: list[dict[str, Any]] = []
    for model_name, model in models.items():
        bic_value = float(model.bic_llf) if hasattr(model, "bic_llf") else float(getattr(model, "bic", np.nan))
        fit_stat = {
            "model_name": model_name,
            "n_obs": int(model.nobs),
            "aic": float(model.aic),
            "bic": bic_value,
        }
        if hasattr(model, "prsquared"):
            fit_stat["model_type"] = "Logit"
            fit_stat["fit_metric_name"] = "mcfadden_pseudo_r2"
            fit_stat["fit_metric_value"] = float(model.prsquared)
        elif getattr(getattr(model, "model", None), "family", None).__class__.__name__ == "Binomial":
            fit_stat["model_type"] = "Logit"
            fit_stat["fit_metric_name"] = "mcfadden_pseudo_r2"
            llnull = getattr(model, "llnull", np.nan)
            fit_stat["fit_metric_value"] = float(1.0 - (model.llf / llnull)) if np.isfinite(llnull) else np.nan
        elif hasattr(model, "rsquared_adj"):
            fit_stat["model_type"] = "OLS"
            fit_stat["fit_metric_name"] = "adj_r2"
            fit_stat["fit_metric_value"] = float(model.rsquared_adj)
        else:
            fit_stat["model_type"] = "Unknown"
            llnull = getattr(model, "llnull", np.nan)
            fit_stat["fit_metric_name"] = "mcfadden_pseudo_r2"
            fit_stat["fit_metric_value"] = float(1.0 - (model.llf / llnull)) if llnull else np.nan
        rows.append(fit_stat)
    return pd.DataFrame(rows).set_index("model_name")


def _build_predictive_pipeline(
    numeric_columns: Sequence[str],
    binary_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> Pipeline:
    """Build a simple preprocessing-plus-linear-regression pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(numeric_columns)),
            ("binary", "passthrough", list(binary_columns)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                list(categorical_columns),
            ),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])
