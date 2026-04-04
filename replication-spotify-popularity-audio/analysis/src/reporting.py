from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection

try:
    from .config import AUDIO_FEATURE_COLUMNS, GENRE_COLUMN, WITHIN_GENRE_FOCAL_FEATURES
except ImportError:  # pragma: no cover - compatibility for direct module execution
    from config import AUDIO_FEATURE_COLUMNS, GENRE_COLUMN, WITHIN_GENRE_FOCAL_FEATURES

matplotlib.use("Agg")

TERM_LABELS = {
    "duration_ms_z": "Duration (z)",
    "log_duration_z": "Log duration (z)",
    "explicit": "Explicit",
    "genre_count_z": "Genre count (z)",
    "had_multiple_genres": "Had multiple genres",
    "danceability_z": "Dance orientation (z)",
    "energy_z": "Sonic intensity (z)",
    "loudness_z": "Loudness (z)",
    "speechiness_z": "Speechiness (z)",
    "acousticness_z": "Acousticness (z)",
    "instrumentalness_z": "Instrumentalness (z)",
    "liveness_z": "Liveness (z)",
    "valence_z": "Positive affect (z)",
    "tempo_z": "Tempo (z)",
    "danceability_dev_z": "Dance orientation deviation (z)",
    "energy_dev_z": "Sonic intensity deviation (z)",
    "valence_dev_z": "Positive affect deviation (z)",
    "danceability_abs_dev_z": "Absolute deviation in dance orientation (z)",
    "energy_abs_dev_z": "Absolute deviation in sonic intensity (z)",
    "valence_abs_dev_z": "Absolute deviation in positive affect (z)",
}


def build_main_regression_table(results: dict[str, Any]) -> pd.DataFrame:
    """Build a manuscript-facing regression table for the three nested OLS models."""

    models = results["models"]
    comparison = results["comparison"]
    terms = _collect_terms(models, exclude_genre_fixed_effects=True)

    rows: list[dict[str, Any]] = []
    for term in terms:
        row = {
            "term": _label_term(term),
            "row_type": "estimate",
            "model_1_flag": np.nan,
            "model_2_flag": np.nan,
            "model_3_flag": np.nan,
        }
        for model_name, model in models.items():
            row[f"{model_name}_coef"] = model.params.get(term, float("nan"))
            row[f"{model_name}_se"] = model.bse.get(term, float("nan"))
            row[f"{model_name}_p_value"] = model.pvalues.get(term, float("nan"))
        rows.append(row)

    rows.extend(
        [
            _summary_row("N", comparison, field="n_obs"),
            _summary_row("Adjusted R^2", comparison, field="adj_r2"),
            _summary_row("Adj. R^2 gain vs Model 2", comparison, field="adj_r2_gain_vs_model_2"),
            _flag_row("Genre fixed effects", {"model_1": "No", "model_2": "Yes", "model_3": "Yes"}),
            _flag_row(
                "Affective/audio predictors",
                {"model_1": "No", "model_2": "No", "model_3": "Yes"},
            ),
        ]
    )
    return pd.DataFrame(rows)


def build_robustness_summary_table(results: dict[str, Any]) -> pd.DataFrame:
    """Build a compact long-form robustness summary table."""

    rows: list[dict[str, Any]] = []
    comparison = results["comparison"]
    for model_name, model in results["models"].items():
        fit_metric_name = comparison.loc[model_name, "fit_metric_name"]
        fit_metric_value = comparison.loc[model_name, "fit_metric_value"]
        model_type = comparison.loc[model_name, "model_type"]
        for term in _collect_terms({model_name: model}, exclude_genre_fixed_effects=True):
            rows.append(
                {
                    "model_name": model_name,
                    "model_type": model_type,
                    "term": _label_term(term),
                    "coefficient": model.params.get(term, float("nan")),
                    "std_error": model.bse.get(term, float("nan")),
                    "p_value": model.pvalues.get(term, float("nan")),
                    "fit_metric_name": fit_metric_name,
                    "fit_metric_value": fit_metric_value,
                    "n_obs": comparison.loc[model_name, "n_obs"],
                }
            )
    return pd.DataFrame(rows)


def export_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a table to CSV and return the destination path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def plot_main_coefficients(results: dict[str, Any], path: str | Path) -> Path:
    """Plot Model 3 affective/audio coefficients with 95% confidence intervals."""

    model = results["models"]["model_3"]
    plot_rows = []
    for column in AUDIO_FEATURE_COLUMNS:
        term = f"{column}_z"
        plot_rows.append(
            {
                "term": _label_term(term),
                "coefficient": float(model.params[term]),
                "lower": float(model.conf_int().loc[term, 0]),
                "upper": float(model.conf_int().loc[term, 1]),
            }
        )
    plot_df = pd.DataFrame(plot_rows).sort_values("coefficient")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(
        plot_df["coefficient"],
        plot_df["term"],
        xerr=[
            plot_df["coefficient"] - plot_df["lower"],
            plot_df["upper"] - plot_df["coefficient"],
        ],
        fmt="o",
        color="#1f4e79",
        ecolor="#7a99b8",
        capsize=3,
    )
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("")
    ax.set_title("Model 3 affective/audio coefficients")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_within_genre_interaction_table(results: dict[str, Any]) -> pd.DataFrame:
    """Build a compact pooled interaction table for the within-genre design."""

    model = results["pooled_model"]
    pooled_comparison = results["pooled_comparison"]
    selected_genres = results["selected_genres"]
    focal_columns = [f"{feature}_z" for feature in results["focal_features"]]

    rows: list[dict[str, Any]] = []
    for term in model.params.index:
        if term == "Intercept":
            continue
        if term.startswith(f"C({GENRE_COLUMN})[T.") and ":" not in term:
            continue
        if term in focal_columns:
            term_group = "focal_main_effect"
        elif any(_is_feature_interaction(term, feature) for feature in focal_columns):
            term_group = "feature_genre_interaction"
        else:
            term_group = "control"
        rows.append(
            {
                "term_group": term_group,
                "term": _label_term(term),
                "coefficient": float(model.params[term]),
                "std_error": float(model.bse[term]),
                "p_value": float(model.pvalues[term]),
                "n_obs": pooled_comparison["n_obs"],
                "adj_r2": pooled_comparison["adj_r2"],
                "selected_genres": " | ".join(selected_genres),
            }
        )
    return pd.DataFrame(rows)


def build_within_genre_followup_table(results: dict[str, Any]) -> pd.DataFrame:
    """Build a long-form table of genre-specific focal-feature coefficients."""

    comparison = results["followup_comparison"].set_index("genre")
    focal_terms = [f"{feature}_z" for feature in results["focal_features"]]

    rows: list[dict[str, Any]] = []
    for genre, model in results["followup_models"].items():
        for term in focal_terms:
            rows.append(
                {
                    "genre": genre,
                    "term": _label_term(term),
                    "coefficient": float(model.params[term]),
                    "std_error": float(model.bse[term]),
                    "p_value": float(model.pvalues[term]),
                    "n_obs": float(comparison.loc[genre, "n_obs"]),
                    "adj_r2": float(comparison.loc[genre, "adj_r2"]),
                }
            )
    table = pd.DataFrame(rows)
    rejected, q_values = fdrcorrection(table["p_value"].to_numpy())
    table["q_value"] = q_values
    table["fdr_reject"] = rejected
    return table


def build_within_genre_robustness_table(results: dict[str, Any]) -> pd.DataFrame:
    """Build a compact table for the binary interaction robustness check."""

    model = results["robustness_model"]
    comparison = results["robustness_comparison"]
    focal_columns = [f"{feature}_z" for feature in results["focal_features"]]

    rows: list[dict[str, Any]] = []
    for term in model.params.index:
        if term == "Intercept":
            continue
        if term.startswith(f"C({GENRE_COLUMN})[T.") and ":" not in term:
            continue
        if term in focal_columns:
            term_group = "focal_main_effect"
        elif any(_is_feature_interaction(term, feature) for feature in focal_columns):
            term_group = "feature_genre_interaction"
        else:
            term_group = "control"
        rows.append(
            {
                "model_type": comparison["model_type"],
                "term_group": term_group,
                "term": _label_term(term),
                "coefficient": float(model.params[term]),
                "std_error": float(model.bse[term]),
                "p_value": float(model.pvalues[term]),
                "fit_metric_name": comparison["fit_metric_name"],
                "fit_metric_value": float(comparison["fit_metric_value"]),
                "n_obs": float(comparison["n_obs"]),
            }
        )
    return pd.DataFrame(rows)


def build_genre_selection_summary_table(selection: pd.DataFrame, min_count: int) -> pd.DataFrame:
    """Build a manuscript-facing summary of the selected market-relevant genres."""

    selection_table = selection.copy()
    if "selection_rank" in selection_table.columns:
        selection_table = selection_table.sort_values("selection_rank").reset_index(drop=True)

    selection_table["manuscript_genre_label"] = selection_table.apply(
        lambda row: _format_manuscript_genre_label(
            target_genre=str(row["target_genre"]),
            selected_genre=str(row["selected_genre"]),
        ),
        axis=1,
    )
    selection_table["threshold_rule"] = f"minimum n >= {min_count}"
    selection_table["market_rule"] = "market-relevant genre screen with explicit threshold"
    selection_table["substitute_note"] = selection_table.apply(
        lambda row: _format_substitute_note(
            target_genre=str(row["target_genre"]),
            selected_genre=str(row["selected_genre"]),
            selection_type=str(row["selection_type"]),
        ),
        axis=1,
    )
    return selection_table.loc[
        :,
        [
            "selection_rank",
            "target_genre",
            "manuscript_genre_label",
            "selected_genre",
            "n_tracks",
            "selection_type",
            "threshold_rule",
            "market_rule",
            "substitute_note",
        ],
    ]


def build_within_genre_joint_test_table(joint_test: dict[str, Any]) -> pd.DataFrame:
    """Build a one-row omnibus interaction-test table."""

    return pd.DataFrame([joint_test])


def build_within_genre_predictive_check_table(predictive: pd.DataFrame) -> pd.DataFrame:
    """Build a compact manuscript-facing predictive support table."""

    column_order = [
        "model_name",
        "test_r2",
        "test_rmse",
        "n_train",
        "n_test",
        "group_overlap_count",
        "r2_gain_vs_no_interaction",
        "rmse_reduction_vs_no_interaction",
    ]
    available_columns = [column for column in column_order if column in predictive.columns]
    return predictive.loc[:, available_columns].copy()


def build_within_genre_selection_robustness_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Build the selection-threshold robustness summary table."""

    column_order = [
        "min_count",
        "selected_genres",
        "n_genres",
        "sample_size",
        "no_interaction_adj_r2",
        "interaction_adj_r2",
        "adj_r2_gain",
        "wald_test_type",
        "wald_statistic",
        "wald_df",
        "wald_p_value",
    ]
    return summary.loc[:, column_order].copy()


def build_within_genre_repeated_holdout_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Build the repeated-holdout summary table."""

    column_order = [
        "n_repetitions",
        "mean_test_r2_gain",
        "std_test_r2_gain",
        "min_test_r2_gain",
        "max_test_r2_gain",
        "mean_rmse_reduction",
        "std_rmse_reduction",
        "min_rmse_reduction",
        "max_rmse_reduction",
    ]
    return summary.loc[:, column_order].copy()


def build_within_genre_scope_comparison_table(
    selected_results: dict[str, Any],
    eligible_results: dict[str, Any],
) -> pd.DataFrame:
    """Compare the market-facing subset against the all-eligible genre specification."""

    rows: list[dict[str, Any]] = []
    for analysis_scope, results in (
        ("market_facing_subset", selected_results),
        ("all_eligible_genres", eligible_results),
    ):
        joint_test = _extract_joint_test(results)
        rows.append(
            {
                "analysis_scope": analysis_scope,
                "n_genres": int(len(results["selected_genres"])),
                "sample_size": int(results["pooled_model"].nobs),
                "selected_genres": " | ".join(results["selected_genres"]),
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


def plot_within_genre_feature_comparison(results: dict[str, Any], path: str | Path) -> Path:
    """Plot genre-specific focal-feature coefficients with 95% confidence intervals."""

    plot_df = build_within_genre_followup_table(results).copy()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features = [_label_term(f"{feature}_z") for feature in results["focal_features"]]
    fig, axes = plt.subplots(len(features), 1, figsize=(8, 2.8 * len(features)), sharex=True)
    if len(features) == 1:
        axes = [axes]

    for ax, feature in zip(axes, features):
        feature_df = plot_df.loc[plot_df["term"] == feature].sort_values("coefficient")
        lower = feature_df["coefficient"] - 1.96 * feature_df["std_error"]
        upper = feature_df["coefficient"] + 1.96 * feature_df["std_error"]
        ax.errorbar(
            feature_df["coefficient"],
            feature_df["genre"],
            xerr=[feature_df["coefficient"] - lower, upper - feature_df["coefficient"]],
            fmt="o",
            color="#0f4c5c",
            ecolor="#88a9b0",
            capsize=3,
        )
        ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
        ax.set_title(feature)
        ax.set_ylabel("")
    axes[-1].set_xlabel("Within-genre coefficient")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_within_genre_selection_robustness(summary: pd.DataFrame, path: str | Path) -> Path:
    """Plot adjusted R^2 gain by minimum-count threshold."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = summary.sort_values("min_count")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(plot_df["min_count"], plot_df["adj_r2_gain"], marker="o", color="#1f4e79")
    ax.set_xlabel("Minimum genre count threshold")
    ax.set_ylabel("Adjusted R^2 gain")
    ax.set_title("Within-genre interaction gain across selection thresholds")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_genre_deviation_profile_summary_table(results: dict[str, Any]) -> pd.DataFrame:
    """Return the genre profile summary table for the deviation analysis."""

    return results["profile_summary"].copy()


def build_genre_deviation_model_table(results: dict[str, Any]) -> pd.DataFrame:
    """Build a compact table for the main genre-deviation model."""

    model = results["main_model"]
    comparison = results["main_comparison"]
    raw_columns = {f"{feature}_z" for feature in results["focal_features"]}
    abs_columns = {f"{feature}_abs_dev_z" for feature in results["focal_features"]}

    rows: list[dict[str, Any]] = []
    for term in model.params.index:
        if term == "Intercept":
            continue
        if term.startswith(f"C({GENRE_COLUMN})[T."):
            continue
        if term in raw_columns:
            term_group = "raw_feature"
        elif term in abs_columns:
            term_group = "absolute_deviation"
        else:
            term_group = "control"
        rows.append(
            {
                "term_group": term_group,
                "term": _label_term(term),
                "coefficient": float(model.params[term]),
                "std_error": float(model.bse[term]),
                "p_value_raw": float(model.pvalues[term]),
                "p_value": _format_p_value_for_display(float(model.pvalues[term])),
                "p_value_display": _format_p_value_for_display(float(model.pvalues[term])),
                "n_obs": float(comparison["n_obs"]),
                "adj_r2": float(comparison["fit_metric_value"]),
            }
        )
    return pd.DataFrame(rows)


def build_genre_deviation_robustness_table(results: dict[str, Any]) -> pd.DataFrame:
    """Build a compact table for the signed-deviation robustness model."""

    model = results["robustness_model"]
    comparison = results["robustness_comparison"]
    signed_columns = {f"{feature}_dev_z" for feature in results["focal_features"]}

    rows: list[dict[str, Any]] = []
    for term in model.params.index:
        if term == "Intercept":
            continue
        if term.startswith(f"C({GENRE_COLUMN})[T."):
            continue
        if term in signed_columns:
            term_group = "signed_deviation"
        else:
            term_group = "control"
        rows.append(
            {
                "term_group": term_group,
                "term": _label_term(term),
                "coefficient": float(model.params[term]),
                "std_error": float(model.bse[term]),
                "p_value_raw": float(model.pvalues[term]),
                "p_value": _format_p_value_for_display(float(model.pvalues[term])),
                "p_value_display": _format_p_value_for_display(float(model.pvalues[term])),
                "n_obs": float(comparison["n_obs"]),
                "adj_r2": float(comparison["fit_metric_value"]),
            }
        )
    return pd.DataFrame(rows)


def plot_genre_deviation_effects(results: dict[str, Any], path: str | Path) -> Path:
    """Plot the main and robustness deviation-term coefficients with 95% confidence intervals."""

    main_df = build_genre_deviation_model_table(results)
    main_df = main_df.loc[main_df["term_group"] == "absolute_deviation"].copy()
    main_df["model_label"] = "Absolute deviation"

    robustness_df = build_genre_deviation_robustness_table(results)
    robustness_df = robustness_df.loc[robustness_df["term_group"] == "signed_deviation"].copy()
    robustness_df["model_label"] = "Signed deviation"

    plot_df = pd.concat([main_df, robustness_df], ignore_index=True)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6.2), sharex=False)
    for ax, model_label in zip(axes, ["Absolute deviation", "Signed deviation"]):
        model_df = plot_df.loc[plot_df["model_label"] == model_label].sort_values("coefficient")
        lower = model_df["coefficient"] - 1.96 * model_df["std_error"]
        upper = model_df["coefficient"] + 1.96 * model_df["std_error"]
        ax.errorbar(
            model_df["coefficient"],
            model_df["term"],
            xerr=[model_df["coefficient"] - lower, upper - model_df["coefficient"]],
            fmt="o",
            color="#7a3e00" if model_label == "Absolute deviation" else "#0f4c5c",
            ecolor="#d7b58d" if model_label == "Absolute deviation" else "#88a9b0",
            capsize=3,
        )
        ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
        ax.set_title(model_label)
        ax.set_ylabel("")
    axes[-1].set_xlabel("Coefficient")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _collect_terms(models: dict[str, Any], exclude_genre_fixed_effects: bool) -> list[str]:
    """Collect and order terms across models while optionally hiding genre FE coefficients."""

    terms: list[str] = []
    for model in models.values():
        for term in model.params.index:
            if term == "Intercept":
                continue
            if exclude_genre_fixed_effects and term.startswith(f"C({GENRE_COLUMN})[T."):
                continue
            if term not in terms:
                terms.append(term)
    return terms


def _label_term(term: str) -> str:
    """Create a readable label for a model term."""

    if term in TERM_LABELS:
        return TERM_LABELS[term]
    if term.startswith("C(key)[T."):
        return f"Key = {term.split('[T.')[1].rstrip(']')}"
    if term.startswith("C(mode)[T."):
        return f"Mode = {term.split('[T.')[1].rstrip(']')}"
    if term.startswith("C(time_signature)[T."):
        return f"Time signature = {term.split('[T.')[1].rstrip(']')}"
    if term.startswith(f"C({GENRE_COLUMN})[T.") and ":" in term:
        genre = term.split("[T.")[1].split("]")[0]
        feature = term.split(":", maxsplit=1)[1]
        return f"{_label_term(feature)} x Genre: {genre}"
    if ":" in term and term.split(":", maxsplit=1)[0] in {
        f"{feature}_z" for feature in WITHIN_GENRE_FOCAL_FEATURES
    }:
        feature, genre_term = term.split(":", maxsplit=1)
        genre = genre_term.split("[T.")[1].split("]")[0]
        return f"{_label_term(feature)} x Genre: {genre}"
    return term


def _format_manuscript_genre_label(target_genre: str, selected_genre: str) -> str:
    """Return a manuscript-safe genre label, exposing substitute labels explicitly."""

    if target_genre == selected_genre:
        return target_genre
    return f"{selected_genre} (dataset proxy)"


def _format_p_value_for_display(p_value: float) -> str:
    """Format p-values to match manuscript table conventions."""

    if np.isnan(p_value):
        return ""
    rounded = round(float(p_value), 3)
    if rounded == 0.0:
        return "< 0.001"
    return f"{rounded:.3f}"


def _format_substitute_note(target_genre: str, selected_genre: str, selection_type: str) -> str:
    """Describe whether the selected genre is exact or a dataset substitute."""

    if selection_type == "exact":
        return "Exact dataset label used"
    return f"Selected as substitute for broader {target_genre} category"


def _extract_joint_test(results: dict[str, Any]) -> dict[str, Any]:
    """Extract or compute the pooled interaction omnibus test."""

    if "joint_test" in results:
        return results["joint_test"]

    model = results["pooled_model"]
    interaction_terms = [
        term
        for term in model.params.index
        if ":" in term and any(f"{feature}_z" in term for feature in results["focal_features"])
    ]
    restrictions = np.zeros((len(interaction_terms), len(model.params.index)))
    for row_index, term in enumerate(interaction_terms):
        restrictions[row_index, list(model.params.index).index(term)] = 1.0

    wald_result = model.wald_test(restrictions, use_f=False, scalar=True)
    return {
        "test_type": "Wald chi-square",
        "statistic": float(np.asarray(wald_result.statistic).squeeze()),
        "df": int(len(interaction_terms)),
        "p_value": float(np.asarray(wald_result.pvalue).squeeze()),
        "term_count": int(len(interaction_terms)),
    }


def _is_feature_interaction(term: str, feature: str) -> bool:
    """Check whether a term is an interaction involving a focal feature."""

    return f":{feature}" in term or term.startswith(f"{feature}:")


def _summary_row(label: str, comparison: pd.DataFrame, field: str) -> dict[str, Any]:
    """Create a summary-stat row aligned to the main regression table layout."""

    row: dict[str, Any] = {
        "term": label,
        "row_type": "summary",
        "model_1_flag": np.nan,
        "model_2_flag": np.nan,
        "model_3_flag": np.nan,
    }
    for model_name in comparison.index:
        row[f"{model_name}_coef"] = comparison.loc[model_name, field]
        row[f"{model_name}_se"] = np.nan
        row[f"{model_name}_p_value"] = np.nan
    return row


def _flag_row(label: str, flags: dict[str, str]) -> dict[str, Any]:
    """Create a yes/no indicator row aligned to the main regression table layout."""

    row: dict[str, Any] = {"term": label, "row_type": "flag"}
    for model_name, value in flags.items():
        row[f"{model_name}_coef"] = np.nan
        row[f"{model_name}_se"] = np.nan
        row[f"{model_name}_p_value"] = np.nan
        row[f"{model_name}_flag"] = value
    for model_name in ("model_1", "model_2", "model_3"):
        row.setdefault(f"{model_name}_flag", np.nan)
    return row
