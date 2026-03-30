from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from .audit import audit_dataset
    from .config import (
        CORRELATION_MATRIX_PATH,
        DESCRIPTIVE_TABLE_PATH,
        GENRE_DEVIATION_FIGURE_PATH,
        GENRE_DEVIATION_MODEL_TABLE_PATH,
        GENRE_DEVIATION_ROBUSTNESS_TABLE_PATH,
        GENRE_PROFILE_SUMMARY_TABLE_PATH,
        GENRE_SELECTION_TABLE_PATH,
        MAIN_COEFFICIENT_FIGURE_PATH,
        MAIN_REGRESSION_TABLE_PATH,
        PREDICTIVE_CHECK_TABLE_PATH,
        PRIMARY_EXPORT_PATH,
        ROBUSTNESS_EXPORT_PATH,
        ROBUSTNESS_SUMMARY_TABLE_PATH,
        WITHIN_GENRE_ALL_ELIGIBLE_COMPARISON_TABLE_PATH,
        WITHIN_GENRE_FIGURE_PATH,
        WITHIN_GENRE_FOLLOWUP_TABLE_PATH,
        WITHIN_GENRE_INTERACTION_TABLE_PATH,
        WITHIN_GENRE_JOINT_TEST_TABLE_PATH,
        WITHIN_GENRE_PREDICTIVE_CHECK_TABLE_PATH,
        WITHIN_GENRE_REPEATED_HOLDOUT_RAW_TABLE_PATH,
        WITHIN_GENRE_REPEATED_HOLDOUT_SUMMARY_TABLE_PATH,
        WITHIN_GENRE_ROBUSTNESS_TABLE_PATH,
        WITHIN_GENRE_SELECTION_ROBUSTNESS_FIGURE_PATH,
        WITHIN_GENRE_SELECTION_ROBUSTNESS_TABLE_PATH,
    )
    from .modeling import (
        compute_within_genre_joint_test,
        create_descriptive_outputs,
        fit_all_eligible_within_genre_analysis,
        fit_genre_deviation_analysis,
        fit_main_regression_models,
        fit_robustness_models,
        fit_within_genre_analysis,
        load_analysis_dataset,
        run_grouped_predictive_check,
        run_within_genre_predictive_check,
        run_within_genre_repeated_holdout_validation,
        run_within_genre_selection_rule_robustness,
    )
    from .reporting import (
        build_genre_deviation_model_table,
        build_genre_deviation_profile_summary_table,
        build_genre_deviation_robustness_table,
        build_genre_selection_summary_table,
        build_main_regression_table,
        build_robustness_summary_table,
        build_within_genre_followup_table,
        build_within_genre_interaction_table,
        build_within_genre_joint_test_table,
        build_within_genre_predictive_check_table,
        build_within_genre_repeated_holdout_summary_table,
        build_within_genre_robustness_table,
        build_within_genre_scope_comparison_table,
        build_within_genre_selection_robustness_table,
        export_table,
        plot_genre_deviation_effects,
        plot_main_coefficients,
        plot_within_genre_feature_comparison,
        plot_within_genre_selection_robustness,
    )
except ImportError:  # pragma: no cover - compatibility for direct module execution
    from audit import audit_dataset
    from config import (
        CORRELATION_MATRIX_PATH,
        DESCRIPTIVE_TABLE_PATH,
        GENRE_DEVIATION_FIGURE_PATH,
        GENRE_DEVIATION_MODEL_TABLE_PATH,
        GENRE_DEVIATION_ROBUSTNESS_TABLE_PATH,
        GENRE_PROFILE_SUMMARY_TABLE_PATH,
        GENRE_SELECTION_TABLE_PATH,
        MAIN_COEFFICIENT_FIGURE_PATH,
        MAIN_REGRESSION_TABLE_PATH,
        PREDICTIVE_CHECK_TABLE_PATH,
        PRIMARY_EXPORT_PATH,
        ROBUSTNESS_EXPORT_PATH,
        ROBUSTNESS_SUMMARY_TABLE_PATH,
        WITHIN_GENRE_ALL_ELIGIBLE_COMPARISON_TABLE_PATH,
        WITHIN_GENRE_FIGURE_PATH,
        WITHIN_GENRE_FOLLOWUP_TABLE_PATH,
        WITHIN_GENRE_INTERACTION_TABLE_PATH,
        WITHIN_GENRE_JOINT_TEST_TABLE_PATH,
        WITHIN_GENRE_PREDICTIVE_CHECK_TABLE_PATH,
        WITHIN_GENRE_REPEATED_HOLDOUT_RAW_TABLE_PATH,
        WITHIN_GENRE_REPEATED_HOLDOUT_SUMMARY_TABLE_PATH,
        WITHIN_GENRE_ROBUSTNESS_TABLE_PATH,
        WITHIN_GENRE_SELECTION_ROBUSTNESS_FIGURE_PATH,
        WITHIN_GENRE_SELECTION_ROBUSTNESS_TABLE_PATH,
    )
    from modeling import (
        compute_within_genre_joint_test,
        create_descriptive_outputs,
        fit_all_eligible_within_genre_analysis,
        fit_genre_deviation_analysis,
        fit_main_regression_models,
        fit_robustness_models,
        fit_within_genre_analysis,
        load_analysis_dataset,
        run_grouped_predictive_check,
        run_within_genre_predictive_check,
        run_within_genre_repeated_holdout_validation,
        run_within_genre_selection_rule_robustness,
    )
    from reporting import (
        build_genre_deviation_model_table,
        build_genre_deviation_profile_summary_table,
        build_genre_deviation_robustness_table,
        build_genre_selection_summary_table,
        build_main_regression_table,
        build_robustness_summary_table,
        build_within_genre_followup_table,
        build_within_genre_interaction_table,
        build_within_genre_joint_test_table,
        build_within_genre_predictive_check_table,
        build_within_genre_repeated_holdout_summary_table,
        build_within_genre_robustness_table,
        build_within_genre_scope_comparison_table,
        build_within_genre_selection_robustness_table,
        export_table,
        plot_genre_deviation_effects,
        plot_main_coefficients,
        plot_within_genre_feature_comparison,
        plot_within_genre_selection_robustness,
    )


def find_project_root(start: str | Path | None = None) -> Path:
    """Locate the project root containing the analysis source package."""

    start_path = Path.cwd() if start is None else Path(start)
    if start_path.is_file():
        start_path = start_path.parent
    start_path = start_path.resolve()

    for candidate in (start_path, *start_path.parents):
        if (candidate / "analysis" / "src").exists():
            return candidate
    raise RuntimeError("Could not locate project root containing analysis/src.")


def run_main_analysis_workflow(project_root: str | Path) -> dict[str, Any]:
    """Run the full main-analysis workflow and export all main outputs."""

    root = Path(project_root).resolve()
    primary_sample = load_analysis_dataset(root / PRIMARY_EXPORT_PATH)
    robustness_sample = load_analysis_dataset(root / ROBUSTNESS_EXPORT_PATH)

    descriptive_table, correlation_matrix = create_descriptive_outputs(primary_sample)
    export_table(descriptive_table, root / DESCRIPTIVE_TABLE_PATH)
    export_table(correlation_matrix.reset_index().rename(columns={"index": "variable"}), root / CORRELATION_MATRIX_PATH)

    main_results = fit_main_regression_models(primary_sample)
    main_regression_table = build_main_regression_table(main_results)
    export_table(main_regression_table, root / MAIN_REGRESSION_TABLE_PATH)
    predictive_summary = run_grouped_predictive_check(primary_sample)
    export_table(predictive_summary, root / PREDICTIVE_CHECK_TABLE_PATH)
    main_figure_path = plot_main_coefficients(main_results, root / MAIN_COEFFICIENT_FIGURE_PATH)

    within_genre_results = fit_within_genre_analysis(primary_sample)
    genre_selection_table = build_genre_selection_summary_table(
        within_genre_results["selection_table"],
        min_count=within_genre_results["selection_rule"]["min_count"],
    )
    within_genre_interaction_table = build_within_genre_interaction_table(within_genre_results)
    within_genre_followup_table = build_within_genre_followup_table(within_genre_results)
    within_genre_robustness_table = build_within_genre_robustness_table(within_genre_results)
    within_genre_joint_test_table = build_within_genre_joint_test_table(
        compute_within_genre_joint_test(within_genre_results)
    )
    within_genre_predictive_check_table = build_within_genre_predictive_check_table(
        run_within_genre_predictive_check(
            primary_sample,
            min_count=within_genre_results["selection_rule"]["min_count"],
            focal_features=within_genre_results["focal_features"],
        )
    )
    export_table(genre_selection_table, root / GENRE_SELECTION_TABLE_PATH)
    export_table(within_genre_interaction_table, root / WITHIN_GENRE_INTERACTION_TABLE_PATH)
    export_table(within_genre_followup_table, root / WITHIN_GENRE_FOLLOWUP_TABLE_PATH)
    export_table(within_genre_robustness_table, root / WITHIN_GENRE_ROBUSTNESS_TABLE_PATH)
    export_table(within_genre_joint_test_table, root / WITHIN_GENRE_JOINT_TEST_TABLE_PATH)
    export_table(within_genre_predictive_check_table, root / WITHIN_GENRE_PREDICTIVE_CHECK_TABLE_PATH)
    within_genre_figure_path = plot_within_genre_feature_comparison(
        within_genre_results,
        root / WITHIN_GENRE_FIGURE_PATH,
    )

    selection_robustness_table = build_within_genre_selection_robustness_table(
        run_within_genre_selection_rule_robustness(primary_sample)
    )
    selection_robustness_figure_path = plot_within_genre_selection_robustness(
        selection_robustness_table,
        root / WITHIN_GENRE_SELECTION_ROBUSTNESS_FIGURE_PATH,
    )
    repeated_holdout = run_within_genre_repeated_holdout_validation(
        primary_sample,
        min_count=within_genre_results["selection_rule"]["min_count"],
        focal_features=within_genre_results["focal_features"],
    )
    repeated_holdout_summary = build_within_genre_repeated_holdout_summary_table(repeated_holdout["summary"])
    all_eligible_results = fit_all_eligible_within_genre_analysis(
        primary_sample,
        min_count=within_genre_results["selection_rule"]["min_count"],
        focal_features=within_genre_results["focal_features"],
    )
    scope_comparison_table = build_within_genre_scope_comparison_table(
        within_genre_results,
        all_eligible_results,
    )
    export_table(selection_robustness_table, root / WITHIN_GENRE_SELECTION_ROBUSTNESS_TABLE_PATH)
    export_table(repeated_holdout_summary, root / WITHIN_GENRE_REPEATED_HOLDOUT_SUMMARY_TABLE_PATH)
    export_table(repeated_holdout["raw_results"], root / WITHIN_GENRE_REPEATED_HOLDOUT_RAW_TABLE_PATH)
    export_table(scope_comparison_table, root / WITHIN_GENRE_ALL_ELIGIBLE_COMPARISON_TABLE_PATH)

    genre_deviation_results = fit_genre_deviation_analysis(primary_sample)
    genre_profile_summary_table = build_genre_deviation_profile_summary_table(genre_deviation_results)
    genre_deviation_model_table = build_genre_deviation_model_table(genre_deviation_results)
    genre_deviation_robustness_table = build_genre_deviation_robustness_table(genre_deviation_results)
    export_table(genre_profile_summary_table, root / GENRE_PROFILE_SUMMARY_TABLE_PATH)
    export_table(genre_deviation_model_table, root / GENRE_DEVIATION_MODEL_TABLE_PATH)
    export_table(genre_deviation_robustness_table, root / GENRE_DEVIATION_ROBUSTNESS_TABLE_PATH)
    genre_deviation_figure_path = plot_genre_deviation_effects(
        genre_deviation_results,
        root / GENRE_DEVIATION_FIGURE_PATH,
    )

    return {
        "sample_overview": {
            "primary_shape": primary_sample.shape,
            "robustness_shape": robustness_sample.shape,
            "primary_audit": audit_dataset(primary_sample),
            "robustness_audit": audit_dataset(robustness_sample),
        },
        "descriptives": {
            "table_path": str(root / DESCRIPTIVE_TABLE_PATH),
            "correlation_path": str(root / CORRELATION_MATRIX_PATH),
            "table_preview": descriptive_table.head(10),
        },
        "main_models": {
            "comparison": main_results["comparison"],
            "table_path": str(root / MAIN_REGRESSION_TABLE_PATH),
        },
        "main_outputs": {
            "predictive_summary": predictive_summary,
            "predictive_table_path": str(root / PREDICTIVE_CHECK_TABLE_PATH),
            "figure_path": str(main_figure_path),
        },
        "within_genre": {
            "selected_genres": genre_selection_table,
            "pooled_formula": within_genre_results["pooled_formula"],
            "joint_test": within_genre_joint_test_table,
            "predictive_check": within_genre_predictive_check_table,
            "robustness_formula": within_genre_results["robustness_formula"],
            "figure_path": str(within_genre_figure_path),
        },
        "within_genre_support": {
            "selection_rule_robustness": selection_robustness_table,
            "repeated_holdout_summary": repeated_holdout_summary,
            "all_eligible_scope_comparison": scope_comparison_table,
            "figure_path": str(selection_robustness_figure_path),
        },
        "genre_deviation": {
            "selected_genres": genre_deviation_results["selected_genres"],
            "main_formula": genre_deviation_results["main_formula"],
            "robustness_formula": genre_deviation_results["robustness_formula"],
            "figure_path": str(genre_deviation_figure_path),
        },
    }


def run_appendix_robustness_workflow(project_root: str | Path) -> dict[str, Any]:
    """Run the appendix robustness workflow and export the appendix table."""

    root = Path(project_root).resolve()
    robustness_sample = load_analysis_dataset(root / ROBUSTNESS_EXPORT_PATH)
    robustness_results = fit_robustness_models(robustness_sample)
    robustness_summary_table = build_robustness_summary_table(robustness_results)
    export_table(robustness_summary_table, root / ROBUSTNESS_SUMMARY_TABLE_PATH)

    return {
        "sample_overview": {
            "robustness_shape": robustness_sample.shape,
            "robustness_audit": audit_dataset(robustness_sample),
            "had_multiple_genres_count": int(robustness_sample["had_multiple_genres"].sum()),
        },
        "robustness": {
            "comparison": robustness_results["comparison"],
            "table_path": str(root / ROBUSTNESS_SUMMARY_TABLE_PATH),
            "table_preview": robustness_summary_table.head(12),
        },
    }
