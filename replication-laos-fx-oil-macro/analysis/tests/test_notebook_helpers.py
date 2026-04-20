from pathlib import Path

import pandas as pd

from laos_fx_oil_macro.notebook_helpers import (
    build_notebook_project,
    compact_interval_pivot,
    discover_project_root,
    path_status_table,
    variable_definitions_table,
)


ROOT = Path(__file__).resolve().parents[1]


def test_discover_project_root_finds_repo_markers_from_notebooks_dir():
    assert discover_project_root(ROOT / "notebooks") == ROOT


def test_build_notebook_project_exposes_core_research_paths():
    project = build_notebook_project(ROOT)

    assert project.root == ROOT
    assert project.dataset_path.name == "laos_fx_oil_macro_monthly_panel_2006-01-31_2026-02-28.csv"
    assert project.processed_panel.name == "laos_fx_oil_macro_research_panel.csv"
    assert project.results_summary.name == "results_summary.md"


def test_path_status_table_lists_existing_core_artifacts():
    project = build_notebook_project(ROOT)

    status = path_status_table(project)
    by_label = status.set_index("label")

    assert bool(by_label.loc["project_root", "exists"])
    assert bool(by_label.loc["dataset", "exists"])
    assert bool(by_label.loc["processed_panel", "exists"])
    assert bool(by_label.loc["output_root", "exists"])
    assert "manuscript" not in by_label.index


def test_compact_interval_pivot_formats_local_projection_rows_for_notebooks():
    lp = pd.DataFrame(
        {
            "horizon": [0, 0, 1, 1],
            "regime": ["pre_2022", "post_2022", "pre_2022", "post_2022"],
            "coefficient": [0.039508, 0.494450, 0.020830, 0.437576],
            "lower_90": [0.014961, 0.397433, 0.004225, 0.278911],
            "upper_90": [0.064055, 0.591466, 0.037434, 0.596240],
        }
    )

    table = compact_interval_pivot(lp, index="horizon", columns="regime")

    assert table.loc[0, "pre_2022"] == "0.040 [0.015, 0.064]"
    assert table.loc[0, "post_2022"] == "0.494 [0.397, 0.591]"


def test_variable_definitions_table_covers_paper_variables():
    definitions = variable_definitions_table()

    assert {"variable", "definition", "model_role"}.issubset(definitions.columns)
    assert {"GPR_ME", "oil_shock", "fx_dep", "inflation_mom", "CPI_Inf"}.issubset(
        set(definitions["variable"])
    )
