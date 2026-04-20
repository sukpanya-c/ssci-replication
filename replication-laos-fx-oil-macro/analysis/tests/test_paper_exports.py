from pathlib import Path

import pandas as pd

from laos_fx_oil_macro.paper_exports import (
    build_submission_bundle,
    export_figure_1_macro_context,
    export_figure_3_oil_to_fx_lp,
    export_figure_6_tvp_svar_scenarios,
    make_table_3_fx_pass_through_lp,
    make_table_5_tvp_svar_scenarios,
)


ROOT = Path(__file__).resolve().parents[1]


def test_make_table_3_fx_pass_through_lp_formats_readable_intervals():
    lp = pd.read_csv(ROOT / "output" / "tables" / "lp_fx_to_inflation.csv")

    table = make_table_3_fx_pass_through_lp(lp)

    assert "Horizon (months)" in table.columns
    assert "Pre-2022 estimate [90% CI]" in table.columns
    assert "Post-2022 estimate [90% CI]" in table.columns
    assert table.loc[0, "Pre-2022 estimate [90% CI]"] == "-0.049 [-0.137, 0.039]"
    assert table.loc[0, "Post-2022 estimate [90% CI]"] == "0.494 [0.397, 0.591]"


def test_make_table_5_tvp_svar_scenarios_excludes_gpr_to_oil_channel():
    irf = pd.read_csv(ROOT / "output" / "models" / "tvp_svar_irf_summary.csv")

    table = make_table_5_tvp_svar_scenarios(irf)

    assert "Channel" in table.columns
    assert "Impact median [90% CI]" in table.columns
    assert set(table["Channel"]) == {
        "Oil shock -> FX depreciation",
        "Oil shock -> monthly inflation",
        "FX depreciation -> monthly inflation",
    }


def test_exported_paper_figures_do_not_embed_figure_name_titles(monkeypatch):
    captured: dict[str, object] = {}

    def fake_save(fig, path):
        captured[str(path)] = fig
        return path

    monkeypatch.setattr("laos_fx_oil_macro.paper_exports._save_figure", fake_save)

    panel = pd.read_csv(ROOT / "output" / "data" / "laos_fx_oil_macro_research_panel.csv", parse_dates=["Date"])
    panel = panel.dropna(subset=["oil_shock", "fx_dep", "inflation_mom"])
    lp = pd.read_csv(ROOT / "output" / "tables" / "lp_oil_to_fx.csv")
    irf = pd.read_csv(ROOT / "output" / "models" / "tvp_svar_irf_summary.csv")

    export_figure_1_macro_context(panel, ROOT / "tmp_figure_1.png")
    export_figure_3_oil_to_fx_lp(lp, ROOT / "tmp_figure_3.png")
    export_figure_6_tvp_svar_scenarios(irf, ROOT / "tmp_figure_6.png")

    fig1 = captured[str(ROOT / "tmp_figure_1.png")]
    fig3 = captured[str(ROOT / "tmp_figure_3.png")]
    fig6 = captured[str(ROOT / "tmp_figure_6.png")]

    assert fig1._suptitle is None
    assert fig3.axes[0].get_title() == ""
    assert fig6._suptitle is None


def test_build_submission_bundle_writes_paper_ready_outputs(tmp_path):
    output_root = tmp_path / "output"
    paper_dir = tmp_path / "paper"

    manifest = build_submission_bundle(
        project_root=ROOT,
        output_root=output_root,
        paper_dir=paper_dir,
        force_pipeline=True,
        tvp_draws=60,
        tvp_burnin=20,
        tvp_thin=1,
        tvp_horizons=4,
        lp_horizons=4,
        lp_lags=2,
    )

    expected_keys = {
        "table_1_descriptive_stats_csv",
        "table_1_descriptive_stats_md",
        "table_2_break_diagnostics_csv",
        "table_2_break_diagnostics_md",
        "table_3_fx_pass_through_lp_csv",
        "table_3_fx_pass_through_lp_md",
        "table_4_robustness_csv",
        "table_4_robustness_md",
        "table_5_tvp_svar_scenarios_csv",
        "table_5_tvp_svar_scenarios_md",
        "figure_1_macro_context",
        "figure_2_crisis_overlay",
        "figure_3_oil_to_fx_lp",
        "figure_4_oil_to_inflation_lp",
        "figure_5_fx_to_inflation_lp",
        "figure_6_tvp_svar_scenarios",
        "results_summary",
    }

    assert expected_keys.issubset(manifest.keys())
    for key in expected_keys:
        assert manifest[key].exists(), key

    results_summary = manifest["results_summary"].read_text(encoding="utf-8")
    assert "post-2022" in results_summary
    assert "fx_dep -> inflation_mom" in results_summary
    assert "should not be overstated" in results_summary
