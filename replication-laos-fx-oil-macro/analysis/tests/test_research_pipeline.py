from pathlib import Path

from laos_fx_oil_macro.pipeline import run_research_pipeline


ROOT = Path(__file__).resolve().parents[1]


def test_run_research_pipeline_writes_core_artifacts(tmp_path):
    manifest = run_research_pipeline(
        project_root=ROOT,
        output_root=tmp_path,
        tvp_draws=60,
        tvp_burnin=20,
        tvp_thin=1,
        tvp_horizons=4,
        skip_tvp=False,
    )

    required_keys = {
        "processed_panel",
        "stationarity_table",
        "regime_summary_table",
        "break_diagnostics_table",
        "event_snapshot_table",
        "gpr_to_oil_table",
        "gpr_alt_to_oil_table",
        "lp_oil_to_fx_table",
        "lp_oil_to_inflation_table",
        "lp_fx_to_inflation_table",
        "lp_fx_to_cpi_inf_table",
        "lp_fx_to_inflation_truncated_table",
        "robustness_summary_table",
        "tvp_irf_table",
        "tvp_fit_table",
    }
    assert required_keys.issubset(manifest.keys())
    for key in required_keys:
        assert manifest[key].exists(), key

    assert manifest["processed_panel"].name.endswith(".csv")
    assert manifest["tvp_irf_table"].name == "tvp_svar_irf_summary.csv"
