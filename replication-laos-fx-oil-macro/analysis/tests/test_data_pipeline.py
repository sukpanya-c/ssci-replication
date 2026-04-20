from pathlib import Path

import numpy as np
import pandas as pd

from laos_fx_oil_macro.data import (
    build_alt_gpr_series,
    summarize_stationarity,
    transform_panel,
)


ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT.parent / "dataset" / "laos_fx_oil_macro_monthly_panel_2006-01-31_2026-02-28.csv"
COMPONENTS_PATH = ROOT.parent / "dataset" / "GPR_ME_middle_east_components_month_end_2006-01-31_2026-02-28.csv"


def test_transform_panel_creates_research_variables_and_tags():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2021-12-31", "2022-01-31", "2022-03-31", "2023-10-31", "2026-02-28"]
            ),
            "Oil": [100.0, 110.0, 121.0, 133.1, 146.41],
            "USDLAK": [10000.0, 11000.0, 12100.0, 13310.0, 14641.0],
            "CPI": [100.0, 102.0, 104.04, 106.1208, 108.243216],
            "CPI_Inf": [2.0, 3.0, 4.0, 5.0, 6.0],
            "GPR_ME": [0.2, 0.3, 0.4, 0.5, 0.6],
            "FFR": [1.0, 1.0, 1.0, 1.0, 1.0],
            "GSCPI": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    transformed = transform_panel(df)

    assert {"oil_shock", "fx_dep", "inflation_mom", "post_2022", "event_tag"}.issubset(
        transformed.columns
    )
    assert transformed.loc[1, "post_2022"] == 1
    assert transformed.loc[0, "post_2022"] == 0
    assert transformed.loc[2, "event_tag"] == "benchmark_2022_03"
    assert transformed.loc[3, "event_tag"] == "scenario_2023_10"
    assert transformed.loc[4, "event_tag"] == "scenario_2026_02"
    assert np.isclose(transformed.loc[1, "oil_shock"], 100 * np.log(1.1))
    assert np.isclose(transformed.loc[1, "fx_dep"], 100 * np.log(1.1))
    assert np.isclose(transformed.loc[1, "inflation_mom"], 100 * np.log(1.02))


def test_build_alt_gpr_series_excludes_south_africa():
    components = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2023-01-31", "2023-02-28"]),
            "Saudi Arabia": [0.1, 0.2],
            "Israel": [0.3, 0.4],
            "Turkey": [0.5, 0.6],
            "South Africa": [0.7, 0.8],
            "Egypt": [0.9, 1.0],
        }
    )

    alt = build_alt_gpr_series(components)

    assert list(alt.columns) == ["Date", "GPR_ME_alt"]
    assert np.isclose(alt.loc[0, "GPR_ME_alt"], np.mean([0.1, 0.3, 0.5, 0.9]))
    assert np.isclose(alt.loc[1, "GPR_ME_alt"], np.mean([0.2, 0.4, 0.6, 1.0]))


def test_stationarity_summary_flags_levels_vs_changes_on_real_panel():
    raw = pd.read_csv(PANEL_PATH, parse_dates=["Date"])
    assert raw["Date"].min() == pd.Timestamp("2006-01-31")
    components = pd.read_csv(COMPONENTS_PATH, parse_dates=["Date"])
    assert components["Date"].min() == pd.Timestamp("2006-01-31")
    transformed = transform_panel(raw)

    summary = summarize_stationarity(
        transformed,
        ["USDLAK", "CPI", "fx_dep", "oil_shock", "inflation_mom"],
    )

    by_variable = summary.set_index("variable")
    assert by_variable.loc["USDLAK", "stationary_5pct"] == 0
    assert by_variable.loc["CPI", "stationary_5pct"] == 0
    assert by_variable.loc["oil_shock", "stationary_5pct"] == 1
    assert by_variable.loc["fx_dep", "p_value"] < 0.25
    assert by_variable.loc["fx_dep", "p_value"] < by_variable.loc["USDLAK", "p_value"]
    assert by_variable.loc["inflation_mom", "p_value"] < 0.20
    assert by_variable.loc["inflation_mom", "p_value"] < by_variable.loc["CPI", "p_value"]
