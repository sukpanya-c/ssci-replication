from pathlib import Path

import pandas as pd

from laos_fx_oil_macro.data import transform_panel
from laos_fx_oil_macro.models import run_local_projection


ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT.parent / "datasets" / "laos_fx_oil_macro_monthly_panel_2006-01-31_2026-02-28.csv"


def test_local_projection_returns_horizon_table_for_post_2022_regime():
    raw = pd.read_csv(PANEL_PATH, parse_dates=["Date"])
    transformed = transform_panel(raw)

    irf = run_local_projection(
        transformed,
        response="inflation_mom",
        impulse="fx_dep",
        controls=["oil_shock", "GPR_ME", "FFR", "GSCPI"],
        horizons=6,
        lags=2,
        regime="post_2022",
    )

    assert list(irf["horizon"]) == list(range(7))
    assert {"coefficient", "lower_90", "upper_90", "nobs", "regime", "impulse", "response"}.issubset(
        irf.columns
    )
    assert (irf["regime"] == "post_2022").all()
    assert (irf["nobs"] > 20).all()
