import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_tvp_svar.R"


def test_tvp_svar_runner_writes_irf_summary(tmp_path):
    dates = pd.date_range("2010-01-31", periods=72, freq="ME")
    rng = np.random.default_rng(7)
    data = pd.DataFrame(
        {
            "Date": dates,
            "GPR_ME": np.abs(rng.normal(0.2, 0.1, len(dates))),
            "oil_shock": rng.normal(0.0, 3.0, len(dates)),
            "fx_dep": rng.normal(0.0, 2.0, len(dates)),
            "inflation_mom": rng.normal(0.2, 0.7, len(dates)),
        }
    )
    input_path = tmp_path / "tvp_input.csv"
    output_dir = tmp_path / "tvp_output"
    data.to_csv(input_path, index=False)

    cmd = [
        "Rscript",
        str(SCRIPT_PATH),
        "--input",
        str(input_path),
        "--output",
        str(output_dir),
        "--draws",
        "60",
        "--burnin",
        "20",
        "--thin",
        "1",
        "--horizons",
        "4",
        "--dates",
        "2012-03-31,2013-10-31",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr

    irf_path = output_dir / "tvp_svar_irf_summary.csv"
    assert irf_path.exists()

    irf = pd.read_csv(irf_path)
    assert {"date", "impulse", "response", "horizon", "median", "lower_90", "upper_90", "scenario"}.issubset(
        irf.columns
    )
    assert set(irf["date"]) == {"2012-03-31", "2013-10-31"}
