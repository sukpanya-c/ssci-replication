from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _subset_regime(df: pd.DataFrame, regime: str | None) -> pd.DataFrame:
    if regime is None or regime == "full_sample":
        return df.copy()
    if regime == "post_2022":
        return df[df["Date"] >= pd.Timestamp("2022-01-31")].copy()
    if regime == "pre_2022":
        return df[df["Date"] < pd.Timestamp("2022-01-31")].copy()
    raise ValueError(f"Unknown regime: {regime}")


def _prepare_lp_matrix(
    df: pd.DataFrame,
    response: str,
    impulse: str,
    controls: Iterable[str],
    horizon: int,
    lags: int,
) -> pd.DataFrame:
    controls = list(controls)
    work = df.copy()
    work[f"{response}_lead_{horizon}"] = work[response].shift(-horizon)
    regressors = [impulse] + controls
    for variable in [response, impulse, *controls]:
        for lag in range(1, lags + 1):
            lag_col = f"{variable}_lag_{lag}"
            work[lag_col] = work[variable].shift(lag)
            regressors.append(lag_col)
    needed = [f"{response}_lead_{horizon}", *regressors]
    return work.dropna(subset=needed)[needed]


def run_local_projection(
    df: pd.DataFrame,
    response: str,
    impulse: str,
    controls: Iterable[str],
    horizons: int = 12,
    lags: int = 2,
    regime: str | None = None,
) -> pd.DataFrame:
    sample = _subset_regime(df, regime)
    rows: list[dict[str, float | int | str]] = []
    for horizon in range(horizons + 1):
        matrix = _prepare_lp_matrix(sample, response, impulse, controls, horizon, lags)
        y = matrix[f"{response}_lead_{horizon}"]
        X = sm.add_constant(matrix.drop(columns=f"{response}_lead_{horizon}"))
        fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max(lags, horizon + 1)})
        coeff = float(fit.params[impulse])
        se = float(fit.bse[impulse])
        band = 1.645 * se
        rows.append(
            {
                "horizon": horizon,
                "coefficient": coeff,
                "lower_90": coeff - band,
                "upper_90": coeff + band,
                "stderr": se,
                "nobs": int(fit.nobs),
                "regime": regime or "full_sample",
                "impulse": impulse,
                "response": response,
            }
        )
    return pd.DataFrame(rows)


def run_hac_ols(
    df: pd.DataFrame,
    response: str,
    regressors: Iterable[str],
    hac_lags: int = 3,
    regime: str | None = None,
) -> pd.DataFrame:
    sample = _subset_regime(df, regime).dropna(subset=[response, *regressors]).copy()
    y = sample[response]
    X = sm.add_constant(sample[list(regressors)])
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    rows = []
    for term in fit.params.index:
        rows.append(
            {
                "response": response,
                "term": term,
                "coefficient": float(fit.params[term]),
                "std_error": float(fit.bse[term]),
                "p_value": float(fit.pvalues[term]),
                "nobs": int(fit.nobs),
                "regime": regime or "full_sample",
                "r_squared": float(fit.rsquared),
            }
        )
    return pd.DataFrame(rows)

