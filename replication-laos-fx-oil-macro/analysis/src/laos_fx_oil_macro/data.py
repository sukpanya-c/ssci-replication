from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.tsa.stattools import adfuller


DISTRESS_BREAK = pd.Timestamp("2022-01-31")
EVENT_TAGS = {
    pd.Timestamp("2022-03-31"): "benchmark_2022_03",
    pd.Timestamp("2023-10-31"): "scenario_2023_10",
    pd.Timestamp("2026-02-28"): "scenario_2026_02",
}


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def dataset_root(self) -> Path:
        candidates = [self.root / "dataset", self.root.parent / "dataset"]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    @property
    def panel_path(self) -> Path:
        return self.dataset_root / "laos_fx_oil_macro_monthly_panel_2006-01-31_2026-02-28.csv"

    @property
    def gpr_components_path(self) -> Path:
        return self.dataset_root / "GPR_ME_middle_east_components_month_end_2006-01-31_2026-02-28.csv"


def load_panel(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)


def build_alt_gpr_series(components: pd.DataFrame) -> pd.DataFrame:
    frame = components.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    cols = ["Saudi Arabia", "Israel", "Turkey", "Egypt"]
    frame["GPR_ME_alt"] = frame[cols].mean(axis=1)
    return frame[["Date", "GPR_ME_alt"]].sort_values("Date").reset_index(drop=True)


def transform_panel(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date").reset_index(drop=True)
    frame["oil_shock"] = 100 * np.log(frame["Oil"]).diff()
    frame["fx_dep"] = 100 * np.log(frame["USDLAK"]).diff()
    frame["inflation_mom"] = 100 * np.log(frame["CPI"]).diff()
    frame["post_2022"] = (frame["Date"] >= DISTRESS_BREAK).astype(int)
    frame["regime"] = np.where(frame["post_2022"].eq(1), "post_2022", "pre_2022")
    frame["event_tag"] = frame["Date"].map(EVENT_TAGS).fillna("")
    return frame


def enrich_panel_with_alt_gpr(panel: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    alt = build_alt_gpr_series(components)
    merged = panel.merge(alt, on="Date", how="left")
    return merged


def summarize_stationarity(df: pd.DataFrame, variables: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for variable in variables:
        series = df[variable].dropna()
        stat, pvalue, *_ = adfuller(series, autolag="AIC")
        rows.append(
            {
                "variable": variable,
                "adf_stat": float(stat),
                "p_value": float(pvalue),
                "stationary_5pct": int(pvalue < 0.05),
                "nobs": int(series.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def regime_summary(
    df: pd.DataFrame,
    variables: Iterable[str] = ("GPR_ME", "oil_shock", "fx_dep", "inflation_mom", "CPI_Inf"),
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for regime, subset in (("pre_2022", df[df["Date"] < DISTRESS_BREAK]), ("post_2022", df[df["Date"] >= DISTRESS_BREAK])):
        clean = subset.dropna(subset=list(variables))
        for variable in variables:
            rows.append(
                {
                    "regime": regime,
                    "variable": variable,
                    "mean": float(clean[variable].mean()),
                    "std": float(clean[variable].std(ddof=1)),
                    "median": float(clean[variable].median()),
                    "nobs": int(clean[variable].shape[0]),
                }
            )
    return pd.DataFrame(rows)


def event_snapshot_table(df: pd.DataFrame, event_dates: Iterable[str | pd.Timestamp], window: int = 2) -> pd.DataFrame:
    frame = df.copy().reset_index(drop=True)
    rows: list[pd.DataFrame] = []
    for event_date in event_dates:
        ts = pd.Timestamp(event_date)
        idx = frame.index[frame["Date"] == ts]
        if len(idx) == 0:
            continue
        center = idx[0]
        lo = max(center - window, 0)
        hi = min(center + window, len(frame) - 1)
        window_frame = frame.loc[lo:hi, ["Date", "GPR_ME", "oil_shock", "fx_dep", "inflation_mom", "CPI_Inf"]].copy()
        window_frame.insert(1, "event_date", ts.strftime("%Y-%m-%d"))
        rows.append(window_frame)
    if not rows:
        return pd.DataFrame(columns=["Date", "event_date", "GPR_ME", "oil_shock", "fx_dep", "inflation_mom", "CPI_Inf"])
    return pd.concat(rows, ignore_index=True)


def break_diagnostics(
    df: pd.DataFrame,
    variables: Iterable[str] = ("GPR_ME", "oil_shock", "fx_dep", "inflation_mom", "CPI_Inf"),
) -> pd.DataFrame:
    pre = df[df["Date"] < DISTRESS_BREAK]
    post = df[df["Date"] >= DISTRESS_BREAK]
    rows: list[dict[str, float | str]] = []
    for variable in variables:
        pre_series = pre[variable].dropna()
        post_series = post[variable].dropna()
        t_stat, p_value, _ = ttest_ind(post_series, pre_series, usevar="unequal")
        rows.append(
            {
                "variable": variable,
                "pre_mean": float(pre_series.mean()),
                "post_mean": float(post_series.mean()),
                "difference_post_minus_pre": float(post_series.mean() - pre_series.mean()),
                "t_stat": float(t_stat),
                "p_value": float(p_value),
            }
        )
    return pd.DataFrame(rows)
