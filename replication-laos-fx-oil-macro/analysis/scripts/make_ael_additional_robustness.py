from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from laos_fx_oil_macro.data import DISTRESS_BREAK, ProjectPaths, load_panel, transform_panel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATHS = ProjectPaths(PROJECT_ROOT)
OUTPUT_TABLES = PROJECT_ROOT / "output" / "paper_tables"
OUTPUT_FIGURES = PROJECT_ROOT / "output" / "paper_figures"


def run_interacted_local_projection(
    df: pd.DataFrame,
    *,
    response: str,
    impulse: str,
    controls: list[str],
    break_date: pd.Timestamp,
    horizons: int = 12,
    lags: int = 2,
) -> pd.DataFrame:
    work = df.copy()
    work["post_break"] = (work["Date"] >= break_date).astype(int)
    work["interaction"] = work[impulse] * work["post_break"]

    rows: list[dict[str, float | int | str]] = []
    for horizon in range(horizons + 1):
        sample = work.copy()
        lead = f"{response}_lead_{horizon}"
        sample[lead] = sample[response].shift(-horizon)
        regressors = [impulse, "post_break", "interaction", *controls]
        for variable in [response, impulse, *controls]:
            for lag in range(1, lags + 1):
                lag_name = f"{variable}_lag_{lag}"
                sample[lag_name] = sample[variable].shift(lag)
                regressors.append(lag_name)
        sample = sample.dropna(subset=[lead, *regressors])
        y = sample[lead]
        X = sm.add_constant(sample[regressors])
        fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max(lags, horizon + 1)})
        base = float(fit.params[impulse])
        interaction = float(fit.params["interaction"])
        interaction_se = float(fit.bse["interaction"])
        band = 1.645 * interaction_se
        rows.append(
            {
                "break_date": break_date.strftime("%Y-%m-%d"),
                "horizon": horizon,
                "pre_effect": base,
                "interaction": interaction,
                "interaction_lower_90": interaction - band,
                "interaction_upper_90": interaction + band,
                "interaction_p_value": float(fit.pvalues["interaction"]),
                "post_effect": base + interaction,
                "nobs": int(fit.nobs),
            }
        )
    return pd.DataFrame(rows)


def make_placebo_scan(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for break_date in pd.date_range("2021-06-30", "2022-06-30", freq="ME"):
        rows.append(
            run_interacted_local_projection(
                panel,
                response="inflation_mom",
                impulse="fx_dep",
                controls=["oil_shock", "GPR_ME", "FFR", "GSCPI"],
                break_date=break_date,
                horizons=1,
                lags=2,
            )
        )
    return pd.concat(rows, ignore_index=True)


def export_placebo_figure(scan: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    colors = {0: "#8c1d40", 1: "#1f4e79"}
    labels = {0: "h=0 interaction", 1: "h=1 interaction"}
    work = scan.copy()
    work["break_date"] = pd.to_datetime(work["break_date"])
    for horizon in [0, 1]:
        subset = work[work["horizon"] == horizon].sort_values("break_date")
        ax.plot(
            subset["break_date"],
            subset["interaction"],
            marker="o",
            linewidth=1.8,
            markersize=4.0,
            color=colors[horizon],
            label=labels[horizon],
        )
        ax.fill_between(
            subset["break_date"],
            subset["interaction_lower_90"],
            subset["interaction_upper_90"],
            color=colors[horizon],
            alpha=0.15,
        )
    ax.axvline(DISTRESS_BREAK, color="#444444", linestyle="--", linewidth=1.1)
    ax.axhline(0, color="#777777", linewidth=0.9)
    ax.set_ylabel("Interaction coefficient")
    ax.set_xlabel("Candidate break date")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

    panel = transform_panel(load_panel(PATHS.panel_path))

    interaction = run_interacted_local_projection(
        panel,
        response="inflation_mom",
        impulse="fx_dep",
        controls=["oil_shock", "GPR_ME", "FFR", "GSCPI"],
        break_date=DISTRESS_BREAK,
        horizons=12,
        lags=2,
    )
    interaction.to_csv(OUTPUT_TABLES / "table_6_interaction_lp.csv", index=False)

    placebo = make_placebo_scan(panel)
    placebo.to_csv(OUTPUT_TABLES / "table_7_placebo_break_scan.csv", index=False)
    export_placebo_figure(placebo, OUTPUT_FIGURES / "figure_ael_placebo_break.png")


if __name__ == "__main__":
    main()
