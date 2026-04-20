from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

from .data import DISTRESS_BREAK, load_panel
from .pipeline import DEFAULT_TVP_DATES, run_research_pipeline


DEFAULT_DIGITS = 3
VARIABLE_ORDER = ["GPR_ME", "oil_shock", "fx_dep", "inflation_mom", "CPI_Inf"]
VARIABLE_LABELS = {
    "GPR_ME": "Middle East geopolitical risk",
    "oil_shock": "Oil shock, 100*Δlog(Oil)",
    "fx_dep": "FX depreciation, 100*Δlog(USD/LAK)",
    "inflation_mom": "Monthly inflation, 100*Δlog(CPI)",
    "CPI_Inf": "CPI inflation, y/y",
}
REGIME_LABELS = {
    "pre_2022": "Pre-2022",
    "post_2022": "Post-2022",
}
CHANNEL_ORDER = [
    ("oil_shock", "fx_dep"),
    ("oil_shock", "inflation_mom"),
    ("fx_dep", "inflation_mom"),
]
CHANNEL_LABELS = {
    ("oil_shock", "fx_dep"): "Oil shock -> FX depreciation",
    ("oil_shock", "inflation_mom"): "Oil shock -> monthly inflation",
    ("fx_dep", "inflation_mom"): "FX depreciation -> monthly inflation",
}
CHECK_LABELS = {
    "baseline_fx_to_inflation_h0": "Baseline monthly inflation response at h=0",
    "alt_response_cpi_inf_h0": "Alternative response: CPI inflation (y/y) at h=0",
    "truncated_sample_fx_to_inflation_h0": "Truncated sample through 2025-12 at h=0",
    "alt_gpr_to_oil": "Alternative GPR specification in monthly GPR -> oil HAC OLS",
}
CORE_OUTPUT_RELATIVE_PATHS = {
    "processed_panel": Path("data") / "laos_fx_oil_macro_research_panel.csv",
    "regime_summary": Path("tables") / "regime_summary.csv",
    "break_diagnostics": Path("tables") / "break_diagnostics.csv",
    "lp_oil_to_fx": Path("tables") / "lp_oil_to_fx.csv",
    "lp_oil_to_inflation": Path("tables") / "lp_oil_to_inflation.csv",
    "lp_fx_to_inflation": Path("tables") / "lp_fx_to_inflation.csv",
    "lp_fx_to_cpi_inf": Path("tables") / "lp_fx_to_cpi_inf.csv",
    "lp_fx_to_inflation_truncated": Path("tables") / "lp_fx_to_inflation_truncated.csv",
    "robustness_summary": Path("tables") / "robustness_summary.csv",
    "gpr_to_oil": Path("tables") / "gpr_to_oil_hac_ols.csv",
    "gpr_alt_to_oil": Path("tables") / "gpr_alt_to_oil_hac_ols.csv",
    "tvp_irf": Path("models") / "tvp_svar_irf_summary.csv",
    "tvp_fit": Path("models") / "tvp_svar_fit_summary.csv",
}


@dataclass(frozen=True)
class SubmissionBundlePaths:
    project_root: Path
    output_root: Path
    paper_dir: Path
    paper_tables_dir: Path
    paper_figures_dir: Path
    results_summary_path: Path

    @property
    def processed_panel(self) -> Path:
        return self.output_root / CORE_OUTPUT_RELATIVE_PATHS["processed_panel"]

    @property
    def regime_summary(self) -> Path:
        return self.output_root / CORE_OUTPUT_RELATIVE_PATHS["regime_summary"]

    @property
    def break_diagnostics(self) -> Path:
        return self.output_root / CORE_OUTPUT_RELATIVE_PATHS["break_diagnostics"]

    @property
    def lp_oil_to_fx(self) -> Path:
        return self.output_root / CORE_OUTPUT_RELATIVE_PATHS["lp_oil_to_fx"]

    @property
    def lp_oil_to_inflation(self) -> Path:
        return self.output_root / CORE_OUTPUT_RELATIVE_PATHS["lp_oil_to_inflation"]

    @property
    def lp_fx_to_inflation(self) -> Path:
        return self.output_root / CORE_OUTPUT_RELATIVE_PATHS["lp_fx_to_inflation"]

    @property
    def lp_fx_to_cpi_inf(self) -> Path:
        return self.output_root / CORE_OUTPUT_RELATIVE_PATHS["lp_fx_to_cpi_inf"]

    @property
    def lp_fx_to_inflation_truncated(self) -> Path:
        return self.output_root / CORE_OUTPUT_RELATIVE_PATHS["lp_fx_to_inflation_truncated"]

    @property
    def robustness_summary(self) -> Path:
        return self.output_root / CORE_OUTPUT_RELATIVE_PATHS["robustness_summary"]

    @property
    def gpr_to_oil(self) -> Path:
        return self.output_root / CORE_OUTPUT_RELATIVE_PATHS["gpr_to_oil"]

    @property
    def gpr_alt_to_oil(self) -> Path:
        return self.output_root / CORE_OUTPUT_RELATIVE_PATHS["gpr_alt_to_oil"]

    @property
    def tvp_irf(self) -> Path:
        return self.output_root / CORE_OUTPUT_RELATIVE_PATHS["tvp_irf"]

    @property
    def tvp_fit(self) -> Path:
        return self.output_root / CORE_OUTPUT_RELATIVE_PATHS["tvp_fit"]


def build_submission_paths(
    project_root: str | Path,
    output_root: str | Path | None = None,
    paper_dir: str | Path | None = None,
) -> SubmissionBundlePaths:
    project = Path(project_root).resolve()
    output = (project / "output") if output_root is None else Path(output_root).resolve()
    paper = (project / "paper") if paper_dir is None else Path(paper_dir).resolve()
    return SubmissionBundlePaths(
        project_root=project,
        output_root=output,
        paper_dir=paper,
        paper_tables_dir=output / "paper_tables",
        paper_figures_dir=output / "paper_figures",
        results_summary_path=paper / "results_summary.md",
    )


def _ensure_bundle_dirs(paths: SubmissionBundlePaths) -> None:
    paths.paper_tables_dir.mkdir(parents=True, exist_ok=True)
    paths.paper_figures_dir.mkdir(parents=True, exist_ok=True)
    paths.paper_dir.mkdir(parents=True, exist_ok=True)


def _round_numeric_columns(frame: pd.DataFrame, digits: int = DEFAULT_DIGITS) -> pd.DataFrame:
    rounded = frame.copy()
    numeric_cols = rounded.select_dtypes(include=["number"]).columns
    rounded[numeric_cols] = rounded[numeric_cols].round(digits)
    return rounded


def _format_number(value: float | int, digits: int = DEFAULT_DIGITS) -> str:
    return f"{float(value):.{digits}f}"


def _format_interval(
    coefficient: float,
    lower: float,
    upper: float,
    digits: int = DEFAULT_DIGITS,
) -> str:
    return (
        f"{_format_number(coefficient, digits)} "
        f"[{_format_number(lower, digits)}, {_format_number(upper, digits)}]"
    )


def _escape_markdown(value: object) -> str:
    return str(value).replace("|", "\\|")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    df = _round_numeric_columns(frame)
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(_escape_markdown(value) for value in row) + " |"
        for row in df.astype(object).itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _write_table_artifacts(
    frame: pd.DataFrame,
    *,
    csv_path: Path,
    md_path: Path,
    title: str,
    note: str | None = None,
) -> tuple[Path, Path]:
    rounded = _round_numeric_columns(frame)
    rounded.to_csv(csv_path, index=False)

    lines = [f"# {title}", ""]
    if note:
        lines.extend([note, ""])
    lines.append(dataframe_to_markdown(rounded))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def _apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": 13,
        }
    )


def _relative_path(path: Path, anchor: Path) -> str:
    try:
        return str(path.relative_to(anchor))
    except ValueError:
        return str(path)


def _ensure_core_outputs(
    paths: SubmissionBundlePaths,
    *,
    force_pipeline: bool = False,
    tvp_draws: int = 1500,
    tvp_burnin: int = 500,
    tvp_thin: int = 5,
    tvp_horizons: int = 12,
    lp_horizons: int = 12,
    lp_lags: int = 2,
) -> dict[str, Path]:
    required = [paths.output_root / rel for rel in CORE_OUTPUT_RELATIVE_PATHS.values()]
    missing = [path for path in required if not path.exists()]
    if force_pipeline or missing:
        return run_research_pipeline(
            project_root=paths.project_root,
            output_root=paths.output_root,
            tvp_draws=tvp_draws,
            tvp_burnin=tvp_burnin,
            tvp_thin=tvp_thin,
            tvp_horizons=tvp_horizons,
            lp_horizons=lp_horizons,
            lp_lags=lp_lags,
            skip_tvp=False,
        )
    return {key: paths.output_root / rel for key, rel in CORE_OUTPUT_RELATIVE_PATHS.items()}


def _load_inputs(paths: SubmissionBundlePaths) -> dict[str, pd.DataFrame]:
    return {
        "processed_panel": load_panel(paths.processed_panel),
        "regime_summary": pd.read_csv(paths.regime_summary),
        "break_diagnostics": pd.read_csv(paths.break_diagnostics),
        "lp_oil_to_fx": pd.read_csv(paths.lp_oil_to_fx),
        "lp_oil_to_inflation": pd.read_csv(paths.lp_oil_to_inflation),
        "lp_fx_to_inflation": pd.read_csv(paths.lp_fx_to_inflation),
        "lp_fx_to_cpi_inf": pd.read_csv(paths.lp_fx_to_cpi_inf),
        "lp_fx_to_inflation_truncated": pd.read_csv(paths.lp_fx_to_inflation_truncated),
        "robustness_summary": pd.read_csv(paths.robustness_summary),
        "gpr_to_oil": pd.read_csv(paths.gpr_to_oil),
        "gpr_alt_to_oil": pd.read_csv(paths.gpr_alt_to_oil),
        "tvp_irf": pd.read_csv(paths.tvp_irf),
        "tvp_fit": pd.read_csv(paths.tvp_fit),
    }


def make_table_1_descriptive_stats(regime_summary: pd.DataFrame) -> pd.DataFrame:
    work = regime_summary.copy()
    work["variable"] = pd.Categorical(work["variable"], categories=VARIABLE_ORDER, ordered=True)
    work["regime"] = pd.Categorical(work["regime"], categories=["pre_2022", "post_2022"], ordered=True)
    work = work.sort_values(["variable", "regime"])

    rows: list[dict[str, object]] = []
    for variable in VARIABLE_ORDER:
        pre = work[(work["variable"] == variable) & (work["regime"] == "pre_2022")].iloc[0]
        post = work[(work["variable"] == variable) & (work["regime"] == "post_2022")].iloc[0]
        rows.append(
            {
                "Variable": VARIABLE_LABELS[variable],
                "Pre-2022 mean": pre["mean"],
                "Pre-2022 SD": pre["std"],
                "Pre-2022 median": pre["median"],
                "Pre-2022 N": int(pre["nobs"]),
                "Post-2022 mean": post["mean"],
                "Post-2022 SD": post["std"],
                "Post-2022 median": post["median"],
                "Post-2022 N": int(post["nobs"]),
            }
        )
    return pd.DataFrame(rows)


def make_table_2_break_diagnostics(break_diagnostics: pd.DataFrame) -> pd.DataFrame:
    work = break_diagnostics.copy()
    work["variable"] = pd.Categorical(work["variable"], categories=VARIABLE_ORDER, ordered=True)
    work = work.sort_values("variable")
    return pd.DataFrame(
        {
            "Variable": [VARIABLE_LABELS[variable] for variable in work["variable"]],
            "Pre-2022 mean": work["pre_mean"].to_list(),
            "Post-2022 mean": work["post_mean"].to_list(),
            "Difference (post - pre)": work["difference_post_minus_pre"].to_list(),
            "t-statistic": work["t_stat"].to_list(),
            "p-value": work["p_value"].to_list(),
        }
    )


def make_table_3_fx_pass_through_lp(lp_fx_to_inflation: pd.DataFrame) -> pd.DataFrame:
    work = lp_fx_to_inflation.copy()
    work = work.sort_values(["horizon", "regime"])
    rows: list[dict[str, object]] = []
    for horizon in sorted(work["horizon"].unique()):
        pre = work[(work["horizon"] == horizon) & (work["regime"] == "pre_2022")].iloc[0]
        post = work[(work["horizon"] == horizon) & (work["regime"] == "post_2022")].iloc[0]
        rows.append(
            {
                "Horizon (months)": int(horizon),
                "Pre-2022 estimate [90% CI]": _format_interval(
                    pre["coefficient"], pre["lower_90"], pre["upper_90"]
                ),
                "Pre-2022 N": int(pre["nobs"]),
                "Post-2022 estimate [90% CI]": _format_interval(
                    post["coefficient"], post["lower_90"], post["upper_90"]
                ),
                "Post-2022 N": int(post["nobs"]),
            }
        )
    return pd.DataFrame(rows)


def make_table_4_robustness(
    robustness_summary: pd.DataFrame,
    gpr_alt_to_oil: pd.DataFrame | None = None,
) -> pd.DataFrame:
    work = robustness_summary.copy()
    rows: list[dict[str, object]] = []

    for check in [
        "baseline_fx_to_inflation_h0",
        "alt_response_cpi_inf_h0",
        "truncated_sample_fx_to_inflation_h0",
    ]:
        subset = work[work["check"] == check].copy()
        for regime in ["pre_2022", "post_2022"]:
            row = subset[subset["regime"] == regime].iloc[0]
            rows.append(
                {
                    "Specification": CHECK_LABELS[check],
                    "Regime": REGIME_LABELS[regime],
                    "Effect summary": _format_interval(row["coefficient"], row["lower_90"], row["upper_90"]),
                    "N": int(row["nobs"]),
                }
            )

    if gpr_alt_to_oil is None:
        subset = work[work["check"] == "alt_gpr_to_oil"].copy()
        for regime in ["pre_2022", "post_2022"]:
            row = subset[subset["regime"] == regime].iloc[0]
            rows.append(
                {
                    "Specification": CHECK_LABELS["alt_gpr_to_oil"],
                    "Regime": REGIME_LABELS[regime],
                    "Effect summary": (
                        f"{_format_number(row['coefficient'])} "
                        f"(SE {_format_number(row['lower_90'])}, p={_format_number(row['upper_90'])})"
                    ),
                    "N": int(row["nobs"]),
                }
            )
    else:
        subset = gpr_alt_to_oil[gpr_alt_to_oil["term"] == "GPR_ME_alt"].copy()
        for regime in ["pre_2022", "post_2022"]:
            row = subset[subset["regime"] == regime].iloc[0]
            rows.append(
                {
                    "Specification": CHECK_LABELS["alt_gpr_to_oil"],
                    "Regime": REGIME_LABELS[regime],
                    "Effect summary": (
                        f"{_format_number(row['coefficient'])} "
                        f"(SE {_format_number(row['std_error'])}, p={_format_number(row['p_value'])})"
                    ),
                    "N": int(row["nobs"]),
                }
            )
    return pd.DataFrame(rows)


def make_table_5_tvp_svar_scenarios(irf_summary: pd.DataFrame) -> pd.DataFrame:
    work = irf_summary.copy()
    work = work[
        work.apply(lambda row: (row["impulse"], row["response"]) in CHANNEL_ORDER, axis=1)
    ].copy()

    rows: list[dict[str, object]] = []
    for date in DEFAULT_TVP_DATES:
        for impulse, response in CHANNEL_ORDER:
            subset = work[(work["date"] == date) & (work["impulse"] == impulse) & (work["response"] == response)].copy()
            if subset.empty:
                continue
            impact = subset[subset["horizon"] == 0].iloc[0]
            peak_idx = subset["median"].abs().idxmax()
            peak = subset.loc[peak_idx]
            rows.append(
                {
                    "Scenario date": date,
                    "Channel": CHANNEL_LABELS[(impulse, response)],
                    "Impact median [90% CI]": _format_interval(
                        impact["median"], impact["lower_90"], impact["upper_90"]
                    ),
                    "Peak median [90% CI]": _format_interval(
                        peak["median"], peak["lower_90"], peak["upper_90"]
                    ),
                    "Peak horizon (months)": int(peak["horizon"]),
                }
            )
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def export_figure_1_macro_context(panel: pd.DataFrame, path: Path) -> Path:
    _apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    plot_vars = [
        ("GPR_ME", "Middle East geopolitical risk"),
        ("oil_shock", "Oil shock, 100*Δlog(Oil)"),
        ("fx_dep", "FX depreciation, 100*Δlog(USD/LAK)"),
        ("inflation_mom", "Monthly inflation, 100*Δlog(CPI)"),
    ]
    for ax, (column, title) in zip(axes.flat, plot_vars):
        ax.plot(panel["Date"], panel[column], color="#1f4b99", linewidth=1.7)
        ax.axvline(DISTRESS_BREAK, color="#b22222", linestyle="--", linewidth=1.2)
        ax.set_title(title)
        ax.set_ylabel("Percent")
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Date")
    fig.autofmt_xdate()
    return _save_figure(fig, path)


def export_figure_2_crisis_overlay(panel: pd.DataFrame, path: Path) -> Path:
    _apply_paper_style()
    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    line1 = ax1.plot(
        panel["Date"],
        panel["CPI_Inf"],
        color="#b22222",
        linewidth=2.0,
        label="CPI inflation (y/y)",
    )
    ax1.axvline(DISTRESS_BREAK, color="#444444", linestyle="--", linewidth=1.2)
    ax1.set_ylabel("CPI inflation, percent")
    ax1.set_xlabel("Date")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        panel["Date"],
        panel["USDLAK"],
        color="#1f4b99",
        linewidth=1.8,
        label="USD/LAK",
    )
    ax2.set_ylabel("LAK per USD")

    handles = line1 + line2
    labels = [handle.get_label() for handle in handles]
    ax1.legend(handles, labels, frameon=False, loc="upper left")
    fig.autofmt_xdate()
    return _save_figure(fig, path)


def _export_lp_figure(lp: pd.DataFrame, *, title: str, y_label: str, path: Path) -> Path:
    _apply_paper_style()
    colors = {"pre_2022": "#1f4b99", "post_2022": "#b22222"}
    labels = {"pre_2022": "Pre-2022", "post_2022": "Post-2022"}

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for regime in ["pre_2022", "post_2022"]:
        subset = lp[lp["regime"] == regime].sort_values("horizon")
        ax.plot(
            subset["horizon"],
            subset["coefficient"],
            label=labels[regime],
            color=colors[regime],
            linewidth=2.0,
        )
        ax.fill_between(
            subset["horizon"],
            subset["lower_90"],
            subset["upper_90"],
            color=colors[regime],
            alpha=0.18,
        )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Horizon (months)")
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    return _save_figure(fig, path)


def export_figure_3_oil_to_fx_lp(lp_oil_to_fx: pd.DataFrame, path: Path) -> Path:
    return _export_lp_figure(
        lp_oil_to_fx,
        title="Figure 3. Local projections: oil shocks and FX depreciation",
        y_label="Response of FX depreciation",
        path=path,
    )


def export_figure_4_oil_to_inflation_lp(lp_oil_to_inflation: pd.DataFrame, path: Path) -> Path:
    return _export_lp_figure(
        lp_oil_to_inflation,
        title="Figure 4. Local projections: oil shocks and monthly inflation",
        y_label="Response of monthly inflation",
        path=path,
    )


def export_figure_5_fx_to_inflation_lp(lp_fx_to_inflation: pd.DataFrame, path: Path) -> Path:
    return _export_lp_figure(
        lp_fx_to_inflation,
        title="Figure 5. Local projections: FX depreciation and monthly inflation",
        y_label="Response of monthly inflation",
        path=path,
    )


def export_figure_6_tvp_svar_scenarios(irf_summary: pd.DataFrame, path: Path) -> Path:
    _apply_paper_style()
    colors = {
        "2022-03-31": "#b22222",
        "2023-10-31": "#1f4b99",
        "2026-02-28": "#444444",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    for ax, (impulse, response) in zip(axes, CHANNEL_ORDER):
        for date in DEFAULT_TVP_DATES:
            subset = irf_summary[
                (irf_summary["date"] == date)
                & (irf_summary["impulse"] == impulse)
                & (irf_summary["response"] == response)
            ].sort_values("horizon")
            ax.plot(subset["horizon"], subset["median"], color=colors[date], linewidth=2.0, label=date)
            ax.fill_between(
                subset["horizon"],
                subset["lower_90"],
                subset["upper_90"],
                color=colors[date],
                alpha=0.12,
            )
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(CHANNEL_LABELS[(impulse, response)])
        ax.set_xlabel("Horizon (months)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Response")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    return _save_figure(fig, path)


def build_results_summary(inputs: dict[str, pd.DataFrame], paths: SubmissionBundlePaths) -> str:
    breaks = inputs["break_diagnostics"].set_index("variable")
    lp_fx = inputs["lp_fx_to_inflation"]
    lp_oil_fx = inputs["lp_oil_to_fx"]
    lp_oil_inflation = inputs["lp_oil_to_inflation"]
    lp_cpi = inputs["lp_fx_to_cpi_inf"]
    lp_truncated = inputs["lp_fx_to_inflation_truncated"]
    gpr_to_oil = inputs["gpr_to_oil"]
    gpr_alt_to_oil = inputs["gpr_alt_to_oil"]
    irf = inputs["tvp_irf"]

    fx_pre_h0 = lp_fx[(lp_fx["regime"] == "pre_2022") & (lp_fx["horizon"] == 0)].iloc[0]
    fx_post_h0 = lp_fx[(lp_fx["regime"] == "post_2022") & (lp_fx["horizon"] == 0)].iloc[0]
    fx_post_h1 = lp_fx[(lp_fx["regime"] == "post_2022") & (lp_fx["horizon"] == 1)].iloc[0]

    oil_fx_post_h0 = lp_oil_fx[(lp_oil_fx["regime"] == "post_2022") & (lp_oil_fx["horizon"] == 0)].iloc[0]
    oil_fx_post_h2 = lp_oil_fx[(lp_oil_fx["regime"] == "post_2022") & (lp_oil_fx["horizon"] == 2)].iloc[0]
    oil_inflation_post_h3 = lp_oil_inflation[
        (lp_oil_inflation["regime"] == "post_2022") & (lp_oil_inflation["horizon"] == 3)
    ].iloc[0]

    cpi_post_h1 = lp_cpi[(lp_cpi["regime"] == "post_2022") & (lp_cpi["horizon"] == 1)].iloc[0]
    truncated_post_h0 = lp_truncated[
        (lp_truncated["regime"] == "post_2022") & (lp_truncated["horizon"] == 0)
    ].iloc[0]

    gpr_pre = gpr_to_oil[(gpr_to_oil["term"] == "GPR_ME") & (gpr_to_oil["regime"] == "pre_2022")].iloc[0]
    gpr_post = gpr_to_oil[(gpr_to_oil["term"] == "GPR_ME") & (gpr_to_oil["regime"] == "post_2022")].iloc[0]
    gpr_alt_pre = gpr_alt_to_oil[
        (gpr_alt_to_oil["term"] == "GPR_ME_alt") & (gpr_alt_to_oil["regime"] == "pre_2022")
    ].iloc[0]
    gpr_alt_post = gpr_alt_to_oil[
        (gpr_alt_to_oil["term"] == "GPR_ME_alt") & (gpr_alt_to_oil["regime"] == "post_2022")
    ].iloc[0]

    def irf_h0(date: str, impulse: str, response: str) -> pd.Series:
        return irf[
            (irf["date"] == date)
            & (irf["impulse"] == impulse)
            & (irf["response"] == response)
            & (irf["horizon"] == 0)
        ].iloc[0]

    scenario_2022_fx = irf_h0("2022-03-31", "oil_shock", "fx_dep")
    scenario_2023_fx = irf_h0("2023-10-31", "oil_shock", "fx_dep")
    scenario_2026_fx = irf_h0("2026-02-28", "oil_shock", "fx_dep")
    scenario_2022_infl = irf_h0("2022-03-31", "oil_shock", "inflation_mom")
    scenario_2023_infl = irf_h0("2023-10-31", "oil_shock", "inflation_mom")
    scenario_2026_infl = irf_h0("2026-02-28", "oil_shock", "inflation_mom")

    lines = [
        "# Results Summary",
        "",
        "This file is generated from the current output tables and model summaries. It is intended as a drafting aid for the manuscript and submission package, not as a replacement for the underlying analysis files.",
        "",
        "## Key empirical numbers to cite",
        "",
        (
            f"- Break diagnostics: mean FX depreciation rises from {_format_number(breaks.loc['fx_dep', 'pre_mean'])} "
            f"to {_format_number(breaks.loc['fx_dep', 'post_mean'])}, while mean monthly inflation rises from "
            f"{_format_number(breaks.loc['inflation_mom', 'pre_mean'])} to "
            f"{_format_number(breaks.loc['inflation_mom', 'post_mean'])} and mean CPI inflation (y/y) rises from "
            f"{_format_number(breaks.loc['CPI_Inf', 'pre_mean'])} to {_format_number(breaks.loc['CPI_Inf', 'post_mean'])}."
        ),
        (
            f"- Benchmark fx_dep -> inflation_mom pass-through at horizon 0 rises from "
            f"{_format_interval(fx_pre_h0['coefficient'], fx_pre_h0['lower_90'], fx_pre_h0['upper_90'])} "
            f"in the pre-2022 sample to "
            f"{_format_interval(fx_post_h0['coefficient'], fx_post_h0['lower_90'], fx_post_h0['upper_90'])} "
            f"in the post-2022 sample. The post-2022 horizon-1 estimate is "
            f"{_format_interval(fx_post_h1['coefficient'], fx_post_h1['lower_90'], fx_post_h1['upper_90'])}."
        ),
        (
            f"- Oil_shock -> fx_dep becomes positive in the stressed regime, with post-2022 estimates of "
            f"{_format_interval(oil_fx_post_h0['coefficient'], oil_fx_post_h0['lower_90'], oil_fx_post_h0['upper_90'])} "
            f"at horizon 0 and "
            f"{_format_interval(oil_fx_post_h2['coefficient'], oil_fx_post_h2['lower_90'], oil_fx_post_h2['upper_90'])} "
            f"at horizon 2."
        ),
        (
            f"- Oil_shock -> inflation_mom remains positive in the stressed regime and reaches "
            f"{_format_interval(oil_inflation_post_h3['coefficient'], oil_inflation_post_h3['lower_90'], oil_inflation_post_h3['upper_90'])} "
            f"by horizon 3."
        ),
        (
            f"- TVP-SV-VAR impact responses are strongest on 2022-03-31 for oil_shock -> fx_dep "
            f"({_format_interval(scenario_2022_fx['median'], scenario_2022_fx['lower_90'], scenario_2022_fx['upper_90'])}) "
            f"and oil_shock -> inflation_mom "
            f"({_format_interval(scenario_2022_infl['median'], scenario_2022_infl['lower_90'], scenario_2022_infl['upper_90'])}). "
            f"By 2023-10-31 they are smaller but still positive, and by 2026-02-28 they are near zero "
            f"for oil_shock -> fx_dep ({_format_interval(scenario_2026_fx['median'], scenario_2026_fx['lower_90'], scenario_2026_fx['upper_90'])}) "
            f"and oil_shock -> inflation_mom ({_format_interval(scenario_2026_infl['median'], scenario_2026_infl['lower_90'], scenario_2026_infl['upper_90'])})."
        ),
        "",
        "## Interpretation",
        "",
        (
            "The strongest supported result remains the post-2022 amplification of exchange-rate pass-through. "
            "In the benchmark local projections, depreciation becomes far more inflationary after the January 2022 break, "
            "which is the cleanest evidence for the paper's debt-distress-amplified pass-through argument."
        ),
        "",
        (
            "The oil shock channel matters mainly through the exchange rate and only in the stressed regime. "
            "That is why the manuscript should continue to frame the contribution as state-dependent shock transmission "
            "rather than as a broad claim that geopolitical risk directly causes Laos inflation every month."
        ),
        "",
        "## Robustness summary",
        "",
        (
            f"- Alternative inflation response: post-2022 fx_dep -> CPI_Inf at horizon 1 is "
            f"{_format_interval(cpi_post_h1['coefficient'], cpi_post_h1['lower_90'], cpi_post_h1['upper_90'])}, "
            "which preserves the stronger stressed-regime pass-through result on a slower-moving inflation measure."
        ),
        (
            f"- Truncated sample check: post-2022 fx_dep -> inflation_mom at horizon 0 remains "
            f"{_format_interval(truncated_post_h0['coefficient'], truncated_post_h0['lower_90'], truncated_post_h0['upper_90'])} "
            "when the sample is truncated at 2025-12."
        ),
        (
            f"- Alternative GPR construction: the monthly GPR_ME_alt -> oil estimate is "
            f"{_format_number(gpr_alt_pre['coefficient'])} (SE {_format_number(gpr_alt_pre['std_error'])}, p={_format_number(gpr_alt_pre['p_value'])}) "
            f"in the pre-2022 sample and {_format_number(gpr_alt_post['coefficient'])} "
            f"(SE {_format_number(gpr_alt_post['std_error'])}, p={_format_number(gpr_alt_post['p_value'])}) in the post-2022 sample, "
            "so the alternative construction does not materially strengthen the direct monthly GPR-to-oil relation."
        ),
        "",
        "## What should not be overstated",
        "",
        (
            f"- The direct reduced-form monthly GPR -> oil link is weak: the GPR_ME coefficient is "
            f"{_format_number(gpr_pre['coefficient'])} (p={_format_number(gpr_pre['p_value'])}) before 2022 and "
            f"{_format_number(gpr_post['coefficient'])} (p={_format_number(gpr_post['p_value'])}) after 2022."
        ),
        "- The main contribution is not a sharp causal identification of Middle East conflict on Laos inflation.",
        "- The February 2026 scenario is a sample-endpoint IRF date, not realized ex post evidence.",
        "",
        "## Generated paper-ready outputs",
        "",
        f"- Paper tables directory: `{_relative_path(paths.paper_tables_dir, paths.project_root)}`",
        f"- Paper figures directory: `{_relative_path(paths.paper_figures_dir, paths.project_root)}`",
        "",
    ]
    return "\n".join(lines)


def build_submission_bundle(
    project_root: str | Path,
    *,
    output_root: str | Path | None = None,
    paper_dir: str | Path | None = None,
    force_pipeline: bool = False,
    tvp_draws: int = 1500,
    tvp_burnin: int = 500,
    tvp_thin: int = 5,
    tvp_horizons: int = 12,
    lp_horizons: int = 12,
    lp_lags: int = 2,
) -> dict[str, Path]:
    paths = build_submission_paths(project_root, output_root=output_root, paper_dir=paper_dir)
    _ensure_bundle_dirs(paths)
    _ensure_core_outputs(
        paths,
        force_pipeline=force_pipeline,
        tvp_draws=tvp_draws,
        tvp_burnin=tvp_burnin,
        tvp_thin=tvp_thin,
        tvp_horizons=tvp_horizons,
        lp_horizons=lp_horizons,
        lp_lags=lp_lags,
    )
    inputs = _load_inputs(paths)

    table_1 = make_table_1_descriptive_stats(inputs["regime_summary"])
    table_2 = make_table_2_break_diagnostics(inputs["break_diagnostics"])
    table_3 = make_table_3_fx_pass_through_lp(inputs["lp_fx_to_inflation"])
    table_4 = make_table_4_robustness(inputs["robustness_summary"], inputs["gpr_alt_to_oil"])
    table_5 = make_table_5_tvp_svar_scenarios(inputs["tvp_irf"])

    manifest: dict[str, Path] = {}

    csv_path, md_path = _write_table_artifacts(
        table_1,
        csv_path=paths.paper_tables_dir / "table_1_descriptive_stats.csv",
        md_path=paths.paper_tables_dir / "table_1_descriptive_stats.md",
        title="Table 1. Descriptive statistics by regime",
        note="Break date fixed at January 2022.",
    )
    manifest["table_1_descriptive_stats_csv"] = csv_path
    manifest["table_1_descriptive_stats_md"] = md_path

    csv_path, md_path = _write_table_artifacts(
        table_2,
        csv_path=paths.paper_tables_dir / "table_2_break_diagnostics.csv",
        md_path=paths.paper_tables_dir / "table_2_break_diagnostics.md",
        title="Table 2. Break diagnostics around January 2022",
        note="Difference reported as post-2022 minus pre-2022.",
    )
    manifest["table_2_break_diagnostics_csv"] = csv_path
    manifest["table_2_break_diagnostics_md"] = md_path

    csv_path, md_path = _write_table_artifacts(
        table_3,
        csv_path=paths.paper_tables_dir / "table_3_fx_pass_through_lp.csv",
        md_path=paths.paper_tables_dir / "table_3_fx_pass_through_lp.md",
        title="Table 3. FX pass-through local projections",
        note="Entries show coefficient estimates with 90 percent confidence intervals.",
    )
    manifest["table_3_fx_pass_through_lp_csv"] = csv_path
    manifest["table_3_fx_pass_through_lp_md"] = md_path

    csv_path, md_path = _write_table_artifacts(
        table_4,
        csv_path=paths.paper_tables_dir / "table_4_robustness.csv",
        md_path=paths.paper_tables_dir / "table_4_robustness.md",
        title="Table 4. Robustness checks",
        note="LP rows report 90 percent confidence intervals; the alternative GPR row reports HAC standard errors and p-values.",
    )
    manifest["table_4_robustness_csv"] = csv_path
    manifest["table_4_robustness_md"] = md_path

    csv_path, md_path = _write_table_artifacts(
        table_5,
        csv_path=paths.paper_tables_dir / "table_5_tvp_svar_scenarios.csv",
        md_path=paths.paper_tables_dir / "table_5_tvp_svar_scenarios.md",
        title="Table 5. TVP-SV-VAR scenario summaries",
        note="Channels restricted to the three supported transmission links used in the paper narrative.",
    )
    manifest["table_5_tvp_svar_scenarios_csv"] = csv_path
    manifest["table_5_tvp_svar_scenarios_md"] = md_path

    panel_for_context = inputs["processed_panel"].dropna(subset=["oil_shock", "fx_dep", "inflation_mom"]).copy()
    manifest["figure_1_macro_context"] = export_figure_1_macro_context(
        panel_for_context, paths.paper_figures_dir / "figure_1_macro_context.png"
    )
    manifest["figure_2_crisis_overlay"] = export_figure_2_crisis_overlay(
        inputs["processed_panel"], paths.paper_figures_dir / "figure_2_crisis_overlay.png"
    )
    manifest["figure_3_oil_to_fx_lp"] = export_figure_3_oil_to_fx_lp(
        inputs["lp_oil_to_fx"], paths.paper_figures_dir / "figure_3_oil_to_fx_lp.png"
    )
    manifest["figure_4_oil_to_inflation_lp"] = export_figure_4_oil_to_inflation_lp(
        inputs["lp_oil_to_inflation"], paths.paper_figures_dir / "figure_4_oil_to_inflation_lp.png"
    )
    manifest["figure_5_fx_to_inflation_lp"] = export_figure_5_fx_to_inflation_lp(
        inputs["lp_fx_to_inflation"], paths.paper_figures_dir / "figure_5_fx_to_inflation_lp.png"
    )
    manifest["figure_6_tvp_svar_scenarios"] = export_figure_6_tvp_svar_scenarios(
        inputs["tvp_irf"], paths.paper_figures_dir / "figure_6_tvp_svar_scenarios.png"
    )

    results_summary = build_results_summary(inputs, paths)
    paths.results_summary_path.write_text(results_summary, encoding="utf-8")
    manifest["results_summary"] = paths.results_summary_path

    return manifest
