from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from .data import (
    ProjectPaths,
    break_diagnostics,
    enrich_panel_with_alt_gpr,
    event_snapshot_table,
    load_panel,
    regime_summary,
    summarize_stationarity,
    transform_panel,
)
from .models import run_hac_ols, run_local_projection


DEFAULT_TVP_DATES = ("2022-03-31", "2023-10-31", "2026-02-28")


def _ensure_output_dirs(output_root: Path) -> dict[str, Path]:
    dirs = {
        "data": output_root / "data",
        "tables": output_root / "tables",
        "figures": output_root / "figures",
        "models": output_root / "models",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


def _plot_macro_context(panel: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    plot_vars = [
        ("GPR_ME", "Middle East geopolitical risk"),
        ("oil_shock", "Oil shock, 100*Δlog(Oil)"),
        ("fx_dep", "FX depreciation, 100*Δlog(USDLAK)"),
        ("inflation_mom", "Monthly inflation, 100*Δlog(CPI)"),
    ]
    for ax, (column, title) in zip(axes.flat, plot_vars):
        ax.plot(panel["Date"], panel[column], color="#1f4b99", linewidth=1.5)
        ax.axvline(pd.Timestamp("2022-01-31"), color="#b22222", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_crisis_overlay(panel: pd.DataFrame, path: Path) -> Path:
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(panel["Date"], panel["CPI_Inf"], color="#b22222", linewidth=1.8, label="CPI inflation (y/y)")
    ax1.set_ylabel("Percent")
    ax1.axvline(pd.Timestamp("2022-01-31"), color="#444444", linestyle="--", linewidth=1)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(panel["Date"], panel["USDLAK"], color="#1f4b99", linewidth=1.4, label="USD/LAK")
    ax2.set_ylabel("LAK per USD")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_local_projection(lp: pd.DataFrame, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"pre_2022": "#1f4b99", "post_2022": "#b22222", "full_sample": "#444444"}
    for regime, subset in lp.groupby("regime"):
        subset = subset.sort_values("horizon")
        ax.plot(subset["horizon"], subset["coefficient"], label=regime, color=colors.get(regime, "#444444"))
        ax.fill_between(
            subset["horizon"],
            subset["lower_90"],
            subset["upper_90"],
            alpha=0.2,
            color=colors.get(regime, "#444444"),
        )
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Horizon (months)")
    ax.set_ylabel("Response")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _run_regime_local_projections(
    panel: pd.DataFrame,
    response: str,
    impulse: str,
    controls: Iterable[str],
    horizons: int,
    lags: int,
) -> pd.DataFrame:
    pieces = []
    for regime in ("pre_2022", "post_2022"):
        pieces.append(
            run_local_projection(
                panel,
                response=response,
                impulse=impulse,
                controls=controls,
                horizons=horizons,
                lags=lags,
                regime=regime,
            )
        )
    return pd.concat(pieces, ignore_index=True)


def _run_tvp_svar(
    project_root: Path,
    processed_path: Path,
    models_dir: Path,
    draws: int,
    burnin: int,
    thin: int,
    horizons: int,
    dates: Iterable[str],
) -> tuple[Path, Path]:
    script = project_root / "scripts" / "run_tvp_svar.R"
    cmd = [
        "Rscript",
        str(script),
        "--input",
        str(processed_path),
        "--output",
        str(models_dir),
        "--draws",
        str(draws),
        "--burnin",
        str(burnin),
        "--thin",
        str(thin),
        "--horizons",
        str(horizons),
        "--dates",
        ",".join(dates),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"TVP-SV-VAR run failed:\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
    return models_dir / "tvp_svar_irf_summary.csv", models_dir / "tvp_svar_fit_summary.csv"


def run_research_pipeline(
    project_root: str | Path,
    output_root: str | Path | None = None,
    tvp_draws: int = 1500,
    tvp_burnin: int = 500,
    tvp_thin: int = 5,
    tvp_horizons: int = 12,
    lp_horizons: int = 12,
    lp_lags: int = 2,
    skip_tvp: bool = False,
) -> dict[str, Path]:
    project_root = Path(project_root).resolve()
    output_root = Path(output_root).resolve() if output_root is not None else project_root / "output"
    dirs = _ensure_output_dirs(output_root)

    paths = ProjectPaths(project_root)
    raw_panel = load_panel(paths.panel_path)
    gpr_components = load_panel(paths.gpr_components_path)
    panel = enrich_panel_with_alt_gpr(transform_panel(raw_panel), gpr_components)

    processed_path = _write_csv(panel, dirs["data"] / "laos_fx_oil_macro_research_panel.csv")

    stationarity = summarize_stationarity(
        panel,
        ["USDLAK", "CPI", "CPI_Inf", "GPR_ME", "GPR_ME_alt", "oil_shock", "fx_dep", "inflation_mom"],
    )
    regime = regime_summary(panel)
    breaks = break_diagnostics(panel)
    event_snapshots = event_snapshot_table(panel, DEFAULT_TVP_DATES, window=2)

    gpr_to_oil = pd.concat(
        [
            run_hac_ols(panel, "oil_shock", ["GPR_ME", "FFR", "GSCPI"], regime="pre_2022"),
            run_hac_ols(panel, "oil_shock", ["GPR_ME", "FFR", "GSCPI"], regime="post_2022"),
        ],
        ignore_index=True,
    )
    gpr_alt_to_oil = pd.concat(
        [
            run_hac_ols(panel, "oil_shock", ["GPR_ME_alt", "FFR", "GSCPI"], regime="pre_2022"),
            run_hac_ols(panel, "oil_shock", ["GPR_ME_alt", "FFR", "GSCPI"], regime="post_2022"),
        ],
        ignore_index=True,
    )

    lp_oil_to_fx = _run_regime_local_projections(
        panel,
        response="fx_dep",
        impulse="oil_shock",
        controls=["GPR_ME", "FFR", "GSCPI", "inflation_mom"],
        horizons=lp_horizons,
        lags=lp_lags,
    )
    lp_oil_to_inflation = _run_regime_local_projections(
        panel,
        response="inflation_mom",
        impulse="oil_shock",
        controls=["fx_dep", "GPR_ME", "FFR", "GSCPI"],
        horizons=lp_horizons,
        lags=lp_lags,
    )
    lp_fx_to_inflation = _run_regime_local_projections(
        panel,
        response="inflation_mom",
        impulse="fx_dep",
        controls=["oil_shock", "GPR_ME", "FFR", "GSCPI"],
        horizons=lp_horizons,
        lags=lp_lags,
    )
    lp_fx_to_cpi_inf = _run_regime_local_projections(
        panel,
        response="CPI_Inf",
        impulse="fx_dep",
        controls=["oil_shock", "GPR_ME", "FFR", "GSCPI"],
        horizons=lp_horizons,
        lags=lp_lags,
    )
    truncated_panel = panel[panel["Date"] <= pd.Timestamp("2025-12-31")].copy()
    lp_fx_to_inflation_truncated = _run_regime_local_projections(
        truncated_panel,
        response="inflation_mom",
        impulse="fx_dep",
        controls=["oil_shock", "GPR_ME", "FFR", "GSCPI"],
        horizons=lp_horizons,
        lags=lp_lags,
    )
    robustness_summary = pd.concat(
        [
            lp_fx_to_inflation.query("horizon == 0").assign(check="baseline_fx_to_inflation_h0")[
                ["check", "regime", "coefficient", "lower_90", "upper_90", "nobs"]
            ],
            lp_fx_to_cpi_inf.query("horizon == 0").assign(check="alt_response_cpi_inf_h0")[
                ["check", "regime", "coefficient", "lower_90", "upper_90", "nobs"]
            ],
            lp_fx_to_inflation_truncated.query("horizon == 0").assign(check="truncated_sample_fx_to_inflation_h0")[
                ["check", "regime", "coefficient", "lower_90", "upper_90", "nobs"]
            ],
            gpr_alt_to_oil[gpr_alt_to_oil["term"] == "GPR_ME_alt"].assign(
                check="alt_gpr_to_oil"
            )[["check", "regime", "coefficient", "std_error", "p_value", "nobs"]].rename(
                columns={"std_error": "lower_90", "p_value": "upper_90"}
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    manifest = {
        "processed_panel": processed_path,
        "stationarity_table": _write_csv(stationarity, dirs["tables"] / "stationarity.csv"),
        "regime_summary_table": _write_csv(regime, dirs["tables"] / "regime_summary.csv"),
        "break_diagnostics_table": _write_csv(breaks, dirs["tables"] / "break_diagnostics.csv"),
        "event_snapshot_table": _write_csv(event_snapshots, dirs["tables"] / "event_snapshots.csv"),
        "gpr_to_oil_table": _write_csv(gpr_to_oil, dirs["tables"] / "gpr_to_oil_hac_ols.csv"),
        "gpr_alt_to_oil_table": _write_csv(gpr_alt_to_oil, dirs["tables"] / "gpr_alt_to_oil_hac_ols.csv"),
        "lp_oil_to_fx_table": _write_csv(lp_oil_to_fx, dirs["tables"] / "lp_oil_to_fx.csv"),
        "lp_oil_to_inflation_table": _write_csv(lp_oil_to_inflation, dirs["tables"] / "lp_oil_to_inflation.csv"),
        "lp_fx_to_inflation_table": _write_csv(lp_fx_to_inflation, dirs["tables"] / "lp_fx_to_inflation.csv"),
        "lp_fx_to_cpi_inf_table": _write_csv(lp_fx_to_cpi_inf, dirs["tables"] / "lp_fx_to_cpi_inf.csv"),
        "lp_fx_to_inflation_truncated_table": _write_csv(
            lp_fx_to_inflation_truncated, dirs["tables"] / "lp_fx_to_inflation_truncated.csv"
        ),
        "robustness_summary_table": _write_csv(robustness_summary, dirs["tables"] / "robustness_summary.csv"),
        "macro_context_figure": _plot_macro_context(panel.dropna(subset=["oil_shock", "fx_dep", "inflation_mom"]), dirs["figures"] / "macro_context.png"),
        "crisis_overlay_figure": _plot_crisis_overlay(panel, dirs["figures"] / "crisis_overlay.png"),
        "lp_oil_to_fx_figure": _plot_local_projection(lp_oil_to_fx, "Oil shocks to FX depreciation", dirs["figures"] / "lp_oil_to_fx.png"),
        "lp_oil_to_inflation_figure": _plot_local_projection(lp_oil_to_inflation, "Oil shocks to inflation", dirs["figures"] / "lp_oil_to_inflation.png"),
        "lp_fx_to_inflation_figure": _plot_local_projection(lp_fx_to_inflation, "FX depreciation to inflation", dirs["figures"] / "lp_fx_to_inflation.png"),
    }

    if not skip_tvp:
        tvp_irf_path, tvp_fit_path = _run_tvp_svar(
            project_root=project_root,
            processed_path=processed_path,
            models_dir=dirs["models"],
            draws=tvp_draws,
            burnin=tvp_burnin,
            thin=tvp_thin,
            horizons=tvp_horizons,
            dates=DEFAULT_TVP_DATES,
        )
        manifest["tvp_irf_table"] = tvp_irf_path
        manifest["tvp_fit_table"] = tvp_fit_path

    return manifest
