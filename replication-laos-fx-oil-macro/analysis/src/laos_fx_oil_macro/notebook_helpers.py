from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from .data import DISTRESS_BREAK, ProjectPaths
from .pipeline import DEFAULT_TVP_DATES, run_research_pipeline


@dataclass(frozen=True)
class NotebookProject:
    root: Path
    output_root: Path
    notebooks_dir: Path
    dataset_path: Path
    processed_panel: Path
    output_data_dir: Path
    tables_dir: Path
    figures_dir: Path
    models_dir: Path
    results_summary: Path


VARIABLE_DEFINITIONS = (
    {
        "variable": "GPR_ME",
        "definition": "Project-specific Middle East geopolitical-risk proxy used as the upstream external-risk signal.",
        "model_role": "Benchmark TVP-SV-VAR ordering and reduced-form oil specification.",
    },
    {
        "variable": "oil_shock",
        "definition": "Monthly oil-price shock, defined as 100 * Δlog(Oil).",
        "model_role": "External shock variable in local projections and TVP-SV-VAR.",
    },
    {
        "variable": "fx_dep",
        "definition": "Monthly exchange-rate depreciation, defined as 100 * Δlog(USDLAK).",
        "model_role": "Transmission channel from external shocks into Laos inflation.",
    },
    {
        "variable": "inflation_mom",
        "definition": "Monthly inflation, defined as 100 * Δlog(CPI).",
        "model_role": "Main dynamic inflation outcome in the benchmark models.",
    },
    {
        "variable": "CPI_Inf",
        "definition": "Year-over-year CPI inflation rate for Laos.",
        "model_role": "Crisis-context series and robustness-check outcome.",
    },
    {
        "variable": "FFR",
        "definition": "Effective federal funds rate.",
        "model_role": "External monetary control in reduced-form specifications.",
    },
    {
        "variable": "GSCPI",
        "definition": "New York Fed Global Supply Chain Pressure Index.",
        "model_role": "Global supply-chain control in reduced-form specifications.",
    },
    {
        "variable": "GPR_ME_alt",
        "definition": "Alternative regional GPR average excluding South Africa.",
        "model_role": "Robustness check for the GPR construction.",
    },
)


SERIES_LABELS = {
    "GPR_ME": "Middle East geopolitical risk",
    "oil_shock": "Oil shock, 100*Δlog(Oil)",
    "fx_dep": "FX depreciation, 100*Δlog(USDLAK)",
    "inflation_mom": "Monthly inflation, 100*Δlog(CPI)",
    "CPI_Inf": "CPI inflation, y/y",
}


def discover_project_root(start: str | Path | None = None) -> Path:
    base = Path.cwd() if start is None else Path(start)
    base = base.resolve()
    candidates = (base, *base.parents)
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "laos_fx_oil_macro").exists():
            return candidate
    raise FileNotFoundError(f"Could not find project root from {base}")


def build_notebook_project(
    project_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> NotebookProject:
    root = discover_project_root(project_root)
    output = (root / "output") if output_root is None else Path(output_root).resolve()
    paths = ProjectPaths(root)
    return NotebookProject(
        root=root,
        output_root=output,
        notebooks_dir=root / "notebooks",
        dataset_path=paths.panel_path,
        processed_panel=output / "data" / "laos_fx_oil_macro_research_panel.csv",
        output_data_dir=output / "data",
        tables_dir=output / "tables",
        figures_dir=output / "figures",
        models_dir=output / "models",
        results_summary=root / "paper" / "results_summary.md",
    )


def variable_definitions_table() -> pd.DataFrame:
    return pd.DataFrame(VARIABLE_DEFINITIONS)


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def path_status_table(project: NotebookProject) -> pd.DataFrame:
    rows = [
        ("project_root", project.root),
        ("notebooks_dir", project.notebooks_dir),
        ("dataset", project.dataset_path),
        ("output_root", project.output_root),
        ("processed_panel", project.processed_panel),
        ("tables_dir", project.tables_dir),
        ("figures_dir", project.figures_dir),
        ("models_dir", project.models_dir),
        ("results_summary", project.results_summary),
    ]
    return pd.DataFrame(
        [
            {
                "label": label,
                "path": _relative_path(path, project.root),
                "exists": path.exists(),
                "kind": "dir" if path.is_dir() else "file",
            }
            for label, path in rows
        ]
    )


def quick_links_table(project: NotebookProject) -> pd.DataFrame:
    rows = [
        ("results_summary", project.results_summary),
        ("dataset", project.dataset_path),
        ("output_root", project.output_root),
        ("tables_dir", project.tables_dir),
        ("figures_dir", project.figures_dir),
        ("models_dir", project.models_dir),
        ("notebooks_dir", project.notebooks_dir),
    ]
    return pd.DataFrame(
        [{"label": label, "path": _relative_path(path, project.root)} for label, path in rows]
    )


def output_manifest_table(project: NotebookProject) -> pd.DataFrame:
    rows: list[dict[str, str | int | float]] = []
    for category, directory in (
        ("data", project.output_data_dir),
        ("tables", project.tables_dir),
        ("figures", project.figures_dir),
        ("models", project.models_dir),
    ):
        if not directory.exists():
            continue
        for path in sorted(p for p in directory.iterdir() if p.is_file()):
            rows.append(
                {
                    "category": category,
                    "file": path.name,
                    "path": _relative_path(path, project.root),
                    "size_kb": round(path.stat().st_size / 1024, 1),
                }
            )
    return pd.DataFrame(rows)


def round_numeric_columns(frame: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    rounded = frame.copy()
    numeric_cols = rounded.select_dtypes(include=["number"]).columns
    rounded[numeric_cols] = rounded[numeric_cols].round(digits)
    return rounded


def compact_interval_pivot(
    frame: pd.DataFrame,
    *,
    index: str,
    columns: str,
    coefficient: str = "coefficient",
    lower: str = "lower_90",
    upper: str = "upper_90",
    digits: int = 3,
) -> pd.DataFrame:
    work = frame[[index, columns, coefficient, lower, upper]].copy()
    work["formatted"] = work.apply(
        lambda row: (
            f"{row[coefficient]:.{digits}f} "
            f"[{row[lower]:.{digits}f}, {row[upper]:.{digits}f}]"
        ),
        axis=1,
    )
    table = work.pivot(index=index, columns=columns, values="formatted")
    return table.sort_index()


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def run_pipeline_from_notebook(
    project: NotebookProject,
    *,
    tvp_draws: int = 1000,
    tvp_burnin: int = 300,
    tvp_thin: int = 2,
    tvp_horizons: int = 12,
    lp_horizons: int = 12,
    lp_lags: int = 2,
    skip_tvp: bool = False,
) -> dict[str, Path]:
    return run_research_pipeline(
        project_root=project.root,
        output_root=project.output_root,
        tvp_draws=tvp_draws,
        tvp_burnin=tvp_burnin,
        tvp_thin=tvp_thin,
        tvp_horizons=tvp_horizons,
        lp_horizons=lp_horizons,
        lp_lags=lp_lags,
        skip_tvp=skip_tvp,
    )


def plot_transformed_series(
    panel: pd.DataFrame,
    columns: Iterable[str] = ("GPR_ME", "oil_shock", "fx_dep", "inflation_mom", "CPI_Inf"),
    *,
    break_date: str | pd.Timestamp = DISTRESS_BREAK,
    event_dates: Iterable[str] = DEFAULT_TVP_DATES,
) -> plt.Figure:
    columns = list(columns)
    fig, axes = plt.subplots(len(columns), 1, figsize=(12, 2.5 * len(columns)), sharex=True)
    if len(columns) == 1:
        axes = [axes]
    break_ts = pd.Timestamp(break_date)
    event_dates = [pd.Timestamp(date) for date in event_dates]

    for idx, (ax, column) in enumerate(zip(axes, columns)):
        ax.plot(panel["Date"], panel[column], color="#1f4b99", linewidth=1.5)
        ax.axvline(break_ts, color="#b22222", linestyle="--", linewidth=1.2, label="Jan 2022 split")
        for event_date in event_dates:
            ax.axvline(event_date, color="#444444", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.set_title(SERIES_LABELS.get(column, column))
        ax.grid(alpha=0.25)
        if idx == 0:
            ax.legend(frameon=False, loc="upper left")

    fig.suptitle("Main transformed research series with regime split and event dates")
    fig.tight_layout()
    return fig


def plot_tvp_irf_scenarios(
    irf_summary: pd.DataFrame,
    *,
    impulse: str,
    response: str,
    dates: Iterable[str] = DEFAULT_TVP_DATES,
) -> plt.Figure:
    colors = {
        "2022-03-31": "#b22222",
        "2023-10-31": "#1f4b99",
        "2026-02-28": "#444444",
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for date in dates:
        subset = irf_summary[
            (irf_summary["date"] == date)
            & (irf_summary["impulse"] == impulse)
            & (irf_summary["response"] == response)
        ].sort_values("horizon")
        if subset.empty:
            continue
        color = colors.get(date, "#444444")
        ax.plot(subset["horizon"], subset["median"], color=color, linewidth=2, label=date)
        ax.fill_between(subset["horizon"], subset["lower_90"], subset["upper_90"], color=color, alpha=0.15)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"TVP-SV-VAR scenario IRFs: {impulse} -> {response}")
    ax.set_xlabel("Horizon (months)")
    ax.set_ylabel("Response")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig
