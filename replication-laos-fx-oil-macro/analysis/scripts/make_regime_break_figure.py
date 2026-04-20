#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create the regime-break figure for Laos.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output" / "paper_figures" / "figure_regime_break.png",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root / "src"))

    from laos_fx_oil_macro.data import DISTRESS_BREAK, ProjectPaths, load_panel, transform_panel

    paths = ProjectPaths(project_root)
    panel = transform_panel(load_panel(paths.panel_path)).dropna(subset=["fx_dep", "inflation_mom"]).copy()

    pre = panel[panel["Date"] < DISTRESS_BREAK]
    post = panel[panel["Date"] >= DISTRESS_BREAK]

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.8), sharex=True)
    series_specs = [
        ("fx_dep", "Kip depreciation (%)", "#1f4b99"),
        ("inflation_mom", "CPI inflation (%)", "#b22222"),
    ]

    for ax, (column, ylabel, color) in zip(axes, series_specs):
        ax.plot(panel["Date"], panel[column], color=color, linewidth=1.3)
        ax.axvline(DISTRESS_BREAK, color="#444444", linestyle="--", linewidth=1.2)
        ax.axhline(pre[column].mean(), color="#666666", linestyle=":", linewidth=1.1)
        ax.axhline(post[column].mean(), color=color, linestyle="--", linewidth=1.1, alpha=0.9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=9)

    axes[0].text(
        DISTRESS_BREAK,
        axes[0].get_ylim()[1] * 0.93,
        "Jan 2022 break",
        ha="left",
        va="top",
        fontsize=9,
        color="#444444",
    )
    axes[1].set_xlabel("Date", fontsize=10)
    axes[1].xaxis.set_major_locator(mdates.YearLocator(4))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
