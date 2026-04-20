from __future__ import annotations

import argparse
from pathlib import Path

from laos_fx_oil_macro.pipeline import run_research_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Laos IREF research pipeline.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--tvp-draws", type=int, default=1500)
    parser.add_argument("--tvp-burnin", type=int, default=500)
    parser.add_argument("--tvp-thin", type=int, default=5)
    parser.add_argument("--tvp-horizons", type=int, default=12)
    parser.add_argument("--lp-horizons", type=int, default=12)
    parser.add_argument("--lp-lags", type=int, default=2)
    parser.add_argument("--skip-tvp", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = run_research_pipeline(
        project_root=args.project_root,
        output_root=args.output_root,
        tvp_draws=args.tvp_draws,
        tvp_burnin=args.tvp_burnin,
        tvp_thin=args.tvp_thin,
        tvp_horizons=args.tvp_horizons,
        lp_horizons=args.lp_horizons,
        lp_lags=args.lp_lags,
        skip_tvp=args.skip_tvp,
    )
    for key, value in manifest.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
