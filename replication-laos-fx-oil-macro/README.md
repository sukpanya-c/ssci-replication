# Laos Oil, FX, And Inflation Replication

This repository is the public replication package for the Laos oil, FX, and inflation analysis.
The executable public replication sample in this folder starts in `2006-01`.

## Contents

- `dataset/`: real shared data files used by the replication workflow
- `analysis/src/`: reusable Python analysis modules
- `analysis/scripts/`: command-line runners for the pipeline, paper bundle, and AEL-specific figures
- `analysis/notebooks/`: notebook entry points for project inspection and reruns
- `analysis/output/`: generated data, tables, figures, and model summaries
- `analysis/paper/`: generated manuscript-support summary
- `analysis/tests/`: lightweight regression tests for the replication workflow

## Data Availability

- This public bundle includes the real merged monthly panel used by the analysis:
  - `dataset/laos_fx_oil_macro_monthly_panel_2006-01-31_2026-02-28.csv`
- It also includes the auxiliary Middle East component file required for the alternative GPR robustness specification:
  - `dataset/GPR_ME_middle_east_components_month_end_2006-01-31_2026-02-28.csv`
- No separate data dictionary is included in this public bundle. The main variables used in the empirical workflow are defined below.
- The public replication sample in this folder runs from `2006-01-31` to `2026-02-28`.

## Software Requirements

- Python `>=3.13`
- R with `Rscript` available on the command line
- R package: `bvarsv`

The replication workflow is Python-orchestrated, but the TVP-SV-VAR estimation step is executed through `analysis/scripts/run_tvp_svar.R`. Full end-to-end regeneration therefore requires both the pinned Python environment below and the R dependency above.

## Variables Used

- `Date`: month-end observation date.
- `Oil`: oil price level used to construct the monthly oil shock.
- `USDLAK`: kip per U.S. dollar exchange rate level used to construct monthly depreciation.
- `CPI`: consumer price index level used to construct monthly inflation.
- `CPI_Inf`: year-over-year CPI inflation rate.
- `GPR_ME`: project-specific Middle East geopolitical-risk proxy.
- `FFR`: effective federal funds rate.
- `GSCPI`: New York Fed Global Supply Chain Pressure Index.
- `GPR_ME_alt`: alternative regional GPR average built from the shared component file.
- `oil_shock`: `100 * Δlog(Oil)`.
- `fx_dep`: `100 * Δlog(USDLAK)`.
- `inflation_mom`: `100 * Δlog(CPI)`.

## How To Run

1. Create and activate a Python environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the pinned Python dependency set:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Install the required R package for the TVP-SV-VAR step:

```bash
Rscript analysis/scripts/install_r_dependencies.R
```

4. Launch Jupyter from the replication root:

```bash
jupyter lab
```

5. Open the main notebook first:

- `analysis/notebooks/00_project_hub.ipynb`

6. Then use the remaining notebooks as needed:

- `analysis/notebooks/01_data_overview_and_transformations.ipynb`
- `analysis/notebooks/02_baseline_results.ipynb`
- `analysis/notebooks/03_tvp_svar_and_robustness.ipynb`

The notebooks remain thin wrappers around the Python modules in `analysis/src/laos_fx_oil_macro`, so rerunning the notebooks regenerates outputs from the module code rather than duplicating the empirical logic inside notebook cells.

## Command-Line Execution

Run these commands from the replication root after activating the environment above.

Run the main pipeline:

```bash
PYTHONPATH=analysis/src python analysis/scripts/run_analysis.py \
  --project-root analysis \
  --output-root analysis/output \
  --tvp-draws 1000 \
  --tvp-burnin 300 \
  --tvp-thin 2 \
  --tvp-horizons 12 \
  --lp-horizons 12 \
  --lp-lags 2
```

For the full paper replication, keep the TVP-SV-VAR step enabled. If you only want the Python-only reduced-form outputs, add `--skip-tvp`, but that is not the complete replication path.

Build the paper-ready tables, figures, and manuscript-support summary:

```bash
PYTHONPATH=analysis/src python analysis/scripts/build_submission_bundle.py \
  --project-root analysis \
  --output-root analysis/output \
  --paper-dir analysis/paper
```

Regenerate the AEL-specific figure and robustness extras used in the manuscript:

```bash
PYTHONPATH=analysis/src python analysis/scripts/make_ael_break_figure.py --project-root analysis
PYTHONPATH=analysis/src python analysis/scripts/make_ael_additional_robustness.py
```

You can also execute notebooks non-interactively:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace analysis/notebooks/00_project_hub.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace analysis/notebooks/02_baseline_results.ipynb
```

## Tests

Run the replication test suite:

```bash
PYTHONPATH=analysis/src pytest analysis/tests -q
```

## Notes

- The strongest supported result remains the post-2022 amplification of `fx_dep -> inflation_mom`.
- The direct reduced-form monthly `GPR_ME -> oil_shock` relationship is weak in this sample and should not be treated as the main contribution.
