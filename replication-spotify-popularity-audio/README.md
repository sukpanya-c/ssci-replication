# Spotify Popularity Audio Replication

This folder is the public replication package for the current analysis pipeline.

## Contents

- `dataset/`: placeholder location for restricted local data files
- `analysis/src/`: reusable Python analysis modules
- `analysis/notebooks/`: notebook entry points for replication
- `analysis/output/`: generated data, tables, and figures
- `analysis/tests/`: lightweight regression tests for the pipeline

## Data Availability

- The full raw dataset is not included in this public repository.
- The repository includes:
  - processed analysis-ready files under `analysis/output/data/`
- The notebooks and main analysis workflow run from the processed analysis-ready files, so the current results can still be reproduced without the raw CSV.
- The raw data import and sample-construction layer can only be rerun if an authorized user places the raw file at:
- If access to the underlying datasets is needed, please contact `sukpanya_c@sjtu.edu.cn`.

```text
dataset/spotify_popularity_audio_dataset.csv
```

## Recommended Workflow

1. Create a Python environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Run the main notebook:

- `analysis/notebooks/01_main_analysis.ipynb`

4. Run the appendix notebook:

- `analysis/notebooks/02_appendix_robustness.ipynb`

The notebooks are thin wrappers around the Python workflow module in `analysis/src/workflows.py`, so rerunning the notebooks regenerates the analysis outputs from the module code.

## Optional Command-Line Execution

You can also execute the notebooks from the replication folder root:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace analysis/notebooks/01_main_analysis.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace analysis/notebooks/02_appendix_robustness.ipynb
```

## Tests

Run:

```bash
pytest analysis/tests -q
```
