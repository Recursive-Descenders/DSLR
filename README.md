# DSLR — Data Science × Logistic Regression

A small Hogwarts-themed data-science exercise: explore the student dataset, visualize it, then train a **one-vs-all** logistic regression (implemented with NumPy—no scikit-learn for the core model) to predict houses.

## Warning for 42 Students

This repository is intended as a reference and educational tool. 42 students are strongly advised not to copy this code without fully understanding its functionality. Plagiarism in any form is against 42's principles and could lead to serious academic consequences. Use this repository responsibly to learn and better understand how to implement similar functionalities on your own.

## Credits

- **Data analysis and exploration** — [ChimPansky](https://github.com/ChimPansky)
- **Training, inference, and evaluation** — [Mohamad Zolfaghari Pour](https://github.com/zolfagharipour)

## Setup

Requires **Python 3.11+**.

Install `uv` (pick one): see the [uv install guide](https://docs.astral.sh/uv/getting-started/installation/), or e.g. `pipx install uv` if you already use pipx.

From the repository root:

```bash
uv sync
```

Then either prefix commands with `uv run` (uses the project env automatically):

```bash
uv run describe
uv run train
```

—or activate the env and run the scripts directly: `source .venv/bin/activate` on Unix, then `describe`, `train`, etc.

That exposes console scripts: `describe`, `histogram`, `scatter_plot`, `pair_plot`, `train`, `predict`, and `confusion`. Use `-h` / `--help` on each where available.

**Layout:** training writes `model/model.json`; optional plots go under `visualizations/`. Sample CSVs are `dataset_train.csv`, `dataset_test.csv`, and `dataset_truth.csv` (for evaluation).

**Typical flow**

- **Summarize numeric columns:** `describe [--csv PATH] [--full] [--bonus]`
- **Plots (training CSV):** `histogram` / `scatter_plot` / `pair_plot` — each accepts `--csv` (default `dataset_train.csv`); see `--help` for details.
- **Train:** `train [dataset_train.csv]` — flags include `--optimizer {gd,mbgd,sgd}`, `--lr`, `--epochs`, `--batch-size` (MBGD only), `--plot-loss` (writes `visualizations/training_loss.png`).
- **Predict:** `predict [dataset_test.csv] [model/model.json]` — writes `houses.csv` in the current working directory.
- **Confusion matrix vs ground truth:** `confusion dataset_truth.csv [houses.csv]` — saves `visualizations/confusion_matrix.png`.


## Documentation

- [doc/describe.md](doc/describe.md) — how `describe` treats non-numeric columns, missing values, and NaN in aggregations.
- [doc/train.md](doc/train.md) — training pipeline, **one-vs-all logistic regression algorithm** (loss, gradients, optimizers), features, preprocessing, artifacts (`model/model.json`, optional loss plot).
- [doc/predict.md](doc/predict.md) — loading the model, test CSV rules, and `houses.csv` output.
