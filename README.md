# DSLR — Data Science × Logistic Regression

A small Hogwarts-themed data-science exercise: explore the student dataset, visualize it, then train a **one-vs-all** logistic regression (implemented with NumPy—no scikit-learn for the core model) to predict houses.

## Warning for 42 Students

This repository is intended as a reference and educational tool. 42 students are strongly advised not to copy this code without fully understanding its functionality. Plagiarism in any form is against 42's principles and could lead to serious academic consequences. Use this repository responsibly to learn and better understand how to implement similar functionalities on your own.

## Credits

- **Data analysis and exploration** — [ChimPansky](https://github.com/ChimPansky)
- **Training, inference, and evaluation** — [Mohamad Zolfaghari Pour](https://github.com/zolfagharipour)
## Setup

Requires **Python 3.11+**.

Install **`uv`** (see the [uv install guide](https://docs.astral.sh/uv/getting-started/installation/), or e.g. `pipx install uv`). From the repository root:

```bash
uv sync
```

Run tools with `uv run <command>` or activate `.venv` (`source .venv/bin/activate` on Unix) and use the same names.

**Console scripts:** `describe`, `histogram`, `scatter_plot`, `pair_plot`, `train`, `predict`, `confusion`. Use `-h` / `--help` where supported.

**Outputs:** `model/model.json` after training; figures under `visualizations/`. Sample data: `dataset_train.csv`, `dataset_test.csv`, `dataset_truth.csv` (labels for evaluation).

## Typical flow

| Step | Command (after `uv sync`) | Details |
|------|---------------------------|---------|
| Summarize | `uv run describe [--csv PATH] [--full] [--bonus]` | [doc/describe.md](doc/describe.md) — stats, NaN rules, `--bonus` rows |
| Plot | `uv run histogram` / `scatter_plot` / `pair_plot` each `[--csv PATH]` | [doc/visualizations.md](doc/visualizations.md) — shared CSV contract, filenames |
| Train | `uv run train [PATH]` — `--optimizer {gd,mbgd,sgd}`, `--lr`, `--epochs`, `--batch-size` (MBGD only), `--plot-loss` | [doc/train.md](doc/train.md) — features, preprocessing, **algorithm**, artifacts |
| Predict | `uv run predict [dataset_test.csv] [model/model.json]` | [doc/predict.md](doc/predict.md) — writes `houses.csv` |
| Evaluate | See [Confusion matrix](#confusion-matrix) | Compare predictions to a labeled CSV |

Default CSV for exploration commands is `dataset_train.csv` where not shown.

## Confusion matrix

With `houses.csv` from `predict` and a ground-truth file using the same `Index,Hogwarts House` columns (e.g. `dataset_truth.csv`):

```bash
uv run confusion dataset_truth.csv houses.csv
```

Writes `visualizations/confusion_matrix.png` (true vs predicted counts, fixed house order).

## Documentation

| Doc | Contents |
|-----|----------|
| [doc/describe.md](doc/describe.md) | CLI, numeric-only columns, statistic definitions (base + `--bonus`), NaN handling, display modes |
| [doc/visualizations.md](doc/visualizations.md) | `histogram`, `scatter_plot`, `pair_plot`; shared loader rules; DPI and output names; pointers to loss and confusion figures |
| [doc/train.md](doc/train.md) | Loader and features (aligned with predict), preprocessing, **one-vs-all logistic regression** (loss, gradients, optimizers), `model.json`, `--plot-loss` |
| [doc/predict.md](doc/predict.md) | Model file requirements, test CSV rules, argmax decoding, `houses.csv` contract |
