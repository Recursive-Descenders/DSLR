# `logreg_train` — behavior and data contract

## What it does

- Trains **four** binary logistic-regression models in a **one-vs-all** setup (Gryffindor, Hufflepuff, Ravenclaw, Slytherin). Each model minimizes **binary cross-entropy** with gradient-based updates.
- Fitting the weights and **estimating** `medians`, `mu`, and `sigma` use **only the training set**. those values are written to JSON. At predict time, rows are imputed and scaled with those saved values (see Preprocessing).
- The trainer **accepts a path to the training CSV** (positional) **and several flags** (optimizer, learning rate, epochs, batch size, loss plot, and more). For the **full list of arguments and their defaults**, run it with **`-h`** or **`--help`**. the sections below cover behavior and the data contract, not every flag.

## Input CSV (loader rules)

- Expects a header row, then one row per student. rows with **fewer than 19** columns or an **empty** Hogwarts House field (column index **1**, 0-based) are skipped. Malformed numeric cells in used columns are skipped.

## Features (must match `logreg_predict`)

- **Best Hand:** `Right` → `1.0`, `Left` → `0.0`, empty → NaN.
- **Course grades:** numeric grade columns are taken in the same order as in `load_xy` / `load_x_test` (all default Hogwarts course columns **except** the three left out on purpose, below). Empty cells → NaN; other values are cast to float. **Why three columns are dropped:** *Defense Against the Dark Arts* is very strongly related to *Astronomy* (it largely repeats the same information), so *Defense* is removed and *Astronomy* kept. *Potions* and *Care of Magical Creatures* sit on top of each other in the pair plot, and that pair also overlaps strongly with *Arithmancy*; we drop *Potions* and *Care* (two redundant dimensions) and keep *Arithmancy* among the other courses.
- Feature dimension is fixed by that column set; the bias term is added inside training (one extra weight per model).

## Preprocessing

- **Median values** are estimated **once from the training matrix** (per column: `nanmedian`. an all-NaN column’s median is stored as `0.0`). Those values are what get saved in `medians` in the model file.
- **Training** loads raw features with NaNs, **fills** them with those training medians, then **standardizes** that imputed matrix: `mu` and `sigma` are the mean and std of the **imputed** training columns, with `sigma` floored to at least `1e-12` before division.
- **Prediction** uses the same pipeline on new rows: **fill** NaNs with the **saved** `medians` (not recomputed from test), then standardize with the **saved** `mu` and `sigma`. So imputation runs in both train and predict. what is *not* done on the test set is re-fitting medians or scaling statistics.

## Artifacts

- **`model/model.json`**: `theta` (per-house list of floats: bias plus one weight per feature), `medians`, `mu`, `sigma` (all lists aligned with the feature order above). Paths are relative to the process **current working directory**. the `model` directory is created if needed.
- **`visualizations/training_loss.png`**: only when `--plot-loss` is set. the `visualizations` directory is created if needed.

## Inference

- Same feature columns and saved `medians`, `mu`, and `sigma` as in `load_x_test` in `src/logreg_predict.py` (the prediction entry point is documented in the main README).
