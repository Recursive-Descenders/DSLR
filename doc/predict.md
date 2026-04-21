# `logreg_predict` — behavior and data contract

## What it does

- Loads **`model/model.json`** (one-vs-all weights `theta` plus preprocessing: `medians`, `mu`, `sigma`), reads a **test** CSV, builds the same feature matrix as training, applies **saved** imputation and standardization, computes a **sigmoid probability** per house, then picks the house with the **highest** score and writes **`houses.csv`** (one row per loaded test row, in loader order).
- The script takes **two optional positional arguments**: **test CSV path** (default `dataset_test.csv`), then **model path** (default `model/model.json`). Invocation is documented in the main README.

## Input CSV (loader rules)

- Expects a header row, then one row per student; rows with **fewer than 19** columns are skipped. The **Index** field (first column) must be present for each kept row. The Hogwarts House column may be **empty** on test data. Malformed numeric cells in used feature columns are skipped.
- If the file cannot be read, or **no** valid rows remain, the program errors and **exits with code 1** (see stderr).

## Features

- **Same construction and order as training** — see `load_x_test` in `src/logreg_predict.py` and the feature / drop rationale in [doc/train.md](train.md) (`load_xy` and `load_x_test` must stay aligned).

## Preprocessing (inference)

- For each test row, missing values are filled with the model’s **saved** `medians` (not recomputed from the test set), then each row is standardized with the **saved** `mu` and `sigma` from training.

## Decoding

- For each example, four probabilities are produced (one binary logistic model per house). The predicted house is the **argmax** over those four scores (fixed order: Gryffindor, Hufflepuff, Ravenclaw, Slytherin).

## Artifacts

- **`houses.csv`** in the process **current working directory**, with header `Index,Hogwarts House` and one data row per successfully loaded test row.

## Model file requirements

- **`theta`:** a non-empty weight vector for each house, length **`1 + d`** for **`d` features** (bias plus one weight per feature; same layout as in training’s `logistic_regression` / `predict_probability`).
- **`medians`:** required; if missing, the program asks you to retrain. **Feature count** in the data must **equal** `len(medians)` or the run aborts.
- If the model file is missing, unreadable, or **theta** is invalid / incomplete, the program **exits with code 1**.

## Training

- You need a model produced by `logreg_train` with matching feature layout; see [doc/train.md](train.md) for the training side of the same pipeline.
