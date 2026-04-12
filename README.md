# DSLR — Data Science × Logistic Regression

Hogwarts Sorting Hat replacement: explore the student dataset, then train a one-vs-all logistic regression to predict houses.

Full brief (requirements, forbidden helpers, output formats): **`en.subject.txt`** in this repo.

---

## Who does what

| Part | Topic | Status |
|------|--------|--------|
| **V.1** | Data analysis — `describe.[ext]` prints stats for all **numeric** columns (count, mean, std, min, quartiles, max) **without** using built-in `mean`/`std`/percentile/`describe`-style shortcuts | **Your job** |
| **V.2** | Visualization — `histogram`, `scatter_plot`, `pair_plot` scripts answering the subject’s questions | **Your job** |
| **V.3** | Logistic regression — `logreg_train` + `logreg_predict`, gradient descent, weights file + `houses.csv` predictions | **In progress here** (train/predict logic exists under `src/`, naming and polish may still need to match the subject) |

---

## What’s already here (V.3 WIP)

- `src/train.py` — trains on `dataset_train.csv`, writes weights to `model/model.json`. Still needs a full **one-vs-all** setup for all four houses and CLI args like the subject asks.
- `src/predict.py` — loads the model and scores rows; wire it to **`houses.csv`** and multi-class picks when training is complete.
- `Makefile` — `make setup` (venv + deps), `make train`, `make predict`. There is an `evaluate` target pointing at `src/evaluate.py`, but that file is not in the repo yet.

Subject expects executable names like **`logreg_train`** / **`logreg_predict`** and a prediction file **`houses.csv`** with header `Index,Hogwarts House`. Align naming and outputs with `en.subject.txt` before defense.

---

## What we expect from you (V.1 & V.2)

1. **`describe`** — CLI takes a CSV path; output format should match the subject example for numeric features only; implement stats yourself (no cheating shortcuts per subject).
2. **`histogram`** — which course has the most homogeneous scores across the four houses?
3. **`scatter_plot`** — which two features look most similar?
4. **`pair_plot`** — feature matrix / scatter matrix; use it to justify which features you’ll feed into logistic regression.

Use Python or whatever the team agrees on; keep plotting/stat code honest per Chapter IV.

---

## Quick start

```bash
make setup          # creates .venv and installs requirements
make train          # uses dataset_train.csv (path is hardcoded in train.py for now)
make predict        # runs predict.py; for test data: python src/predict.py dataset_test.csv
```

Data: `dataset_train.csv`, `dataset_test.csv` in the project root.

---

## Defense notes (everyone)

- Peer eval checks predictions on `dataset_test.csv` with **sklearn accuracy ≥ 98%** on houses.
- Bonus only counts if the mandatory part is complete and solid — see subject.

If something’s unclear, read **`en.subject.txt`** first; it’s the source of truth.
