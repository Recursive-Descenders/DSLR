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

## Algorithm

Implementation lives in `src/logreg_train.py` (NumPy only—no scikit-learn for fitting).

**Multiclass (one-vs-all):** There are four **independent** binary classifiers—one per house. For that house, $y^i \in \{0,1\}$ is the label of example $i$ (member vs not). All four models use the **same** preprocessed examples $x^i$; the parameter vectors $\theta$ are **not** shared.

**Binary hypothesis (subject-style):** The design vector $x$ includes a constant 1 for the intercept. With $g$ the sigmoid,

$$
h_\theta(x) = g(\theta^\top x), \qquad g(z) = \frac{1}{1 + e^{-z}}.
$$

In code, $z = \theta^\top x$ is **clipped** to $[-500, 500]$ before $g$ so the exponential stays numerically safe.

**Cost $J(\theta)$ (mean log loss over $m$ training examples, same form as the course notes):**

$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^i \log\bigl(h_\theta(x^i)\bigr) + (1 - y^i) \log\bigl(1 - h_\theta(x^i)\bigr) \right].
$$

**Partial derivatives of $J$ (one component $j$ per feature / bias):**

$$
\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \bigl(h_\theta(x^i) - y^i\bigr) \, x_j^i .
$$

**Vector / batch form (what the implementation uses):** Stacking rows $x^i$ into $X$, and writing $h$ and $y$ for the column vectors with entries $h_\theta(x^i)$ and $y^i$,

$$
\nabla_\theta J = \frac{1}{m} X^\top (h - y), \qquad \theta \leftarrow \theta - \alpha \nabla_\theta J .
$$

For a **minibatch** of size $|b|$, the same pattern applies: use $X_b$, the matching $h$ and $y$, and divide by $|b|$ instead of $m$. Log terms in the definition of $J$ are **clipped** in code to avoid $\log(0)$; the gradient still has the same $(h - y) \cdot x$ structure as above.

**Batching in this repo:**

- **`gd`:** one batch = all $m$ training examples (so $\nabla_\theta J$ uses the full $X$ and full $h - y$ each step within an epoch).
- **`mbgd`:** several minibatches per epoch; $|b| =$ `--batch-size` (default ~25% of $m$ if omitted).
- **`sgd`:** $|b| = 1$.

**Initialization:** $\theta$ is drawn from a small Gaussian (`randn` scaled by `0.01`).

**Tracked loss (`--plot-loss`):** After each epoch, **full-batch** mean binary cross-entropy is evaluated on the **entire** training set for that house (for plotting only; updates still follow the chosen batching rule above).

## Artifacts

- **`model/model.json`**: `theta` (per-house list of floats: bias plus one weight per feature), `medians`, `mu`, `sigma` (all lists aligned with the feature order above). Paths are relative to the process **current working directory**. the `model` directory is created if needed.
- **`visualizations/training_loss.png`**: only when `--plot-loss` is set. the `visualizations` directory is created if needed.

## Inference

- Same feature columns and saved `medians`, `mu`, and `sigma` as in `load_x_test` in `src/logreg_predict.py` (the prediction entry point is documented in the main README).
