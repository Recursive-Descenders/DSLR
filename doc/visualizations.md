# Visualizations — exploration plots

Three entry points build **matplotlib** figures from a **training-style** CSV (house labels + numeric course columns).

|Console command | Output file(s) |
|-----------------|----------------|
| `histogram` | `visualizations/histograms_by_house.png` |
| `scatter_plot` | `visualizations/scatter_pairs_page_01.png`, `…_02.png`, … |
| `pair_plot` | `visualizations/pair_plot_matrix.png` |

Common CLI:

```text
histogram | scatter_plot | pair_plot   [--csv PATH]
```

Default **`--csv`** is `dataset_train.csv`. Use `-h` on each command for help text.

## Shared data contract

All three scripts:

1. **`pandas.read_csv`** the path (exit **1** on missing file, parse error, or OS error).
2. Require a **`Hogwarts House`** column.
3. Derive **houses** as `sorted` unique non-null house names — there must be **1–4** houses.
4. **Subject columns** = numeric dtypes from the frame, **excluding** a column named **`Index`** if present. There must be **1–13** such columns (enforced for the Hogwarts dataset shape).

**Colors:** `HOUSE_COLORS` in `utils.py` maps the **i-th** sorted house to the **i-th** color (Gryffindor → red, Hufflepuff → gold, Ravenclaw → blue, Slytherin → green for the default four).

**Missing values:** Per plot, rows with **`NaN`** in the relevant subject(s) are dropped (`.dropna()`); they are not imputed.

## `histogram`

- One **grid of subplots**: **4 columns**, as many rows as needed; **one subplot per numeric subject**.
- In each subplot, **overlaid histograms** (20 bins, alpha 0.4) — one series per house for that subject.
- Unused subplot slots are hidden.
- Figure title: “Hogwarts Subjects by House”; legend from the first used axis.
- Saved at **150 DPI** via `save_plot` (default in `utils`).

## `scatter_plot`

- Needs **at least 2** subject columns.
- Builds all **unordered pairs** of subjects (`itertools.combinations`), in the order subjects appear in the frame.
- **Pagination:** up to **9** pair plots per figure (3×3 grid). Extra pairs go to **`scatter_pairs_page_02.png`**, etc. (two-digit page index in the filename).
- Each mini-plot: **scatter** per house (alpha 0.55, marker size 10), axes labeled with subject names.
- Empty subplot cells on the last page are turned off.
- Saved at **150 DPI** per page. Stdout summarizes how many pairs and pages were generated.

## `pair_plot`

- Needs **at least 2** subject columns.
- Builds an **n×n matrix** of subplots (`n` = number of subjects): **diagonal** = overlaid **histograms** (18 bins, alpha 0.5) per house for that subject; **off-diagonal** = **scatter** of column vs row subject (alpha 0.55, marker size 8), same color-by-house rule.
- Axis labels: x-label only on bottom row (rotated 45°); y-label only on left column; small ticks for readability.
- Single file **`pair_plot_matrix.png`** at **130 DPI** (pair plot passes `dpi=130` to `save_plot`).
- Prints a short hint about using the matrix to spot separation and redundant pairs.

## Other figures in `visualizations/`

These are produced by other commands but land in the same folder:

| Artifact | Produced by | Doc |
|----------|-------------|-----|
| `training_loss.png` | `train --plot-loss` | [doc/train.md](train.md) (artifacts) |
| `confusion_matrix.png` | `confusion …` | main [README](../README.md) (Confusion matrix) |

Paths are relative to the process **current working directory**.
