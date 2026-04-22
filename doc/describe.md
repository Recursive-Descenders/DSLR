# `describe` — behavior and data contract

Summarizes **numeric** columns of a CSV in a table: **rows** are statistic names, **columns** are feature names (same layout as a typical `DataFrame.describe`-style view). 

## CLI

```text
describe [--csv PATH] [--full] [--bonus]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-c` / `--csv` | `dataset_train.csv` | Input CSV path |
| `--full` | off | Print without column truncation (all numeric columns) |
| `--bonus` | off | Append extra statistic rows (see below) |

Run `describe -h` for built-in help.

## Loading and columns

- The file is read with **`pandas.read_csv`**, then **only numeric dtypes** are kept (`select_dtypes(include="number")`). Purely non-numeric fields (e.g. names, categorical text) never appear in the table.
- If a column named **`Index`** is still present (often integer), it is **dropped** before statistics are computed so the index does not show up as a feature.
- **Errors:** missing file, parse errors, or OS read failures exit with code **1** and a message on stderr.

## Missing values (`NaN`)

- Empty cells and other missing entries become **`NaN`** in pandas.
- For each column, only **non-`NaN`** values enter the ordered sample used for min, quartiles, max, mean, etc.
- **`count`** is the number of **non-`NaN`** values in that column (missing cells are **not** included).
- With **`--bonus`**, `missing_count` and `missing_pct` compare that to the column length (total rows in the frame after numeric selection).

## Statistic definitions

Percentiles (**`25%`**, **`50%`**, **`75%`**) use **linear interpolation** on the **sorted** non-`NaN` values: index position \((n - 1) \times q\) between adjacent sorted points (same idea as `describe.py`’s `percentile` helper).

**Base rows** (always printed, in this order):

| Row | Definition |
|-----|------------|
| `count` | Number of non-`NaN` values |
| `mean` | Arithmetic mean of non-`NaN` values |
| `std` | Sample standard deviation (\(n-1\) denominator); `nan` if fewer than 2 points |
| `min` | Smallest non-`NaN` value |
| `25%` | First quartile (interpolated) |
| `50%` | Median (interpolated) |
| `75%` | Third quartile (interpolated) |
| `max` | Largest non-`NaN` value |

**`--bonus` rows** (appended in this order):

| Row | Definition |
|-----|------------|
| `missing_count` | `len(column) - count` (cells that were `NaN`) |
| `missing_pct` | `100 * missing_count / len(column)`; `nan` if the frame has zero rows |
| `variance` | Sample variance (\(n-1\))); `nan` if fewer than 2 points |
| `range` | `max - min` on sorted non-`NaN` values; `nan` if no data |
| `iqr` | `75%` − `25%` |
| `outliers_iqr_count` | Count of non-`NaN` values **strictly outside** Tukey fences: below \(Q_1 - 1.5 \times \mathrm{IQR}\) or above \(Q_3 + 1.5 \times \mathrm{IQR}\) (using the same \(Q_1\), \(Q_3\) as above); `0` if there are no values |

If a column has **no** non-`NaN` values, base stats are mostly `nan` / `count` 0; bonus rows still report `missing_count` / `missing_pct` where applicable.

## Display

- Without **`--full`**, printing uses a **compact** layout (narrower width and a limited number of columns) so wide tables fit a terminal; with **`--full`**, pandas shows **all** columns.

