import argparse
import math
import sys

import pandas


BASE_STAT_ORDER = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
BONUS_STAT_ORDER = [
    "missing_count",
    "missing_pct",
    "variance",
    "range",
    "iqr",
    "outliers_iqr_count",
]


def percentile(sorted_values: list[float], quantile: float) -> float:
    count = len(sorted_values)
    if count == 0:
        return float("nan")
    if count == 1:
        return sorted_values[0]

    position = (count - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, count - 1)
    fraction = position - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * fraction


def describe_column(column: pandas.Series, include_bonus: bool) -> dict[str, float]:
    total_count = len(column)
    values = [float(value) for value in column if not pandas.isna(value)]
    count = len(values)
    missing_count = total_count - count
    missing_pct = (missing_count / total_count * 100.0) if total_count > 0 else float("nan")

    if count == 0:
        result = {
            "count": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "25%": float("nan"),
            "50%": float("nan"),
            "75%": float("nan"),
            "max": float("nan"),
        }
        if include_bonus:
            result.update(
                {
                    "missing_count": float(missing_count),
                    "missing_pct": missing_pct,
                    "variance": float("nan"),
                    "range": float("nan"),
                    "iqr": float("nan"),
                    "outliers_iqr_count": 0.0,
                }
            )
        return result

    values.sort()
    total = sum(values)
    mean = total / count
    squared_diffs = sum((value - mean) ** 2 for value in values)
    var = squared_diffs / (count - 1) if count > 1 else float("nan")
    std = math.sqrt(var) if count > 1 else float("nan")

    q1 = percentile(values, 0.25)
    q2 = percentile(values, 0.50)
    q3 = percentile(values, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers_iqr_count = float(
        sum(1 for value in values if value < lower_bound or value > upper_bound)
    )

    result = {
        "count": float(count),
        "mean": mean,
        "std": std,
        "min": values[0],
        "25%": q1,
        "50%": q2,
        "75%": q3,
        "max": values[-1],
    }
    if include_bonus:
        result.update(
            {
                "missing_count": float(missing_count),
                "missing_pct": missing_pct,
                "variance": var,
                "range": values[-1] - values[0],
                "iqr": iqr,
                "outliers_iqr_count": outliers_iqr_count,
            }
        )
    return result


def describe_dataframe(df: pandas.DataFrame, include_bonus: bool) -> pandas.DataFrame:
    stat_order = BASE_STAT_ORDER + BONUS_STAT_ORDER if include_bonus else BASE_STAT_ORDER
    df = df.drop(columns=["Index"], errors="ignore")
    description = {
        column_name: describe_column(df[column_name], include_bonus)
        for column_name in df.columns
    }
    return pandas.DataFrame(description).reindex(stat_order)


def print_description_table(table: pandas.DataFrame, full: bool) -> None:
    if full:
        with pandas.option_context(
            "display.max_columns",
            None,
            "display.width",
            0,
        ):
            print(table)
        return

    with pandas.option_context(
        "display.max_columns",
        8,
        "display.width",
        120,
    ):
        print(table)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--csv",
        type=str,
        default="dataset_train.csv",
        help="Path to CSV",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show all subject columns without truncation.",
    )
    parser.add_argument(
        "--bonus",
        action="store_true",
        help="Include bonus statistics (missing, variance, range, iqr, outlier count).",
    )
    args = parser.parse_args(sys.argv[1:])

    # We only care about numeric columns, no name, birthday, etc...
    try:
        df = pandas.read_csv(args.csv).select_dtypes(include="number")
    except FileNotFoundError:
        parser.exit(1, f"error: CSV file not found: {args.csv}\n")
    except pandas.errors.ParserError as error:
        parser.exit(1, f"error: failed to parse CSV '{args.csv}': {error}\n")
    except OSError as error:
        parser.exit(1, f"error: failed to read CSV '{args.csv}': {error}\n")
    description = describe_dataframe(df, include_bonus=args.bonus)
    print_description_table(description, full=args.full)


if __name__ == "__main__":
    main()
