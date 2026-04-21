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

COMPACT_MAX_COLUMNS = 7
COMPACT_WIDTH = 120


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


def numeric_values(column: pandas.Series) -> list[float]:
    return sorted(float(value) for value in column if not pandas.isna(value))


def build_base_stats(values: list[float]) -> tuple[dict[str, float | int], float]:
    count = len(values)
    if count == 0:
        return (
            {
                "count": 0,
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "25%": float("nan"),
                "50%": float("nan"),
                "75%": float("nan"),
                "max": float("nan"),
            },
            float("nan"),
        )

    total = sum(values)
    mean = total / count
    squared_diffs = sum((value - mean) ** 2 for value in values)
    variance = squared_diffs / (count - 1) if count > 1 else float("nan")
    std = math.sqrt(variance) if count > 1 else float("nan")

    return (
        {
            "count": count,
            "mean": mean,
            "std": std,
            "min": values[0],
            "25%": percentile(values, 0.25),
            "50%": percentile(values, 0.50),
            "75%": percentile(values, 0.75),
            "max": values[-1],
        },
        variance,
    )


def build_bonus_stats(
    values: list[float],
    total_count: int,
    q1: float,
    q3: float,
    variance: float,
) -> dict[str, float | int]:
    count = len(values)
    missing_count = total_count - count
    missing_pct = (missing_count / total_count * 100.0) if total_count > 0 else float("nan")

    if count == 0:
        return {
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "variance": float("nan"),
            "range": float("nan"),
            "iqr": float("nan"),
            "outliers_iqr_count": 0,
        }

    value_range = values[-1] - values[0]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers_iqr_count = sum(1 for value in values if value < lower_bound or value > upper_bound)

    return {
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "variance": variance,
        "range": value_range,
        "iqr": iqr,
        "outliers_iqr_count": outliers_iqr_count,
    }


def describe_column(column: pandas.Series, include_bonus: bool) -> dict[str, float | int]:
    total_count = len(column)
    values = numeric_values(column)
    base_stats, variance = build_base_stats(values)
    if not include_bonus:
        return base_stats

    bonus_stats = build_bonus_stats(
        values,
        total_count,
        q1=float(base_stats["25%"]),
        q3=float(base_stats["75%"]),
        variance=variance,
    )
    return {**base_stats, **bonus_stats}


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
        COMPACT_MAX_COLUMNS,
        "display.width",
        COMPACT_WIDTH,
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
