import argparse, math, pandas, sys


STAT_ORDER = ["count", "mean", "std", "min", "25%", "50%", "75%", "max", "variance", "range"]


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


def describe_column(column: pandas.Series) -> dict[str, float]:
    values = [float(value) for value in column if not pandas.isna(value)]
    count = len(values)
    if count == 0:
        return {
            "count": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "25%": float("nan"),
            "50%": float("nan"),
            "75%": float("nan"),
            "max": float("nan"),
            "variance": float("nan"),
            "range": float("nan"),
        }

    values.sort()
    total = sum(values)
    mean = total / count
    squared_diffs = sum((value - mean) ** 2 for value in values)
    var = squared_diffs / (count - 1) if count > 1 else float("nan")    # bonus: sample variance
    std = math.sqrt(var) if count > 1 else float("nan")

    # bonus:

    return {
        "count": float(count),
        "mean": mean,
        "std": std,
        "min": values[0],
        "25%": percentile(values, 0.25),
        "50%": percentile(values, 0.50),
        "75%": percentile(values, 0.75),
        "max": values[-1],
        "variance": var,
        "range": values[-1] - values[0],
    }


def describe_dataframe(df: pandas.DataFrame) -> pandas.DataFrame:
    df = df.drop(columns=["Index"], errors="ignore")
    description = {
        column_name: describe_column(df[column_name])
        for column_name in df.columns
    }
    return pandas.DataFrame(description).reindex(STAT_ORDER)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--csv",
        type=str,
        default="dataset_train.csv",
        help="Path to CSV"
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
    print(describe_dataframe(df))


if __name__ == "__main__":
    main()
