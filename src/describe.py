import argparse, math, pandas, sys


STAT_ORDER = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]


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
        return {stat: float("nan") for stat in STAT_ORDER}

    values.sort()
    total = sum(values)
    mean = total / count
    squared_diffs = sum((value - mean) ** 2 for value in values)
    std = math.sqrt(squared_diffs / (count - 1)) if count > 1 else float("nan")

    return {
        "count": float(count),
        "mean": mean,
        "std": std,
        "min": values[0],
        "25%": percentile(values, 0.25),
        "50%": percentile(values, 0.50),
        "75%": percentile(values, 0.75),
        "max": values[-1],
    }


def describe_dataframe(df: pandas.DataFrame) -> pandas.DataFrame:
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
    df = pandas.read_csv(args.csv).select_dtypes(include="number")
    print(df.columns)
    print(describe_dataframe(df))
    print(df.describe())

if __name__ == "__main__":
    main()
