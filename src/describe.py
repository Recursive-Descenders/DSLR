import argparse, pandas, sys

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
    print(df)
    print(df.describe())

if __name__ == "__main__":
    main()
