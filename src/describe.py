import argparse, sys

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
    print(f"CSV path: {args.csv}")

if __name__ == "__main__":
    main()
