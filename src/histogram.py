import argparse
import math
import sys

import matplotlib.pyplot as plt

from utils import (
    _C,
    _D,
    _G,
    _R,
    extract_houses_and_subjects_or_exit,
    load_csv_or_exit,
    save_plot,
)

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

    df = load_csv_or_exit(parser, args.csv)
    houses, subject_columns = extract_houses_and_subjects_or_exit(parser, df)

    # Show all subjects in one figure so you can compare homogeneity visually.
    columns = 4
    rows = math.ceil(len(subject_columns) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(18, 3.8 * rows))
    axes = axes.flatten()

    for index, subject in enumerate(subject_columns):
        axis = axes[index]
        for house in houses:
            values = df.loc[df["Hogwarts House"] == house, subject].dropna()
            if not values.empty:
                axis.hist(values, bins=20, alpha=0.4, label=house)
        axis.set_title(subject, fontsize=10)
        axis.tick_params(labelsize=8)

    for axis in axes[len(subject_columns):]:
        axis.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(houses))

    fig.suptitle("Hogwarts Subjects by House", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = save_plot(fig, "histograms_by_house.png")
    plt.close(fig)
    print(f"{_G}Saved{_R} {_D}:{_R} {_C}{output_path}{_R}")


if __name__ == "__main__":
    main()
