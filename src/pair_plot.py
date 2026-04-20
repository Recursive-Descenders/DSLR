import argparse
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
        help="Path to CSV",
    )
    args = parser.parse_args(sys.argv[1:])

    df = load_csv_or_exit(parser, args.csv)
    houses, subject_columns = extract_houses_and_subjects_or_exit(parser, df)

    if len(subject_columns) < 2:
        parser.exit(1, "error: need at least 2 numeric subjects to build a pair plot\n")

    palette = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"]
    house_to_color = {house: palette[index] for index, house in enumerate(houses)}

    subject_count = len(subject_columns)
    cell_size = 2.3
    fig, axes = plt.subplots(
        subject_count,
        subject_count,
        figsize=(cell_size * subject_count, cell_size * subject_count),
    )

    for row, y_subject in enumerate(subject_columns):
        for col, x_subject in enumerate(subject_columns):
            axis = axes[row, col]
            if row == col:
                for house in houses:
                    values = df.loc[df["Hogwarts House"] == house, y_subject].dropna()
                    if not values.empty:
                        axis.hist(values, bins=18, alpha=0.5, color=house_to_color[house])
            else:
                for house in houses:
                    pair_data = df.loc[
                        df["Hogwarts House"] == house,
                        [x_subject, y_subject],
                    ].dropna()
                    if not pair_data.empty:
                        axis.scatter(
                            pair_data[x_subject],
                            pair_data[y_subject],
                            alpha=0.55,
                            s=8,
                            c=house_to_color[house],
                        )

            axis.tick_params(labelsize=6)
            if row == subject_count - 1:
                axis.set_xlabel(x_subject, fontsize=7, rotation=45)
            else:
                axis.set_xticklabels([])
            if col == 0:
                axis.set_ylabel(y_subject, fontsize=7)
            else:
                axis.set_yticklabels([])

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=house,
            markerfacecolor=house_to_color[house],
            markersize=7,
        )
        for house in houses
    ]
    fig.legend(legend_handles, houses, loc="upper center", ncol=len(houses))

    fig.suptitle("Pair Plot Matrix by House", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = save_plot(fig, "pair_plot_matrix.png", dpi=130)
    plt.close(fig)

    print(f"{_G}Saved{_R} {_D}:{_R} {_C}{output_path}{_R}")
    print("Use this matrix to manually choose features with class separation and low redundancy.")


if __name__ == "__main__":
    main()
