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


def build_house_color_map(houses: list[str]) -> dict[str, str]:
    palette = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"]
    return {house: palette[index] for index, house in enumerate(houses)}


def plot_diagonal_histogram(axis, df, houses: list[str], subject: str, house_to_color: dict[str, str]) -> None:
    for house in houses:
        values = df.loc[df["Hogwarts House"] == house, subject].dropna()
        if not values.empty:
            axis.hist(values, bins=18, alpha=0.5, color=house_to_color[house])


def plot_off_diagonal_scatter(
    axis,
    df,
    houses: list[str],
    x_subject: str,
    y_subject: str,
    house_to_color: dict[str, str],
) -> None:
    for house in houses:
        house_rows = df["Hogwarts House"] == house
        subject_pair = [x_subject, y_subject]
        pair_data = df.loc[house_rows, subject_pair].dropna()
        if not pair_data.empty:
            axis.scatter(
                pair_data[x_subject],
                pair_data[y_subject],
                alpha=0.55,
                s=8,
                c=house_to_color[house],
            )


def style_matrix_axis(
    axis,
    row: int,
    col: int,
    subject_count: int,
    x_subject: str,
    y_subject: str,
) -> None:
    axis.tick_params(labelsize=6)
    if row == subject_count - 1:
        axis.set_xlabel(x_subject, fontsize=7, rotation=45)
    else:
        axis.set_xticklabels([])

    if col == 0:
        axis.set_ylabel(y_subject, fontsize=7)
    else:
        axis.set_yticklabels([])


def create_house_legend_handles(houses: list[str], house_to_color: dict[str, str]) -> list[Line2D]:
    return [
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


def create_pair_plot_figure(df, houses: list[str], subject_columns: list[str]):
    house_to_color = build_house_color_map(houses)
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
                plot_diagonal_histogram(axis, df, houses, y_subject, house_to_color)
            else:
                plot_off_diagonal_scatter(
                    axis,
                    df,
                    houses,
                    x_subject,
                    y_subject,
                    house_to_color,
                )

            style_matrix_axis(axis, row, col, subject_count, x_subject, y_subject)

    legend_handles = create_house_legend_handles(houses, house_to_color)
    fig.legend(legend_handles, houses, loc="upper center", ncol=len(houses))
    fig.suptitle("Pair Plot Matrix by House", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


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

    fig = create_pair_plot_figure(df, houses, subject_columns)

    output_path = save_plot(fig, "pair_plot_matrix.png", dpi=130)
    plt.close(fig)

    print(f"{_G}Saved{_R} {_D}:{_R} {_C}{output_path}{_R}")
    print("Use this matrix to manually choose features with class separation and low redundancy.")


if __name__ == "__main__":
    main()
