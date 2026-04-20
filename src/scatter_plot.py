import argparse
import itertools
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
        help="Path to CSV",
    )
    args = parser.parse_args(sys.argv[1:])

    df = load_csv_or_exit(parser, args.csv)
    houses, subject_columns = extract_houses_and_subjects_or_exit(parser, df)

    if len(subject_columns) < 2:
        parser.exit(1, "error: need at least 2 numeric subjects to build a scatter plot\n")

    pairs = list(itertools.combinations(subject_columns, 2))
    if not pairs:
        parser.exit(1, "error: no subject pairs available for scatter plots\n")

    plots_per_page = 9
    columns = 3
    rows = math.ceil(plots_per_page / columns)
    page_count = math.ceil(len(pairs) / plots_per_page)

    for page_index in range(page_count):
        start = page_index * plots_per_page
        end = start + plots_per_page
        page_pairs = pairs[start:end]
        fig, axes = plt.subplots(rows, columns, figsize=(16, 12))
        axes = axes.flatten()
        legend_handles = None
        legend_labels = None

        for axis, (x_subject, y_subject) in zip(axes, page_pairs):
            for house in houses:
                house_data = df.loc[df["Hogwarts House"] == house, [x_subject, y_subject]].dropna()
                if not house_data.empty:
                    scatter = axis.scatter(
                        house_data[x_subject],
                        house_data[y_subject],
                        alpha=0.55,
                        s=10,
                        label=house,
                    )

            axis.set_title(f"{x_subject} vs {y_subject}", fontsize=10)
            axis.set_xlabel(x_subject, fontsize=9)
            axis.set_ylabel(y_subject, fontsize=9)
            axis.tick_params(labelsize=8)

            if legend_handles is None:
                legend_handles, legend_labels = axis.get_legend_handles_labels()

        for axis in axes[len(page_pairs):]:
            axis.axis("off")

        if legend_handles:
            fig.legend(legend_handles, legend_labels, loc="upper center", ncol=len(houses))

        fig.suptitle(
            f"Scatter Plots by Subject Pair ({page_index + 1}/{page_count})",
            fontsize=16,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        output_name = f"scatter_pairs_page_{page_index + 1:02d}.png"
        output_path = save_plot(fig, output_name)
        plt.close(fig)
        print(f"{_G}Saved{_R} {_D}:{_R} {_C}{output_path}{_R}")

    print(f"Generated {len(pairs)} subject-pair scatter plots across {page_count} page(s).")


if __name__ == "__main__":
    main()