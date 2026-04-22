import os

import pandas


if os.environ.get("NO_COLOR"):
    _R = _E = _G = _C = _D = _B = ""
else:
    _R = "\033[0m"
    _E = "\033[31m"
    _G = "\033[32m"
    _C = "\033[36m"
    _D = "\033[2m"
    _B = "\033[1m"


# House colors in alphabetical order (Gryffindor, Hufflepuff, Ravenclaw, Slytherin)
HOUSE_COLORS = [
    "#e74c3c",  # Gryffindor - Red
    "#f39c12",  # Hufflepuff - Gold/Yellow
    "#3498db",  # Ravenclaw - Blue
    "#27ae60",  # Slytherin - Green
]


def load_csv_or_exit(parser, csv_path: str, only_numeric: bool = True) -> pandas.DataFrame:
    try:
        df = pandas.read_csv(csv_path)
        if only_numeric:
            df = df.select_dtypes(include="number")
        return df
    except FileNotFoundError:
        parser.exit(1, f"error: CSV file not found: {csv_path}\n")
    except pandas.errors.ParserError as error:
        parser.exit(1, f"error: failed to parse CSV '{csv_path}': {error}\n")
    except Exception as error:
        parser.exit(1, f"error: failed to read CSV '{csv_path}': {error}\n")


def extract_houses_and_subjects_or_exit(
    parser,
    df: pandas.DataFrame,
    max_houses: int = 4,
    max_subjects: int = 13,
) -> tuple[list[str], list[str]]:
    if "Hogwarts House" not in df.columns:
        parser.exit(1, "error: missing required column 'Hogwarts House'\n")

    houses = sorted(df["Hogwarts House"].dropna().unique())
    if len(houses) == 0:
        parser.exit(1, "error: expected at least 1 house, found 0\n")
    if len(houses) > max_houses:
        parser.exit(1, f"error: expected at most {max_houses} houses, found {len(houses)}\n")

    numeric_columns = list(df.select_dtypes(include="number").columns)
    subject_columns = [column for column in numeric_columns if column != "Index"]

    if len(subject_columns) == 0:
        parser.exit(1, "error: expected at least 1 numeric subject, found 0\n")
    if len(subject_columns) > max_subjects:
        parser.exit(
            1,
            f"error: expected at most {max_subjects} subjects, found {len(subject_columns)}\n",
        )

    return houses, subject_columns


def get_house_color_map(houses: list[str]) -> dict[str, str]:
    """Create a mapping from house names to their colors."""
    return {house: HOUSE_COLORS[index] for index, house in enumerate(houses)}


def save_plot(fig, file_name: str, output_dir: str = "visualizations", dpi: int = 150) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    fig.savefig(output_path, dpi=dpi)
    return output_path
