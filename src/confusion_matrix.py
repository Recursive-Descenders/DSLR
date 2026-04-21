import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

if os.environ.get("NO_COLOR"):
    _R = _G = _C = _D = ""
else:
    _R = "\033[0m"
    _G = "\033[32m"
    _C = "\033[36m"
    _D = "\033[2m"

EXPECTED_HEADERS = ("Index", "Hogwarts House")
HOUSES = ("Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin")


def _confusion_matrix_counts(y_true: list[str], y_pred: list[str], labels: tuple[str, ...]) -> list[list[int]]:
    """Counts where row i = true label, column j = predicted label (same ordering as labels)."""
    n = len(labels)
    ix = {h: i for i, h in enumerate(labels)}
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if t not in ix:
            print(f"Error: unknown true label {t!r} (expected one of {labels}).", file=sys.stderr)
            sys.exit(1)
        if p not in ix:
            print(f"Error: unknown predicted label {p!r} (expected one of {labels}).", file=sys.stderr)
            sys.exit(1)
        cm[ix[t]][ix[p]] += 1
    return cm


def _load_labels_csv(path: str) -> dict[int, str]:
    rows: dict[int, str] = {}
    try:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                print(f"Error: {path} is empty.", file=sys.stderr)
                sys.exit(1)
            if tuple(h.strip() for h in header[:2]) != EXPECTED_HEADERS:
                print(
                    f"Error: {path}: expected header {EXPECTED_HEADERS!r}, got {header[:2]!r}.",
                    file=sys.stderr,
                )
                sys.exit(1)
            for line_no, row in enumerate(reader, start=2):
                if not row or len(row) < 2:
                    continue
                try:
                    idx = int(row[0].strip())
                except ValueError:
                    print(f"Error: {path}:{line_no}: invalid Index {row[0]!r}.", file=sys.stderr)
                    sys.exit(1)
                house = row[1].strip()
                if not house:
                    print(f"Error: {path}:{line_no}: empty Hogwarts House for index {idx}.", file=sys.stderr)
                    sys.exit(1)
                rows[idx] = house
    except FileNotFoundError:
        print(f"Error: cannot open {path}: file not found.", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Error: cannot read {path}: {e}", file=sys.stderr)
        sys.exit(1)
    if not rows:
        print(f"Error: no data rows in {path}.", file=sys.stderr)
        sys.exit(1)
    return rows


def _plot_matrix(ax, cm: np.ndarray, row_labels: list[str], col_labels: list[str], title: str) -> None:
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    tick_marks_rows = np.arange(cm.shape[0])
    tick_marks_cols = np.arange(cm.shape[1])
    ax.set_xticks(tick_marks_cols)
    ax.set_yticks(tick_marks_rows)
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    ax.set_title(title)
    thresh = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the multiclass confusion matrix for Hogwarts predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("truth", help="Path to ground-truth CSV (Index, Hogwarts House)")
    parser.add_argument(
        "predictions",
        nargs="?",
        default="houses.csv",
        help="Path to predictions CSV",
    )
    args = parser.parse_args()

    pred = _load_labels_csv(args.predictions)
    truth = _load_labels_csv(args.truth)

    common = sorted(set(pred) & set(truth))
    if not common:
        print("Error: no overlapping Index between predictions and ground truth.", file=sys.stderr)
        sys.exit(1)

    missing_in_pred = set(truth) - set(pred)
    missing_in_truth = set(pred) - set(truth)
    if missing_in_pred:
        print(f"Warning: {len(missing_in_pred)} indices in truth missing from predictions.", file=sys.stderr)
    if missing_in_truth:
        print(f"Warning: {len(missing_in_truth)} indices in predictions missing from truth.", file=sys.stderr)

    y_true = [truth[i] for i in common]
    y_pred = [pred[i] for i in common]

    cm_all = np.asarray(_confusion_matrix_counts(y_true, y_pred, HOUSES), dtype=int)
    fig, ax = plt.subplots(figsize=(8, 7))
    _plot_matrix(ax, cm_all, list(HOUSES), list(HOUSES), "Confusion matrix")

    fig.suptitle(
        f"n={len(common)} overlapping indices  |  truth: {args.truth}  predictions: {args.predictions}"
    )
    fig.tight_layout()

    os.makedirs("visualizations", exist_ok=True)
    out_path = os.path.join("visualizations", "confusion_matrix.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{_G}Saved{_R} {_D}:{_R} {_C}{out_path}{_R}")


if __name__ == "__main__":
    main()
