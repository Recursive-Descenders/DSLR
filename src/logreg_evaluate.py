import csv
import sys

from sklearn.metrics import accuracy_score

EXPECTED_HEADERS = ("Index", "Hogwarts House")


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


def evaluate(predictions_path: str, truth_path: str) -> float:
    pred = _load_labels_csv(predictions_path)
    truth = _load_labels_csv(truth_path)

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
    acc = float(accuracy_score(y_true, y_pred))
    pct = 100.0 * acc
    print(f"Accuracy: {pct:.4f}% ({int(acc * len(common))}/{len(common)} correct on overlapping indices)")
    return acc


def main() -> None:
    pred_path = sys.argv[1] if len(sys.argv) > 1 else "houses.csv"
    truth_path = sys.argv[2] if len(sys.argv) > 2 else "dataset_truth.csv"
    evaluate(pred_path, truth_path)


if __name__ == "__main__":
    main()
