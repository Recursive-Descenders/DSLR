import json, os, sys
import numpy as np
import csv


def load_model(path="model/model.json"):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                model = json.load(f)
                if "mu" in model and isinstance(model["mu"], list):
                    model["mu"] = np.array(model["mu"])
                if "sigma" in model and isinstance(model["sigma"], list):
                    model["sigma"] = np.array(model["sigma"])
                if "medians" in model and isinstance(model["medians"], list):
                    model["medians"] = np.array(model["medians"], dtype=float)
                for house in model["theta"]:
                    model["theta"][house] = np.asarray(model["theta"][house], dtype=float)
                return model
        except Exception as e:
            print(f"Error: cannot read {path}: {e}")
            sys.exit(1)
    return {"theta": {}, "mu": np.array([0.0]), "sigma": np.array([1.0]), "medians": None}


def apply_median_imputation(x, medians):
    x = np.asarray(x, dtype=float).copy()
    m = np.asarray(medians, dtype=float)
    for j in range(x.shape[1]):
        col = x[:, j]
        col[np.isnan(col)] = m[j]
    return x


def load_x_test(csv_path):
    """Same feature construction as training. House column may be empty (e.g. dataset_test.csv)."""
    xs = []
    indices = []
    actual_houses = []

    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                if not row or len(row) < 19:
                    continue

                try:
                    index = row[0].strip()
                    house = row[1].strip()

                    bh = row[5].strip()
                    best_hand_val = np.nan if not bh else (1.0 if bh == "Right" else 0.0)

                    x_row = [best_hand_val]
                    for i in range(6, 19):
                        v = row[i].strip()
                        x_row.append(np.nan if v == "" else float(v))

                    xs.append(x_row)
                    indices.append(index)
                    actual_houses.append(house)

                except (ValueError, IndexError):
                    continue

    except Exception as e:
        print(f"Error: cannot read {csv_path}: {e}", file=sys.stderr)
        sys.exit(1)

    return np.array(xs, dtype=float), indices, actual_houses


def standardize(x, mu, sigma):
    if isinstance(mu, (int, float)):
        mu = np.array([mu])
    if isinstance(sigma, (int, float)):
        sigma = np.array([sigma])
    return (x - mu) / sigma


def predict_probability(x, theta):
    ones = np.ones((x.shape[0], 1))
    xb = np.hstack([ones, x])
    z = np.clip(xb.dot(theta), -500, 500)
    return np.clip(1 / (1 + np.exp(-z)), 0.0, 1.0)


HOUSES = ("Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin")


def predict_house_save_csv(probabilities, indices):
    """Argmax over one-vs-all scores; writes subject-format houses.csv."""
    n = len(indices)
    prob_matrix = np.column_stack([probabilities[h] for h in HOUSES])
    predicted_houses = [HOUSES[j] for j in np.argmax(prob_matrix, axis=1)]

    with open("houses.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Index", "Hogwarts House"])
        for i in range(n):
            w.writerow([indices[i], predicted_houses[i]])
    return predicted_houses


def main():
    """Usage: logreg_predict.py [dataset_test.csv] [model/model.json]"""
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "dataset_test.csv"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "model/model.json"

    m = load_model(model_path)
    theta = m.get("theta")
    mu = m.get("mu", np.array([0.0]))
    sigma = m.get("sigma", np.array([1.0]))
    medians = m.get("medians")

    if not theta or not isinstance(theta, dict):
        print("Error: Model not found or invalid. Please train the model first.", file=sys.stderr)
        sys.exit(1)
    theta = {k: np.asarray(v, dtype=float) for k, v in theta.items()}

    if medians is None or not isinstance(medians, np.ndarray) or medians.size == 0:
        print("Error: model missing 'medians' (retrain with logreg_train).", file=sys.stderr)
        sys.exit(1)
    medians = np.asarray(medians, dtype=float)

    if not isinstance(mu, np.ndarray):
        mu = np.array([mu] if isinstance(mu, (int, float)) else mu)
    if not isinstance(sigma, np.ndarray):
        sigma = np.array([sigma] if isinstance(sigma, (int, float)) else sigma)

    for house in HOUSES:
        if house not in theta or theta[house].size == 0:
            print(f"Error: missing or empty theta for {house}. Please train the model first.", file=sys.stderr)
            sys.exit(1)

    xs_array, indices, actual_houses = load_x_test(csv_file)
    if len(xs_array) == 0 or xs_array.ndim != 2 or xs_array.shape[0] == 0:
        print("Error: no valid test samples after loading.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(xs_array)} test samples")

    if xs_array.shape[1] != medians.shape[0]:
        print("Error: feature count does not match model medians.", file=sys.stderr)
        sys.exit(1)

    xs_scaled = standardize(apply_median_imputation(xs_array, medians), mu, sigma)
    probabilities = {house: predict_probability(xs_scaled, theta[house]) for house in theta}
    predicted_houses = predict_house_save_csv(probabilities, indices)

    print("Predictions saved to houses.csv")

    labeled = [(a, p) for a, p in zip(actual_houses, predicted_houses) if a]
    if labeled:
        correct = sum(1 for a, p in labeled if a == p)
        print(f"Accuracy (labeled rows only): {100 * correct / len(labeled):.2f}% ({correct}/{len(labeled)})")


if __name__ == "__main__":
    main()
