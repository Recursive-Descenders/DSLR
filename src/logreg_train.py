import json, os, sys
import numpy as np
import csv
from arg_parser import build_parser

if os.environ.get("NO_COLOR"):
    _R = _E = _G = _C = _D = _B = ""
else:
    _R = "\033[0m"
    _E = "\033[31m"
    _G = "\033[32m"
    _C = "\033[36m"
    _D = "\033[2m"
    _B = "\033[1m"


def binary_cross_entropy(y_true, h, eps=1e-15):
    h = np.clip(h, eps, 1.0 - eps)
    y_true = np.asarray(y_true, dtype=float)
    return -np.mean(y_true * np.log(h) + (1.0 - y_true) * np.log(1.0 - h))

def load_xy(csv_path):
    """Load training CSV into X and per-house binary labels (one-vs-all). Missing values as NaN."""
    xs = []
    ys_gryffindor = []
    ys_hufflepuff = []
    ys_ravenclaw = []
    ys_slytherin = []

    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                if not row or len(row) < 19:
                    continue

                try:
                    house = row[1].strip()
                    if not house:
                        continue

                    bh = row[5].strip()
                    best_hand_val = np.nan if not bh else (1.0 if bh == "Right" else 0.0)

                    x_row = [best_hand_val]
                    for i in range(6, 19):
                        v = row[i].strip()
                        x_row.append(np.nan if v == "" else float(v))

                    xs.append(x_row)

                    ys_gryffindor.append(1 if house == "Gryffindor" else 0)
                    ys_hufflepuff.append(1 if house == "Hufflepuff" else 0)
                    ys_ravenclaw.append(1 if house == "Ravenclaw" else 0)
                    ys_slytherin.append(1 if house == "Slytherin" else 0)

                except (ValueError, IndexError):
                    continue

    except FileNotFoundError:
        print(f"Error: cannot read {csv_path}: no such file or directory", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: cannot read {csv_path}: {e}", file=sys.stderr)
        sys.exit(1)

    xs_array = np.asarray(xs, dtype=float)
    ys_dict = {
        "Gryffindor": np.array(ys_gryffindor, dtype=int),
        "Hufflepuff": np.array(ys_hufflepuff, dtype=int),
        "Ravenclaw": np.array(ys_ravenclaw, dtype=int),
        "Slytherin": np.array(ys_slytherin, dtype=int),
    }

    return xs_array, ys_dict

def column_medians_for_imputation(x):
    """Training-only column medians (nanmedian); all-NaN columns become 0."""
    med = np.nanmedian(x, axis=0)
    return np.where(np.isnan(med), 0.0, med)

def apply_median_imputation(x, medians):
    x = np.asarray(x, dtype=float).copy()
    m = np.asarray(medians, dtype=float)
    for j in range(x.shape[1]):
        col = x[:, j]
        col[np.isnan(col)] = m[j]
    return x

def standardize(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (x - mu) / sigma, mu, sigma

def logistic_regression(x, y, alpha=0.01, epochs=200, batch_size=None, track_loss=False):
    """Binary logistic regression; theta[0] is bias. batch_size None or m => batch GD; else minibatch or SGD (batch 1)."""
    ones = np.ones((x.shape[0], 1))
    xb = np.hstack([ones, x])
    m = xb.shape[0]
    theta = np.random.randn(xb.shape[1]) * 0.01
    y = np.asarray(y, dtype=float)
    losses = [] if track_loss else None

    if batch_size is None:
        batch_size = m
    elif batch_size <= 0:
        raise ValueError("batch_size must be positive")

    for epoch in range(epochs):
        indices = np.random.permutation(m)
        x_shuffled = xb[indices]
        y_shuffled = y[indices]

        for start in range(0, m, batch_size):
            end = start + batch_size
            x_batch = x_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            z = np.clip(x_batch.dot(theta), -500, 500)
            h = 1 / (1 + np.exp(-z))
            gradient = x_batch.T.dot(h - y_batch) / len(y_batch)
            theta = theta - alpha * gradient

        if track_loss:
            z_full = np.clip(xb.dot(theta), -500, 500)
            h_full = 1 / (1 + np.exp(-z_full))
            losses.append(binary_cross_entropy(y, h_full))

    return theta, losses

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.optimizer == "gd" and args.batch_size is not None:
        parser.error("--batch-size applies only to --optimizer mbgd")
    if args.optimizer == "sgd" and args.batch_size is not None:
        parser.error("--batch-size is not used with --optimizer sgd (one example per update)")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be a positive integer")

    csv_file = args.csv
    xs_array, ys_dict = load_xy(csv_file)
    if xs_array.size == 0 or xs_array.ndim != 2 or xs_array.shape[0] == 0:
        print("Error: no valid training samples after loading.", file=sys.stderr)
        sys.exit(1)

    m = xs_array.shape[0]
    medians = column_medians_for_imputation(xs_array)
    xs_imputed = apply_median_imputation(xs_array, medians)
    standard_xs_array, mu, sigma = standardize(xs_imputed)

    if args.optimizer == "gd":
        train_batch_size = None
    elif args.optimizer == "sgd":
        train_batch_size = 1
    else:
        if m < 2:
            parser.error("--optimizer mbgd requires at least 2 training samples")
        if args.batch_size is None:
            train_batch_size = min(m, max(2, int(m * 0.25)))
        else:
            if args.batch_size == 1:
                parser.error("batch size 1 is SGD; use --optimizer sgd")
            if args.batch_size > m:
                parser.error(
                    f"--batch-size must be at most the number of training samples ({m})"
                )
            train_batch_size = args.batch_size

    n_params = standard_xs_array.shape[1] + 1
    houses = ("Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin")
    theta = {h: [0.0] * n_params for h in houses}
    loss_by_house = {}
    for house in theta:
        print(f"{_D}train{_R} {_C}{house}{_R}")
        th, losses = logistic_regression(
            standard_xs_array,
            ys_dict[house],
            alpha=args.lr,
            epochs=args.epochs,
            batch_size=train_batch_size,
            track_loss=args.plot_loss,
        )
        theta[house] = th.tolist()
        if losses is not None:
            loss_by_house[house] = losses

    os.makedirs("model", exist_ok=True)
    if args.plot_loss and loss_by_house:
        import matplotlib.pyplot as plt

        os.makedirs("visualizations", exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        epochs_axis = np.arange(1, args.epochs + 1)
        for ax, house in zip(axes.flat, houses):
            ax.plot(epochs_axis, loss_by_house[house])
            ax.set_title(house)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (binary cross-entropy)")
        fig.suptitle("Training loss vs epoch (one-vs-all per house)")
        fig.tight_layout()
        out_path = os.path.join("visualizations", "training_loss.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"{_G}Loss curves saved{_R} {_D}:{_R} {out_path}")
    with open("model/model.json", "w") as f:
        json.dump(
            {
                "theta": theta,
                "mu": mu.tolist() if isinstance(mu, np.ndarray) else mu,
                "sigma": sigma.tolist() if isinstance(sigma, np.ndarray) else sigma,
                "medians": medians.tolist() if isinstance(medians, np.ndarray) else medians,
            },
            f,
            indent=2,
        )
    print(f"{_G}Training complete.{_R} Model saved to {_C}model/model.json{_R}")
    print(
        f"{_D}Mu{_R} {len(mu)}  {_D}Sigma{_R} {len(sigma)}  "
        f"{_D}theta per house{_R} { {h: len(theta[h]) for h in theta} }"
    )

if __name__ == "__main__":
    main()
