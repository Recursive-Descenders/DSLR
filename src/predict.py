import json, os, sys
import numpy as np
import csv


def load_model(path="model/model.json"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            model = json.load(f)
            # Convert lists back to numpy arrays
            if "mu" in model and isinstance(model["mu"], list):
                model["mu"] = np.array(model["mu"])
            if "sigma" in model and isinstance(model["sigma"], list):
                model["sigma"] = np.array(model["sigma"])
            return model
    return {"theta": [], "mu": np.array([0.0]), "sigma": np.array([1.0])}


def load_x_test(csv_path):
    """Load test data features (same processing as train.py)"""
    xs = []
    indices = []
    actual_houses = []  # Store actual house labels for comparison
    
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            
            for row in reader:
                if not row or len(row) < 19:
                    continue
                
                try:
                    # Extract Index (column 0)
                    index = row[0].strip()
                    
                    # Extract Hogwarts House (column 1) for comparison
                    house = row[1].strip()
                    if not house:
                        continue
                    
                    # Extract Best Hand (column 5) and convert to binary
                    best_hand = row[5].strip()
                    if not best_hand:
                        continue
                    best_hand_binary = 1 if best_hand == "Right" else 0
                    
                    # Extract numeric features (columns 6-18)
                    x_row = [best_hand_binary]  # Start with Best Hand
                    has_nan = False
                    for i in range(6, 19):  # Columns 6 to 18
                        val = row[i].strip()
                        if val == "":
                            has_nan = True
                            break
                        x_row.append(float(val))
                    
                    # Skip this row if it contains NaN
                    if has_nan:
                        continue
                    
                    xs.append(x_row)
                    indices.append(index)
                    actual_houses.append(house)
                    
                except (ValueError, IndexError) as e:
                    continue
                    
    except Exception as e:
        print(f"Error: cannot read {csv_path}: {e}")
        exit(0)
    
    xs_array = np.array(xs, dtype=float)
    return xs_array, indices, actual_houses


def standardize(x, mu, sigma):
    """Standardize features using training mu and sigma (per-feature)"""
    # Handle both scalar and array mu/sigma for backward compatibility
    if isinstance(mu, (int, float)):
        mu = np.array([mu])
    if isinstance(sigma, (int, float)):
        sigma = np.array([sigma])
    
    x_scaled = (x - mu) / sigma
    return x_scaled


def predict(x, theta):
    """Make predictions using logistic regression with bias term"""
    # Add bias term (column of ones) to features
    ones = np.ones((x.shape[0], 1))
    x_with_bias = np.hstack([ones, x])
    
    # Calculate probability using sigmoid function with numerical stability
    z = x_with_bias.dot(theta)
    # Clip z to prevent overflow in exp(-z)
    z_clipped = np.clip(z, -500, 500)
    probabilities = 1 / (1 + np.exp(-z_clipped))
    # Ensure probabilities are in [0, 1]
    probabilities = np.clip(probabilities, 0.0, 1.0)
    return probabilities


def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "dataset_train.csv"
    
    # Load model
    m = load_model()
    theta = np.array(m.get("theta", []))
    mu = m.get("mu", np.array([0.0]))
    sigma = m.get("sigma", np.array([1.0]))
    
    # Ensure mu and sigma are numpy arrays
    if not isinstance(mu, np.ndarray):
        mu = np.array([mu] if isinstance(mu, (int, float)) else mu)
    if not isinstance(sigma, np.ndarray):
        sigma = np.array([sigma] if isinstance(sigma, (int, float)) else sigma)
    
    if len(theta) == 0:
        print("Error: Model not found or invalid. Please train the model first.")
        return
    
    print(f"Loaded model: theta shape={theta.shape}, mu shape={mu.shape}, sigma shape={sigma.shape}")
    
    # Load test data
    xs_array, indices, actual_houses = load_x_test(csv_file)
    print(f"Loaded {len(xs_array)} test samples")
    
    # Standardize features
    xs_scaled = standardize(xs_array, mu, sigma)
    
    # Make predictions
    probabilities = predict(xs_scaled, theta)
    
    # Convert probabilities to binary labels (1 for Gryffindor, 0 for not Gryffindor)
    # Then convert to output format: 0 for Gryffindor, 1 for not Gryffindor
    binary_predictions = (probabilities > 0.5).astype(int)
    output_labels = 1 - binary_predictions  # Invert: 0 for Gryffindor, 1 for not Gryffindor
    
    # Convert actual houses to binary labels for accuracy calculation
    # 1 for Gryffindor, 0 for not Gryffindor (matching training labels)
    actual_binary_labels = np.array([1 if house == "Ravenclaw" else 0 for house in actual_houses])
    
    # Calculate accuracy
    correct_predictions = np.sum(binary_predictions == actual_binary_labels)
    total_predictions = len(binary_predictions)
    accuracy = (correct_predictions / total_predictions) * 100
    
    # Print predictions
    print("\nPredictions:")
    print("Index\tActual House\tProbability\tPredicted Label (0=Gryffindor, 1=Not Gryffindor)")
    for idx, house, prob, label in zip(indices, actual_houses, probabilities, output_labels):
        print(f"{idx}\t{house}\t{prob:.6f}\t{label}")
    
    # Print accuracy
    print(f"\nAccuracy: {correct_predictions}/{total_predictions} = {accuracy:.2f}%")


if __name__ == "__main__":
    main()
