import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import csv


def load_xy(csv_path):
    xs = []
    ys_gryffindor = []
    ys_hufflepuff = []
    ys_ravenclaw = []
    ys_slytherin = []
    
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            
            for row in reader:
                if not row or len(row) < 19:
                    continue
                
                try:
                    # Extract Hogwarts House (column 1)
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
                    
                    # Create one-vs-all binary labels for each house
                    ys_gryffindor.append(1 if house == "Gryffindor" else 0)
                    ys_hufflepuff.append(1 if house == "Hufflepuff" else 0)
                    ys_ravenclaw.append(1 if house == "Ravenclaw" else 0)
                    ys_slytherin.append(1 if house == "Slytherin" else 0)
                    
                except (ValueError, IndexError) as e:
                    continue
                    
    except Exception as e:
        print(f"Error: cannot read {csv_path}: {e}")
        exit(0)
    
    xs_array = np.array(xs, dtype=float)
    ys_dict = {
        "Gryffindor": np.array(ys_gryffindor, dtype=int),
        "Hufflepuff": np.array(ys_hufflepuff, dtype=int),
        "Ravenclaw": np.array(ys_ravenclaw, dtype=int),
        "Slytherin": np.array(ys_slytherin, dtype=int)
    }
    
    return xs_array, ys_dict


def standardize(x):
    """Standardize each feature column independently"""
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    # Avoid division by zero
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    x_scaled = (x - mu) / sigma
    return x_scaled, mu, sigma



def logistic_regression(x, y, alpha=0.01, epochs=5000):
    """Logistic regression with bias term"""
    # Add bias term (column of ones) to features
    ones = np.ones((x.shape[0], 1))
    x_with_bias = np.hstack([ones, x])
    
    # Initialize theta (including bias term)
    theta = np.random.randn(x_with_bias.shape[1]) * 0.01
    
    for epoch in range(epochs):
        # Calculate hypothesis using sigmoid
        z = x_with_bias.dot(theta)
        # Clip z to prevent overflow
        z_clipped = np.clip(z, -500, 500)
        h = 1 / (1 + np.exp(-z_clipped))
        
        # Calculate gradient
        gradient = x_with_bias.T.dot(h - y) / len(y)
        
        # Update theta
        theta -= alpha * gradient
        
        # Optional: print progress every 1000 epochs
        if (epoch + 1) % 1000 == 0:
            # Calculate current accuracy for monitoring
            predictions = (h > 0.5).astype(int)
            accuracy = np.mean(predictions == y) * 100
            print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.2f}%")
    
    return theta

def main():
    csv_file = "dataset_train.csv"
    xs_array, ys_dict = load_xy(csv_file)
    standard_xs_array, mu, sigma = standardize(xs_array)
    theta = logistic_regression(standard_xs_array, ys_dict["Ravenclaw"])
    os.makedirs("model", exist_ok=True)
    with open("model/model.json", "w") as f:
        json.dump({
            "theta": theta.tolist(),
            "mu": mu.tolist() if isinstance(mu, np.ndarray) else mu,
            "sigma": sigma.tolist() if isinstance(sigma, np.ndarray) else sigma
        }, f)
    print(f"Training complete. Model saved to model/model.json")
    print(f"Theta shape: {theta.shape}, Mu shape: {mu.shape}, Sigma shape: {sigma.shape}")


if __name__ == "__main__":
    main()
