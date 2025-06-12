import sys
import pandas as pd
import numpy as np
import os


def sigmoid(z):
    """Compute the sigmoid of z."""
    with np.errstate(over='ignore'):
        return 1 / (1 + np.exp(-z))


def predict(X, weights):
    """
    Predict class probabilities and return class with highest probability.

    Args:
        X (np.ndarray): Normalized input features with bias term.
        weights (np.ndarray): Trained weights for each class.

    Returns:
        np.ndarray: Predicted class indices.
    """
    probabilities = sigmoid(np.dot(X, weights.T))
    return np.argmax(probabilities, axis=1)


def logreg_predict(dataset_file, weights_file, stats_file):
    """
    Predict Hogwarts houses using a trained logistic regression model.

    Args:
        dataset_file (str): Path to CSV file containing test data.
        weights_file (str): Path to CSV file containing trained weights.
        stats_file (str): Path to CSV file containing normalization stats.

    Saves:
        Outputs/prediction/houses.csv with predicted house labels.
    """
    try:
        data = pd.read_csv(dataset_file)
    except Exception as e:
        print(f"Error reading dataset file '{dataset_file}': {e}")
        sys.exit(1)

    selected_features = [
        "Charms",
        "Ancient Runes",
        "Defense Against the Dark Arts",
        "Herbology",
        "Flying",
        "Astronomy"
    ]

    missing_features = set(selected_features) - set(data.columns)
    if missing_features:
        print(f"Error: The test data is missing columns: {missing_features}")
        sys.exit(1)

    data[selected_features] = data[selected_features].fillna(data[selected_features].median())

    try:
        X = data[selected_features].astype(float).values
    except ValueError as e:
        print(f"Error: Non-numeric values detected in features after fillna: {e}")
        sys.exit(1)

    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("Error: Features contain NaN or infinite values after preprocessing.")
        sys.exit(1)

    try:
        stats = pd.read_csv(stats_file)
    except Exception as e:
        print(f"Error reading stats file '{stats_file}': {e}")
        sys.exit(1)

    if not {'Mean', 'Std'}.issubset(stats.columns):
        print(f"Error: Stats file '{stats_file}' missing required columns 'Mean' and/or 'Std'.")
        sys.exit(1)

    X_mean = stats["Mean"].to_numpy()
    X_std = stats["Std"].to_numpy()

    if len(X_mean) != len(selected_features) or len(X_std) != len(selected_features):
        print(f"Error: Stats file columns length mismatch with features count.")
        sys.exit(1)

    if np.any(X_std == 0):
        print("Warning: Some features have zero standard deviation. Adjusting to 1 to avoid division by zero.")
        X_std = np.where(X_std == 0, 1, X_std)

    X = (X - X_mean) / X_std
    X = np.clip(X, -5, 5)

    X = np.hstack([np.ones((X.shape[0], 1)), X])

    try:
        weights = np.loadtxt(weights_file, delimiter=",")
    except Exception as e:
        print(f"Error reading weights file '{weights_file}': {e}")
        sys.exit(1)

    if X.shape[1] != weights.shape[1]:
        print(f"Error: Dimension mismatch: X has {X.shape[1]} columns, weights have {weights.shape[1]} columns.")
        sys.exit(1)

    predictions = predict(X, weights)

    unique_labels = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

    if np.any(predictions >= len(unique_labels)):
        print("Error: Predictions contain invalid class indices.")
        sys.exit(1)

    predicted_houses = [unique_labels[p] for p in predictions]

    results = pd.DataFrame({
        "Index": np.arange(len(predicted_houses)),
        "Hogwarts House": predicted_houses
    })

    output_dir = "Outputs/prediction"
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "houses.csv")
    try:
        results.to_csv(results_path, index=False)
    except Exception as e:
        print(f"Error saving predictions to '{results_path}': {e}")
        sys.exit(1)

    print("Predictions saved in the 'Outputs' folder as 'houses.csv'.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python logreg_predict.py dataset.csv")
        sys.exit(1)

    dataset_file = sys.argv[1]
    if not os.path.isfile(dataset_file):
        print(f"Error: The dataset file '{dataset_file}' does not exist.")
        sys.exit(1)

    weights_file = "Outputs/training/weights.csv"
    stats_file = "Outputs/training/stats.csv"

    if not os.path.isfile(weights_file):
        print(f"Error: The weights file '{weights_file}' does not exist.")
        sys.exit(1)

    if not os.path.isfile(stats_file):
        print(f"Error: The stats file '{stats_file}' does not exist.")
        sys.exit(1)

    logreg_predict(dataset_file, weights_file, stats_file)
