import numpy as np
import pandas as pd
import sys
import os


def sigmoid(z):
    """Compute the sigmoid of z."""
    return 1 / (1 + np.exp(-z))


def cost_function(theta, X, y, lambda_):
    """Compute the regularized cost for logistic regression."""
    m = len(y)
    h = sigmoid(X.dot(theta))
    cost = (-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h))) / m
    reg = (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))
    return cost + reg


def compute_gradient(theta, X, y, lambda_):
    """Compute the gradient of the regularized cost function."""
    m = len(y)
    h = sigmoid(X.dot(theta))
    grad = (X.T.dot(h - y)) / m
    grad[1:] += (lambda_ / m) * theta[1:]
    return grad


def gradient_descent(X, y, theta, learning_rate, n_iterations, lambda_):
    """Perform gradient descent optimization."""
    for _ in range(n_iterations):
        grad = compute_gradient(theta, X, y, lambda_)
        theta -= learning_rate * grad
    return theta


def one_vs_all(X, y, num_labels, lambda_, learning_rate=0.00001,
               n_iterations=500):
    """
    Train multiple logistic regression classifiers (one vs all).
    
    Returns a matrix of shape (num_labels, n_features).
    """
    m, n = X.shape
    all_theta = np.zeros((num_labels, n))

    for i in range(num_labels):
        y_i = (y == i).astype(int)
        initial_theta = np.zeros(n)
        all_theta[i] = gradient_descent(
            X, y_i, initial_theta, learning_rate, n_iterations, lambda_
        )

    return all_theta


def logreg_train(dataset_file):
    """
    Train a regularized logistic regression model on the given dataset.

    Saves model weights and normalization stats in 'Outputs/training'.
    """
    data = pd.read_csv(dataset_file)
    output_dir = "Outputs/training"
    os.makedirs(output_dir, exist_ok=True)

    selected_features = [
        "Charms",
        "Ancient Runes",
        "Defense Against the Dark Arts",
        "Herbology",
        "Flying",
        "Astronomy"
    ]

    data = data.dropna(subset=selected_features)
    X = np.array(data[selected_features], dtype=float)
    y = data["Hogwarts House"].values

    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std
    X = np.clip(X, -5, 5)

    stats = pd.DataFrame({
        'Feature': selected_features,
        'Mean': X_mean,
        'Std': X_std
    })
    stats_path = os.path.join(output_dir, "stats.csv")
    stats.to_csv(stats_path, index=False)

    unique_labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_encoded = np.array([label_map[label] for label in y])

    X = np.hstack([np.ones((X.shape[0], 1)), X])

    lambda_ = 1.0
    num_labels = len(unique_labels)
    all_theta = one_vs_all(
        X, y_encoded, num_labels,
        lambda_, learning_rate=0.001, n_iterations=1000
    )

    weights_path = os.path.join(output_dir, "weights.csv")
    np.savetxt(weights_path, all_theta, delimiter=',')

    print(
        "Model trained successfully. "
        "Files 'weights.csv' and 'stats.csv' successfully generated."
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py dataset.csv")
        sys.exit(1)

    dataset_file = sys.argv[1]

    if not os.path.isfile(dataset_file):
        print(f"Error: The dataset file '{dataset_file}' does not exist.")
        sys.exit(1)

    logreg_train(dataset_file)
