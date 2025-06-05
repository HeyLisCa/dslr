import numpy as np
import pandas as pd
import sys
import os


def sigmoid(z):
    """Compute the sigmoid of z."""
    return 1 / (1 + np.exp(-z))


def cost_function(theta, X, y, lambda_):
    """
    Compute regularized logistic regression cost.

    Args:
        theta (np.array): Parameter vector.
        X (np.array): Feature matrix.
        y (np.array): Target vector.
        lambda_ (float): Regularization parameter.

    Returns:
        float: Regularized cost.
    """
    m = len(y)
    h = sigmoid(X.dot(theta))
    cost = (-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h))) / m
    reg = (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))
    return cost + reg


def compute_gradient(theta, X, y, lambda_):
    """
    Compute regularized gradient for logistic regression.

    Args:
        theta (np.array): Parameter vector.
        X (np.array): Feature matrix.
        y (np.array): Target vector.
        lambda_ (float): Regularization parameter.

    Returns:
        np.array: Gradient vector.
    """
    m = len(y)
    h = sigmoid(X.dot(theta))
    grad = (X.T.dot(h - y)) / m
    grad[1:] += (lambda_ / m) * theta[1:]
    return grad


def batch_gradient_descent(X, y, theta, learning_rate, n_iterations, lambda_):
    """Perform batch gradient descent optimization."""
    for _ in range(n_iterations):
        grad = compute_gradient(theta, X, y, lambda_)
        theta -= learning_rate * grad
    return theta


def stochastic_gradient_descent(X, y, theta, learning_rate, n_iterations, lambda_):
    """Perform stochastic gradient descent optimization."""
    m = len(y)
    for _ in range(n_iterations):
        for i in range(m):
            rand_index = np.random.randint(m)
            xi = X[rand_index:rand_index + 1]
            yi = y[rand_index:rand_index + 1]
            grad = compute_gradient(theta, xi, yi, lambda_)
            theta -= learning_rate * grad
    return theta


def mini_batch_gradient_descent(X, y, theta, learning_rate, n_iterations, lambda_, batch_size=32):
    """Perform mini-batch gradient descent optimization."""
    m = len(y)
    for _ in range(n_iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for i in range(0, m, batch_size):
            xi = X_shuffled[i:i + batch_size]
            yi = y_shuffled[i:i + batch_size]
            grad = compute_gradient(theta, xi, yi, lambda_)
            theta -= learning_rate * grad
    return theta


def one_vs_all(X, y, num_labels, lambda_, learning_rate=0.00001,
               n_iterations=500, method="batch"):
    """
    Train multiple logistic regression classifiers using one-vs-all strategy.

    Args:
        X (np.array): Feature matrix with bias term.
        y (np.array): Encoded target vector.
        num_labels (int): Number of unique labels.
        lambda_ (float): Regularization parameter.
        learning_rate (float): Learning rate for gradient descent.
        n_iterations (int): Number of iterations for training.
        method (str): Optimization method: 'batch', 'mini-batch' or 'sgd'.

    Returns:
        np.array: Matrix of trained parameters for each classifier.
    """
    m, n = X.shape
    all_theta = np.zeros((num_labels, n))

    if method == "batch":
        training_fn = batch_gradient_descent
    elif method == "mini-batch":
        training_fn = mini_batch_gradient_descent
    else:
        training_fn = stochastic_gradient_descent

    for i in range(num_labels):
        y_i = (y == i).astype(int)
        initial_theta = np.zeros(n)
        all_theta[i] = training_fn(X, y_i, initial_theta,
                                   learning_rate, n_iterations, lambda_)
    return all_theta


def logreg_train(dataset_file, method="batch"):
    """
    Train logistic regression classifiers on dataset.

    Args:
        dataset_file (str): Path to CSV dataset.
        method (str): Optimization method to use ('batch', 'mini-batch', 'sgd').
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
    learning_rate = 0.001
    n_iterations = 1000

    all_theta = one_vs_all(X, y_encoded, num_labels, lambda_,
                          learning_rate, n_iterations, method=method)

    weights_path = os.path.join(output_dir, "weights.csv")
    np.savetxt(weights_path, all_theta, delimiter=',')
    print(f"Model trained successfully using '{method}' method. "
          f"Files 'weights.csv' and 'stats.csv' generated.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python logreg_train.py dataset.csv [method]")
        print("method: 'batch' (default), 'mini-batch', or 'sgd'")
        sys.exit(1)

    dataset_file = sys.argv[1]
    if not os.path.isfile(dataset_file):
        print(f"Error: The dataset file '{dataset_file}' does not exist.")
        sys.exit(1)

    method = sys.argv[2] if len(sys.argv) == 3 else "batch"
    if method not in ("batch", "mini-batch", "sgd"):
        print("Error: method must be 'batch', 'mini-batch' or 'sgd'.")
        sys.exit(1)

    logreg_train(dataset_file, method=method)
