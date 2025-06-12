import numpy as np
import pandas as pd
import sys
import os


def sigmoid(z):
    """Compute the sigmoid of z, safe against overflow."""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-z))


def cost_function(theta, X, y, lambda_):
    """Compute the regularized logistic regression cost function."""
    m = len(y)
    h = sigmoid(X.dot(theta))
    h = np.clip(h, 1e-15, 1 - 1e-15)
    cost = (-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h))) / m
    reg = (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))
    return cost + reg


def compute_gradient(theta, X, y, lambda_):
    """Compute the regularized gradient for logistic regression."""
    m = len(y)
    h = sigmoid(X.dot(theta))
    grad = (X.T.dot(h - y)) / m
    grad[1:] += (lambda_ / m) * theta[1:]
    return grad


def batch_gradient_descent(X, y, theta, learning_rate, n_iterations, lambda_):
    """Perform batch gradient descent."""
    for _ in range(n_iterations):
        grad = compute_gradient(theta, X, y, lambda_)
        theta -= learning_rate * grad
        if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
            print("Error: theta contains NaN or Inf during training.")
            sys.exit(1)
    return theta


def stochastic_gradient_descent(X, y, theta, learning_rate, n_iterations, lambda_):
    """Perform stochastic gradient descent."""
    m = len(y)
    for _ in range(n_iterations):
        for i in range(m):
            rand_index = np.random.randint(m)
            xi = X[rand_index:rand_index + 1]
            yi = y[rand_index:rand_index + 1]
            grad = compute_gradient(theta, xi, yi, lambda_)
            theta -= learning_rate * grad
            if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
                print("Error: theta contains NaN or Inf during training.")
                sys.exit(1)
    return theta


def mini_batch_gradient_descent(X, y, theta, learning_rate,
                                n_iterations, lambda_, batch_size=32):
    """Perform mini-batch gradient descent."""
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
            if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
                print("Error: theta contains NaN or Inf during training.")
                sys.exit(1)
    return theta


def one_vs_all(X, y, num_labels, lambda_, learning_rate=0.00001,
               n_iterations=500, method="batch"):
    """
    Train one-vs-all logistic regression classifiers.

    Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): Label vector.
        num_labels (int): Number of unique classes.
        lambda_ (float): Regularization parameter.
        learning_rate (float): Learning rate for gradient descent.
        n_iterations (int): Number of iterations.
        method (str): Descent method: 'batch', 'mini-batch', or 'sgd'.

    Returns:
        ndarray: Parameters for each classifier.
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
    Train logistic regression model using one-vs-all strategy.

    Parameters:
        dataset_file (str): Path to the dataset CSV file.
        method (str): Gradient descent method to use.
    """
    try:
        data = pd.read_csv(dataset_file)
    except Exception as e:
        print(f"Error reading dataset file '{dataset_file}': {e}")
        sys.exit(1)

    output_dir = "Outputs/training"
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory '{output_dir}': {e}")
        sys.exit(1)

    selected_features = [
        "Charms",
        "Ancient Runes",
        "Defense Against the Dark Arts",
        "Herbology",
        "Flying",
        "Astronomy"
    ]

    missing_cols = [f for f in selected_features + ["Hogwarts House"]
                    if f not in data.columns]
    if missing_cols:
        print(f"Error: Missing columns in dataset: {missing_cols}")
        sys.exit(1)

    data = data.dropna(subset=selected_features + ["Hogwarts House"])

    if data.empty:
        print("Error: After removing incomplete rows, no valid data remains.")
        sys.exit(1)

    def is_valid_row(row):
        try:
            _ = row.astype(float)
            return True
        except ValueError:
            return False

    valid_mask = data[selected_features].apply(is_valid_row, axis=1)
    data = data[valid_mask]

    if data.empty:
        print("Error: No valid numeric data in selected features after filtering.")
        sys.exit(1)

    try:
        X = data[selected_features].astype(float).values
    except Exception as e:
        print(f"Error converting features to float: {e}")
        sys.exit(1)

    y = data["Hogwarts House"].values

    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)

    zero_std_mask = (X_std == 0)
    if np.any(zero_std_mask):
        print("Warning: Some features have zero std deviation. Adjusting to 1.")
        X_std[zero_std_mask] = 1.0

    X = (X - X_mean) / X_std
    X = np.clip(X, -5, 5)

    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("Error: NaN or Inf values in features after normalization.")
        sys.exit(1)

    stats = pd.DataFrame({
        'Feature': selected_features,
        'Mean': X_mean,
        'Std': X_std
    })

    stats_path = os.path.join(output_dir, "stats.csv")
    try:
        stats.to_csv(stats_path, index=False)
    except Exception as e:
        print(f"Error saving stats to '{stats_path}': {e}")
        sys.exit(1)

    unique_labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    try:
        y_encoded = np.array([label_map[label] for label in y])
    except KeyError as e:
        print(f"Error encoding labels: {e}")
        sys.exit(1)

    X = np.hstack([np.ones((X.shape[0], 1)), X])

    lambda_ = 1.0
    num_labels = len(unique_labels)
    learning_rate = 0.001
    n_iterations = 1000

    all_theta = one_vs_all(
        X, y_encoded, num_labels, lambda_,
        learning_rate, n_iterations, method=method
    )

    weights_path = os.path.join(output_dir, "weights.csv")
    try:
        np.savetxt(weights_path, all_theta, delimiter=',')
    except Exception as e:
        print(f"Error saving weights to '{weights_path}': {e}")
        sys.exit(1)

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
