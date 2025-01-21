import sys
import pandas as pd
import numpy as np
import os

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, weights):
    probabilities = sigmoid(np.dot(X, weights.T))
    return np.argmax(probabilities, axis=1)

def logreg_predict(test_dataset, weights_file, stats_file):
    data = pd.read_csv(test_dataset)
    
    selected_features = ["Transfiguration", "History of Magic", "Charms", "Ancient Runes", "Defense Against the Dark Arts", "Herbology", "Potions", "Flying", "Astronomy"]
    
    missing_features = set(selected_features) - set(data.columns)
    if missing_features:
        raise ValueError(f"The test data is missing the following columns: {missing_features}")
    
    data[selected_features] = data[selected_features].fillna(data[selected_features].median())
    
    X = np.array(data[selected_features], dtype=float)
    
    stats = pd.read_csv(stats_file)
    X_mean = stats["Mean"].to_numpy()
    X_std = stats["Std"].to_numpy()
    
    X = (X - X_mean) / X_std
    X = np.clip(X, -5, 5)
    
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    
    weights = np.loadtxt(weights_file, delimiter=",")
    
    if X.shape[1] != weights.shape[1]:
        raise ValueError(f"Dimension mismatch: X has {X.shape[1]} columns, but the weights have {weights.shape[1]} columns.")
    
    predictions = predict(X, weights)
    
    unique_labels = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    predicted_houses = [unique_labels[p] for p in predictions]
    
    results = pd.DataFrame({
        "Index": np.arange(len(predicted_houses)),
        "Hogwarts House": predicted_houses
    })
    output_dir = "Outputs/prediction"
    results_path = os.path.join(output_dir, "houses.csv")
    results.to_csv(results_path, index=False)
    print("Predictions saved in the 'Outputs' folder as 'houses.csv'.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python logreg_predict.py dataset_test.csv weights.csv stats.csv")
        sys.exit(1)
    
    test_dataset = sys.argv[1]
    weights_file = sys.argv[2]
    stats_file = sys.argv[3]
    logreg_predict(test_dataset, weights_file, stats_file)
