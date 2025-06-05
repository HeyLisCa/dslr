import pandas as pd
from sklearn.metrics import accuracy_score
import sys
import os


def main():
    """
    Compare predicted labels to true labels and print the accuracy score.

    Reads predictions from Outputs/prediction/houses.csv and compares them
    to the true labels provided in the CSV file passed as an argument.
    """
    predictions_file = "Outputs/prediction/houses.csv"
    true_labels_file = sys.argv[1]

    true_labels_df = pd.read_csv(true_labels_file)
    predictions_df = pd.read_csv(predictions_file)

    true_labels_df = true_labels_df.rename(
        columns={"Index": "ID", "Hogwarts House": "True_Label"}
    )
    predictions_df = predictions_df.rename(
        columns={"Index": "ID", "Hogwarts House": "Predicted_Label"}
    )

    merged_df = pd.merge(true_labels_df, predictions_df, on="ID")

    y_true = merged_df["True_Label"]
    y_pred = merged_df["Predicted_Label"]

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compare_prediction.py dataset.csv")
        sys.exit(1)

    test_dataset = sys.argv[1]

    if not os.path.isfile(test_dataset):
        print(f"Error: The dataset file '{test_dataset}' does not exist.")
        sys.exit(1)

    houses_file = "Outputs/prediction/houses.csv"
    if not os.path.isfile(houses_file):
        print(f"Error: The houses file '{houses_file}' does not exist.")
        sys.exit(1)

    main()
