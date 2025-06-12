import pandas as pd
from sklearn.metrics import accuracy_score
import sys
import os
import signal


# Handle the SIGINT signal
signal.signal(signal.SIGINT, signal.SIG_DFL)


def main():
    """
    Compare predicted labels to true labels and print the accuracy score.

    Reads predictions from Outputs/prediction/houses.csv and compares them
    to the true labels provided in the CSV file passed as an argument.
    """
    predictions_file = "Outputs/prediction/houses.csv"
    true_labels_file = sys.argv[1]

    try:
        true_labels_df = pd.read_csv(true_labels_file)
        predictions_df = pd.read_csv(predictions_file)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        sys.exit(1)

    required_columns = {"Index", "Hogwarts House"}
    for df, name in [(true_labels_df, "true labels"), (predictions_df, "predictions")]:
        if not required_columns.issubset(df.columns):
            print(f"Error: The {name} file is missing required columns.")
            sys.exit(1)

    true_labels_df = true_labels_df.rename(
        columns={"Index": "ID", "Hogwarts House": "True_Label"}
    )
    predictions_df = predictions_df.rename(
        columns={"Index": "ID", "Hogwarts House": "Predicted_Label"}
    )

    if true_labels_df["ID"].duplicated().any() or predictions_df["ID"].duplicated().any():
        print("Error: Duplicate IDs found in one of the files.")
        sys.exit(1)

    merged_df = pd.merge(true_labels_df, predictions_df, on="ID", how="inner")

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
