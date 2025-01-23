import pandas as pd
from sklearn.metrics import accuracy_score

def main():
    true_labels_file = "dataset_test.csv"
    predictions_file = "Outputs/prediction/houses.csv"

    true_labels_df = pd.read_csv(true_labels_file)
    predictions_df = pd.read_csv(predictions_file)

    true_labels_df = true_labels_df.rename(columns={"Index": "ID", "Hogwarts House": "True_Label"})
    predictions_df = predictions_df.rename(columns={"Index": "ID", "Hogwarts House": "Predicted_Label"})

    merged_df = pd.merge(true_labels_df, predictions_df, on="ID")

    y_true = merged_df["True_Label"]
    y_pred = merged_df["Predicted_Label"]

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
