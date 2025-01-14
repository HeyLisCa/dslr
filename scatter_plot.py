import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

colors = {
    "Gryffindor": "#C72C48",  # Bright red
    "Hufflepuff": "#F9E03C",  # Golden yellow
    "Ravenclaw": "#1E3A5F",   # Navy blue
    "Slytherin": "#4C9A2A"    # Grass green
}

def read_csv(file_path):
    """Reads data from a CSV file and returns it as a DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: '{file_path}'")
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"Cannot read file: '{file_path}'")

    try:
        return pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file '{file_path}' is empty.")
    except pd.errors.ParserError:
        raise ValueError(f"The file '{file_path}' contains parsing errors.")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred while reading the file: {e}")

def show_scatter(df, class_index):
    """Displays a scatter plot with different colors for each class."""
    df.set_index("Index", drop=True, inplace=True)
    ncols = int((len(df.columns) / 3) + 0.5)
    grouped = df.groupby(class_index)
    fig, dim_axs = plt.subplots(nrows=3, ncols=ncols, figsize=(16, 8))
    axs = dim_axs.flatten()
    i = 0
    for feature, values in df.items():
        if feature == class_index:
            continue
        for house, y in grouped:
            axs[i].scatter(x=y[feature].index, y=y[feature], marker=".", c=colors[house], label=house)
            axs[i].set_title(feature)
            axs[i].legend()
        i += 1
    while i < len(axs):
        axs[i].axis("off")
        i += 1
    plt.show()

def main(file_path):
    """Main entry point of the program."""
    try:
        df = read_csv(file_path)
        non_number = ["First Name", "Last Name", "Birthday", "Best Hand"]
        df.drop(columns=non_number, inplace=True)
        show_scatter(df, "Hogwarts House")
        print("Similar features between all houses should have the same groups for the same colors.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py dataset_train.csv")
        sys.exit(1)
    
    main(sys.argv[1])
