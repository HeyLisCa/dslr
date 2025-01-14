import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

colors = {
    "Gryffindor": "#C72C48",  # Bright red
    "Hufflepuff": "#F9E03C",  # Golden yellow
    "Ravenclaw": "#1E3A5F",   # Navy blue
    "Slytherin": "#4C9A2A"    # Grass green
}

# Helper functions
def read_csv(file_path):
    """Read data from a CSV file and return it as a DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: '{file_path}'")
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"Cannot read file: '{file_path}'")

    try:
        return pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file '{file_path}' is empty.")
    except pd.errors.ParserError:
        raise ValueError(f"The file '{file_path}' contains formatting errors.")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred while reading the file: {e}")

def create_scatter(data, class_column):
    """Display scatter plots for numeric columns, grouped by class, and save as an image."""
    if class_column not in data.columns:
        raise ValueError(f"Column '{class_column}' not found in the dataset.")

    numeric_data = [col for col in data.select_dtypes(include=["number"]).columns if col != "Index"]
    if not numeric_data:
        raise ValueError("No numeric columns found in the dataset.")

    grouped = data.groupby(class_column)
    ncols = min(len(numeric_data), 3)
    nrows = (len(numeric_data) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 12)) 
    axes = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]

    for i, feature in enumerate(numeric_data):
        ax = axes[i]
        for house, group in grouped:
            ax.scatter(
                x=group.index,
                y=group[feature],
                marker=".",
                c=colors.get(house, "black"),
                label=house,
                alpha=0.7
            )
        ax.set_title(feature)
        ax.legend()

    for i in range(len(numeric_data), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    output_dir = "Outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "scatter_plot.png")
    fig.savefig(output_path, dpi=300)

# Main function
def main(file_path):
    """Main entry point of the program."""
    try:
        data = read_csv(file_path)

        class_column = "Hogwarts House"
        if class_column not in data.columns:
            raise ValueError(f"'{class_column}' column is missing from the dataset.")

        create_scatter(data, class_column)
        print("scatter_plot.png saved in the 'Outputs' folder.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py dataset.csv")
        sys.exit(1)

    main(sys.argv[1])
