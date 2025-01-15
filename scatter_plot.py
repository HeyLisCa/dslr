import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define house colors for plotting
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

def create_scatter(data):
    """
    Creates scatter plots for numeric columns in the dataset, grouped by Hogwarts House.
    Saves the plot as 'scatter_plot.png' in the 'Outputs' folder.
    """
    numeric_data = [col for col in data.select_dtypes(include=["number"]).columns if col != "Index"]
    if not numeric_data:
        raise ValueError("No numeric columns found in the dataset.")

    grouped = data.groupby("Hogwarts House")
    ncols = min(len(numeric_data), 5)
    nrows = (len(numeric_data) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 9))
    axes = axes.flatten()

    for i, feature in enumerate(numeric_data):
        for house, group in grouped:
            axes[i].scatter(group.index, group[feature], c=colors[house], alpha=0.7, label=house, marker=".")
        axes[i].set_title(feature, fontsize=10)
        axes[i].set_xlabel("Index", fontsize=9)
        axes[i].set_ylabel("Value", fontsize=9)
        axes[i].legend(fontsize=8)

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
        create_scatter(data)
        print("scatter_plot.png saved in the 'Outputs' folder.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py dataset.csv")
        sys.exit(1)

    main(sys.argv[1])
