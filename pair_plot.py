import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

colors = {
    "Gryffindor": "#C72C48",  # Bright red
    "Hufflepuff": "#F9E03C",  # Golden yellow
    "Ravenclaw": "#1E3A5F",   # Navy blue
    "Slytherin": "#4C9A2A"    # Grass green
}

# Helper functions
def read_csv(file_path):
    """Reads a CSV file and returns it as a pandas DataFrame."""
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

def truncate_name(name, max_length=15):
    """Truncates a name to fit within the specified max length, adding '...' if necessary."""
    return name[:max_length - 3] + "..." if len(name) > max_length else name

def create_pair_plot(data):
    """
    Creates a pair plot for numeric columns in the dataset, grouped by Hogwarts House.
    Saves the plot as 'pair_plot.png' in the 'Outputs' folder.
    """
    data = data.drop(columns=["Index"], errors="ignore")
    numeric_data = data.select_dtypes(include=[float, int])
    if numeric_data.empty:
        raise ValueError("No numeric columns found in the dataset.")
    
    grouped = data.groupby("Hogwarts House")
    length = len(numeric_data.columns)

    fig, axes = plt.subplots(length, length, figsize=(16, 9), gridspec_kw={"wspace": 0, "hspace": 0})
    axes = axes.flatten()

    for i, feature in enumerate(numeric_data.columns):
        for j, against_feature in enumerate(numeric_data.columns):
            ax = axes[i * length + j]
            if feature == against_feature:
                grouped[feature].hist(alpha=0.5, ax=ax, bins=10)
            else:
                for house, group in grouped:
                    ax.scatter(group[against_feature], group[feature], c=colors[house], alpha=0.5, s=10, marker=".")
            if i == length - 1:
                ax.set_xlabel(truncate_name(against_feature), fontsize=6)
            if j == 0:
                ax.set_ylabel(truncate_name(feature), fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    output_dir = "Outputs/images"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pair_plot.png")
    fig.savefig(output_path, dpi=300)

# Main function
def main(file_path):
    """Main entry point of the program."""
    try:
        data = read_csv(file_path)
        create_pair_plot(data)
        print("pair_plot.png saved in the 'Outputs' folder.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pair_plot.py dataset.csv")
        sys.exit(1)

    main(sys.argv[1])
