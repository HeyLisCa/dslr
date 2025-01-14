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
    Creates a pair plot of the numeric features of the given data,
    grouped by Hogwarts House, and saves it to the 'Outputs' folder.
    """
    data = data.drop(columns=['Index'], errors='ignore')
    numeric_data = data.select_dtypes(include=[float, int]).drop(columns=['Hogwarts House'], errors='ignore')

    length = len(numeric_data.columns)
    last_row = length * (length - 1)
    grouped = data.groupby("Hogwarts House")

    fig, dim_axs = plt.subplots(nrows=length, ncols=length, figsize=(20, 16), gridspec_kw={"wspace": 0, "hspace": 0})
    axs = dim_axs.flatten()

    i = 0
    for feature in numeric_data.columns:
        for against_feature in numeric_data.columns:
            truncated_feature = truncate_name(feature)
            truncated_against_feature = truncate_name(against_feature)
            
            if (i % length) == 0:
                axs[i].set_ylabel(f"{truncated_feature}", fontsize=9)
            if i >= last_row:
                axs[i].set_xlabel(f"{truncated_against_feature}", fontsize=9)

            axs[i].set_yticklabels([])
            axs[i].set_yticks([])
            axs[i].set_xticklabels([])
            axs[i].set_xticks([])
            axs[i].grid(False)

            if feature == against_feature:
                grouped[feature].hist(alpha=0.5, ax=axs[i], bins=10)
            else:
                for house, y in grouped:
                    axs[i].scatter(x=y[against_feature], y=y[feature], marker=".", c=colors[house], alpha=0.5, s=10)
            i += 1

    output_dir = "Outputs"
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
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pair_plot.py dataset.csv")
        sys.exit(1)

    main(sys.argv[1])
