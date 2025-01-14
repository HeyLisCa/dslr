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

def create_histogram(data):
    """Creates a histogram of the given data."""
    numeric_data = [col for col in data.select_dtypes(include=['number']).columns.tolist() 
                       if col not in ['Hogwarts House', 'Index']]

    fig, axes = plt.subplots(3, 5, figsize=(16, 9))
    axes = axes.flatten()

    for i, course in enumerate(numeric_data):
        for house in ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']:
            house_scores = data[data['Hogwarts House'] == house][course]
            axes[i].hist(house_scores.dropna(), bins=10, alpha=0.5, label=house, color=colors[house])

        axes[i].set_title(course, fontsize=10)
        axes[i].set_xlabel('Scores', fontsize=9)
        axes[i].set_ylabel('Frequency', fontsize=9)
        axes[i].legend(fontsize=8)

    for i in range(len(numeric_data), len(axes)):
        axes[i].remove()

    plt.tight_layout()

    output_dir = "Outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "histogram.png")
    fig.savefig(output_path, dpi=300)

def calculate_variance(data):
    """Calculates the variance of the given data."""
    numeric_data = [col for col in data.select_dtypes(include=['number']).columns.tolist() 
                       if col not in ['Hogwarts House', 'Index']]

    variances = {}
    for course in numeric_data:
        house_variances = {house: data[data['Hogwarts House'] == house][course].dropna().var() 
                           for house in ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']}
        
        mean_variance = sum(house_variances.values()) / len(house_variances)
        variances[course] = mean_variance

    homogeneous_course = min(variances, key=variances.get)
    print(f"The course with the most homogeneous score distribution is: {homogeneous_course}")

# Main function
def main(file_path):
    try:
        data = read_csv(file_path)
        calculate_variance(data)
        create_histogram(data)
        print("historigram.png saved in the 'Outputs' folder.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python histogram.py dataset.csv")
        sys.exit(1)

    main(sys.argv[1])
