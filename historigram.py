import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import math

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


def create_histogram(data, courses):
    """Creates a histogram of the given data."""
    num_courses = len(courses)

    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    axes = axes.flatten()

    for i, course in enumerate(courses):
        for house in ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']:
            house_scores = data[data['Hogwarts House'] == house][course]
            axes[i].hist(house_scores.dropna(), bins=10, alpha=0.5, label=house, color=colors[house])

        axes[i].set_title(course)
        axes[i].set_xlabel('Scores')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()

    for i in range(num_courses, len(axes)):
        axes[i].remove()

    plt.tight_layout()
    plt.show()

def calculate_variance(data, courses):
    """Calculates the variance of the given data."""
    variances = {}
    for course in courses:
        house_variances = {}
        
        for house in ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']:
            house_scores = data[data['Hogwarts House'] == house][course].dropna()
            house_variances[house] = house_scores.var()
        
        mean_variance = sum(house_variances.values()) / len(house_variances)
        variances[course] = mean_variance

    homogeneous_course = min(variances, key=variances.get)
    print(f"The course with the most homogeneous score distribution is: {homogeneous_course}")

def main(file_path):
    try:
        courses = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']
        data = read_csv(file_path)
        calculate_variance(data, courses)
        create_histogram(data, courses)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python historigram.py dataset_train.csv")
        sys.exit(1)

    main(sys.argv[1])
