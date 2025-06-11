import csv
import os
import sys
import re
import matplotlib.pyplot as plt


colors = {
    "Gryffindor": "#ff0000",
    "Slytherin": "#15ac00",
    "Ravenclaw": "#003cdd",
    "Hufflepuff": "#e6ea01",
}


class InvalidDatasetError(Exception):
    pass


def read_csv(file_path):
    try:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Error: The file '{file_path}' was not found")

        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=',')
            rows = [row for row in reader]

        if not rows:
            raise InvalidDatasetError(f"Error: The file '{file_path}' is empty")

        header = rows[0]
        if not header or all(cell.strip() == "" for cell in header):
            raise InvalidDatasetError(f"Error: The file '{file_path}' has an invalid header")

        column_count = len(header)
        for i, row in enumerate(rows[1:], start=2):
            if len(row) != column_count:
                raise InvalidDatasetError(f"Error: Column mismatch at line {i}")

        return rows

    except FileNotFoundError as e:
        print(e)
    except InvalidDatasetError as e:
        print(e)
    except Exception as e:
        print(f"Error while loading the file '{file_path}': {e}")
    
    return []


def filter_columns(data):
    number_regex = re.compile(r"-?\d+(\.\d+)?([eE][-+]?\d+)?")
    header = data[0]
    rows = data[1:]
    columns = []

    for i, column in enumerate(header):
        if column.lower() in {"id", "index"}:
            continue
        if all(number_regex.fullmatch(row[i]) or row[i] == '' for row in rows) and not all(row[i] == '' for row in rows):
            columns.append(column)
    return columns


def extract_house_data(data, columns):
    header = data[0]
    if "Hogwarts House" not in header:
        raise InvalidDatasetError("Error: 'Hogwarts House' column is missing from the dataset")
    house_idx = header.index("Hogwarts House")

    if all((len(row) <= house_idx or not row[house_idx].strip()) for row in data[1:]):
        raise InvalidDatasetError("Error: All values in 'Hogwarts House' column are empty")

    col_indices = [i for i, col in enumerate(header) if col in columns]
    house_data = {house: {col: [] for col in columns} for house in colors.keys()}

    for row in data[1:]:
        if len(row) <= house_idx or not row[house_idx]:
            continue
        house = row[house_idx]
        if house in house_data:
            for i, column in enumerate(columns):
                if row[col_indices[i]] != '':
                    house_data[house][column].append(float(row[col_indices[i]]))
    
    return house_data


def display_histograms(data):
    columns = filter_columns(data)
    house_data = extract_house_data(data, columns)
    n_cols = len(columns)
    n_rows = (n_cols // 4) + (1 if n_cols % 4 else 0)
    fig, axes = plt.subplots(n_rows, 4, figsize=(13, 11))
    axes = axes.flatten()
    
    for i, column in enumerate(columns):
        for house, house_values in house_data.items():
            values = house_values[column]
            if values:
                axes[i].hist(values, bins=20, color=colors.get(house, 'gray'), alpha=0.4, edgecolor='black', label=house)
        
        axes[i].set_xlabel("Notes")
        axes[i].set_ylabel("Frequency")
        axes[i].set_title(f"{column}")
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].legend()
    
    for i in range(n_cols, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    output_dir = "Outputs/visualization"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "histogram_plot.png")
    fig.savefig(output_path, dpi=300)
    print(f"Histogram plot saved to {output_path}")
    plt.close()


def most_homogeneous_course(data):
    columns = filter_columns(data)
    house_data = extract_house_data(data, columns)
    
    course_variances = {}
    for column in columns:
        house_means = []
        for house in colors.keys():
            if house_data[house][column]:
                house_means.append(sum(house_data[house][column]) / len(house_data[house][column]))

        if len(house_means) == 4:
            mean = sum(house_means) / len(house_means)
            variance = sum((x - mean) ** 2 for x in house_means) / len(house_means)
            course_variances[column] = variance ** 0.5
    
    most_homogeneous_column = None
    smallest_variance = float('inf')
    for column, variance in course_variances.items():
        if variance < smallest_variance:
            smallest_variance = variance
            most_homogeneous_column = column
    
    return most_homogeneous_column


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset.csv>")
    else:
        data = read_csv(sys.argv[1])
        if data:
            try:
                display_histograms(data)
                result = most_homogeneous_course(data)
                print(f"The most homogeneous course is: {result}")
            except InvalidDatasetError as e:
                print(e)
        else:
            print("No numeric columns found in the dataset")