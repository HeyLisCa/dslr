import sys
import os
import csv
import re
import matplotlib.pyplot as plt
import numpy as np
import signal


# Handle the SIGINT signal
signal.signal(signal.SIGINT, signal.SIG_DFL)


COLORS = {
    "Gryffindor": "#f32100",   # red
    "Slytherin": "#15ac00",    # green
    "Ravenclaw": "#003cdd",    # blue
    "Hufflepuff": "#e6ea01",   # yellow
}


class InvalidDatasetError(Exception):
    """Custom exception raised when the dataset is invalid."""
    pass


def read_csv(file_path):
    try:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")

        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=',')
            rows = [row for row in reader]

        if not rows:
            raise InvalidDatasetError(f"Error: The file '{file_path}' is empty.")

        header = rows[0]
        if not header or all(cell.strip() == "" for cell in header):
            raise InvalidDatasetError(f"Error: The file '{file_path}' has an invalid header.")

        column_count = len(header)
        valid_rows = [header]

        numeric_columns = ['Muggle Studies', 'Care of Magical Creatures', 'Charms', 'Defense Against the Dark Arts',
                           'Divination', 'Herbology', 'History of Magic', 'Potions', 'Transfiguration', 'Flying']

        numeric_indices = [header.index(col) for col in numeric_columns if col in header]

        for i, row in enumerate(rows[1:], start=2):
            if len(row) != column_count:
                print(f"Warning: Skipping malformed row {i} in file '{file_path}'.")
                continue

            valid_row = True
            for idx in numeric_indices:
                try:
                    float(row[idx].strip())
                except ValueError:
                    valid_row = False
                    break

            if valid_row:
                valid_rows.append(row)

        if len(valid_rows) <= 1:
            raise InvalidDatasetError("Error: No valid data rows found in the dataset.")

        return valid_rows

    except (FileNotFoundError, InvalidDatasetError, Exception) as e:
        print(e)
        return []


def filter_columns(data):
    """
    Filter numeric columns from the dataset.

    Args:
        data (list): The dataset as a list of rows.

    Returns:
        list: List of numeric column names.
    """
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


def extract_column_data(data):
    """
    Extract and convert numeric column data.

    Args:
        data (list): The dataset as a list of rows.

    Returns:
        dict: A dictionary with column names as keys and lists of floats or None as values.
    """
    header = data[0]
    numeric_columns = filter_columns(data)
    col_indices = [i for i, col in enumerate(header) if col in numeric_columns]

    column_data = {col: [] for col in numeric_columns}

    for row in data[1:]:
        for i, col in zip(col_indices, numeric_columns):
            value = row[i]
            column_data[col].append(float(value) if value != '' else None)

    return column_data


def extract_house_data(data, columns):
    """
    Organize data by house and column.

    Args:
        data (list): The dataset as a list of rows.
        columns (list): List of numeric column names.

    Returns:
        dict: Nested dictionary with house names as keys and column data as inner dictionaries.
    """
    header = data[0]
    col_indices = [i for i, col in enumerate(header) if col in columns]
    house_data = {house: {col: [] for col in columns} for house in COLORS.keys()}

    for row in data[1:]:
        house = row[1]
        if house in house_data:
            for i, column in enumerate(columns):
                if row[col_indices[i]] != '':
                    house_data[house][column].append(float(row[col_indices[i]]))

    return house_data


def create_pair_plot(data):
    """
    Display a pair plot of numeric data grouped by house.

    Args:
        data (list): The dataset as a list of rows.
    """
    column_data = extract_column_data(data)
    columns = list(column_data.keys())

    if len(columns) < 2:
        print("Not enough numeric columns to create a pair plot")
        return

    house_data = extract_house_data(data, columns)
    n = len(columns)

    fig, axes = plt.subplots(n, n, figsize=(13, 11))

    for i, col_x in enumerate(columns):
        for j, col_y in enumerate(columns):
            ax = axes[i, j]

            if i == j:
                for house, house_values in house_data.items():
                    values = house_values[col_x]
                    if values:
                        ax.hist(
                            values, bins=15,
                            color=COLORS.get(house, 'gray'),
                            alpha=0.4, label=house,
                            edgecolor='black'
                        )
            else:
                for house, house_values in house_data.items():
                    x_values = house_values[col_x]
                    y_values = house_values[col_y]

                    mask = [
                        x is not None and y is not None
                        for x, y in zip(x_values, y_values)
                    ]
                    x_values = [x for x, m in zip(x_values, mask) if m]
                    y_values = [y for y, m in zip(y_values, mask) if m]

                    if x_values and y_values:
                        ax.scatter(
                            x_values, y_values,
                            alpha=0.5, s=3, label=house,
                            color=COLORS.get(house, 'gray')
                        )

            ax.set_xticks([])
            ax.set_yticks([])

            if j == 0:
                ax.set_ylabel(col_x[:16], fontsize=7)
            if i == n - 1:
                ax.set_xlabel(col_y[:16], fontsize=7)

            ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    output_dir = "Outputs/visualization"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pair_plot.png")
    fig.savefig(output_path, dpi=300)
    print(f"Pair plot saved to {output_path}")
    plt.close()


def discriminative_features(data, top_n=5):
    columns = filter_columns(data)
    house_data = extract_house_data(data, columns)
    scores = {}

    for col in columns:
        means = []
        variances = []
        for house in COLORS.keys():
            values = house_data[house][col]
            if values:
                means.append(np.mean(values))
                variances.append(np.var(values))
        if len(means) > 1 and all(v > 0 for v in variances):
            score = np.var(means) / (np.mean(variances) + 1e-8)
            scores[col] = score
        else:
            scores[col] = 0

    sorted_cols = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print("\nBest features (combining histogram and scatter plot separation):")
    for col, score in sorted_cols[:top_n]:
        print(f"{col}: separation score = {score:.3f}")


def main():
    """
    Entry point of the program.
    Handles argument validation, file loading and plotting.
    """
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset.csv>")
        sys.exit(1)

    data = read_csv(sys.argv[1])
    if not data:
        print("No numeric columns found in the dataset")
        sys.exit(1)

    header = data[0]
    if "Hogwarts House" not in header:
        print("Error: 'Hogwarts House' column is missing from the dataset")
        sys.exit(1)
    house_idx = header.index("Hogwarts House")
    if all((len(row) <= house_idx or not row[house_idx].strip()) for row in data[1:]):
        print("Error: All values in 'Hogwarts House' column are empty")
        sys.exit(1)

    create_pair_plot(data)
    discriminative_features(data)


if __name__ == "__main__":
    main()
