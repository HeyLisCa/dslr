import csv
import os
import re
import sys
import matplotlib.pyplot as plt
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
    """Exception raised for invalid dataset format or content."""
    pass


def read_csv(file_path):
    """
    Read and validate a CSV file.

    Args:
        file_path (str): The path to the CSV file to read.

    Returns:
        list: A list of rows read from the CSV file, or an empty list if an error occurs.
    """
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
        for i, row in enumerate(rows[1:], start=2):
            if len(row) != column_count:
                raise InvalidDatasetError(f"Error: Column mismatch at line {i}.")

        return rows

    except (FileNotFoundError, InvalidDatasetError, Exception) as e:
        print(e)
        return []


def filter_columns(data):
    """
    Identify numeric columns in the dataset.

    Args:
        data (list): The dataset.

    Returns:
        list: A list of column names containing numeric data.
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
    Extract numeric data from the dataset indexed by column name.

    Args:
        data (list): The dataset.

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
    Extract data related to each house for selected columns.

    Args:
        data (list): The dataset.
        columns (list): List of columns to extract.

    Returns:
        dict: Nested dict of house -> column -> list of floats.
    """
    header = data[0]
    col_indices = [i for i, col in enumerate(header) if col in columns]
    house_data = {house: {col: [] for col in columns} for house in COLORS}

    for row in data[1:]:
        house = row[1]
        if house in house_data:
            for i, column in enumerate(columns):
                if row[col_indices[i]] != '':
                    house_data[house][column].append(float(row[col_indices[i]]))

    return house_data


def display_scatter_plots(data, subject):
    """
    Display scatter plots comparing a subject against other numeric columns.

    Args:
        data (list): The dataset.
        subject (str): The main subject column name.
    """
    column_data = extract_column_data(data)
    columns = list(column_data.keys())

    if subject not in columns:
        print(f"Error: '{subject}' is not a valid subject.")
        return

    n_cols = 4
    relevant_columns = [col for col in columns if col != subject]
    n_rows = (len(relevant_columns) + n_cols - 1) // n_cols
    house_data = extract_house_data(data, columns)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 11))
    axes = axes.flatten()

    for i, col_y in enumerate(relevant_columns):
        ax = axes[i]

        for house, house_values in house_data.items():
            x_values = house_values[subject]
            y_values = house_values[col_y]

            mask = [x is not None and y is not None for x, y in zip(x_values, y_values)]
            x_values = [x for x, m in zip(x_values, mask) if m]
            y_values = [y for y, m in zip(y_values, mask) if m]

            if x_values and y_values:
                ax.scatter(x_values, y_values, alpha=0.8, s=5, label=house, color=COLORS.get(house, 'gray'))

        ax.set_xlabel(subject, fontsize=10)
        ax.set_ylabel(col_y, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Comparing with {subject}", fontsize=12)
    plt.tight_layout()
    plt.show()


def get_valid_subjects(data):
    """
    Return valid numeric subjects.

    Args:
        data (list): The dataset.

    Returns:
        list: List of valid subject names.
    """
    return filter_columns(data)


def user_input_subject(valid_subjects):
    """
    Prompt the user to select a subject.

    Args:
        valid_subjects (list): List of valid subjects.

    Returns:
        str: Selected subject, "Similar Features", or "Exit".
    """
    print("Available subjects:")
    for idx, subject in enumerate(valid_subjects, 1):
        print(f"{idx}. {subject}")
    print(f"{len(valid_subjects) + 1}. Similar Features")
    print(f"{len(valid_subjects) + 2}. Exit")

    while True:
        try:
            choice = int(input(f"Choose a subject (1-{len(valid_subjects) + 2}): "))
            if 1 <= choice <= len(valid_subjects):
                return valid_subjects[choice - 1]
            elif choice == len(valid_subjects) + 1:
                return "Similar Features"
            elif choice == len(valid_subjects) + 2:
                return "Exit"
            else:
                print("Invalid choice, please try again.")
        except ValueError:
            print("Invalid input, please enter a number.")


def pearson_correlation(x, y):
    """
    Compute the Pearson correlation coefficient.

    Args:
        x (list): Values for variable x.
        y (list): Values for variable y.

    Returns:
        float: Pearson correlation coefficient.
    """
    n = len(x)
    if n == 0:
        return 0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denom_x = sum((x[i] - mean_x) ** 2 for i in range(n)) ** 0.5
    denom_y = sum((y[i] - mean_y) ** 2 for i in range(n)) ** 0.5

    return num / (denom_x * denom_y) if denom_x and denom_y else 0


def find_strongest_correlations(result):
    """
    Find the strongest correlation in the dataset.

    Args:
        result (dict): Correlation results.

    Returns:
        tuple: Pair of subjects with strongest correlation.
    """
    correlation_scores = []

    for subject1 in result:
        for subject2 in result[subject1]:
            correlations = result[subject1][subject2].values()
            avg_corr = sum(abs(corr) for corr in correlations) / len(correlations)
            correlation_scores.append(((subject1, subject2), avg_corr))

    correlation_scores.sort(key=lambda x: x[1], reverse=True)
    return correlation_scores[0]


def check_similar_features(data):
    """
    Compute feature correlations and print the strongest one.

    Args:
        data (list): The dataset.
    """
    column_data = extract_column_data(data)
    columns = list(column_data.keys())
    house_data = extract_house_data(data, columns)

    result = {
        col1: {
            col2: {house: [] for house in COLORS}
            for col2 in columns if col2 != col1
        }
        for col1 in columns
    }

    for col1 in columns:
        for col2 in columns:
            if col1 == col2:
                continue
            for house, values in house_data.items():
                x = values[col1]
                y = values[col2]
                mask = [x_ is not None and y_ is not None for x_, y_ in zip(x, y)]
                x_filtered = [x_ for x_, m in zip(x, mask) if m]
                y_filtered = [y_ for y_, m in zip(y, mask) if m]

                correlation = pearson_correlation(x_filtered, y_filtered)
                result[col1][col2][house] = correlation

    best_pair = find_strongest_correlations(result)
    print(f"The strongest correlation is between {best_pair[0][0]} and {best_pair[0][1]}")


def main():
    """
    Entry point of the program.
    Handles CSV loading and user interaction.
    """
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py <dataset.csv>")
        sys.exit(1)

    data = read_csv(sys.argv[1])
    if not data:
        print("No valid data loaded.")
        sys.exit(1)

    header = data[0]
    if "Hogwarts House" not in header:
        print("Error: 'Hogwarts House' column is missing from the dataset")
        sys.exit(1)
    house_idx = header.index("Hogwarts House")
    if all((len(row) <= house_idx or not row[house_idx].strip()) for row in data[1:]):
        print("Error: All values in 'Hogwarts House' column are empty")
        sys.exit(1)

    valid_subjects = get_valid_subjects(data)
    if not valid_subjects:
        print("No valid numeric subjects found in the dataset.")
        sys.exit(1)

    while True:
        subject = user_input_subject(valid_subjects)
        if subject == "Exit":
            print("Exiting program.")
            sys.exit(0)
        elif subject == "Similar Features":
            check_similar_features(data)
        else:
            display_scatter_plots(data, subject)


if __name__ == "__main__":
    main()
