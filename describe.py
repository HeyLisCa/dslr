import csv
import os
import re
import sys


class InvalidDatasetError(Exception):
    """Exception raised when the dataset is invalid."""
    pass


def read_csv(file_path):
    """Read a CSV file and return its rows as a list of lists.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        list: A list of rows, where each row is a list of string values.
    """
    try:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Error: The file '{file_path}' was not found")

        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = list(reader)

        if not rows:
            raise InvalidDatasetError(f"Error: The file '{file_path}' is empty")

        header = rows[0]
        if not header or all(cell.strip() == '' for cell in header):
            raise InvalidDatasetError(f"Error: The file '{file_path}' has an invalid header")

        column_count = len(header)
        for i, row in enumerate(rows[1:], start=2):
            if len(row) != column_count:
                raise InvalidDatasetError(f"Error: Column mismatch at line {i}")

        return rows

    except (FileNotFoundError, InvalidDatasetError) as e:
        print(e)
    except Exception as e:
        print(f"Error while loading the file '{file_path}': {e}")

    return []


def filter_columns(data):
    """Filter and return numeric columns excluding 'id' and 'index'.

    Args:
        data (list): Dataset rows including header.

    Returns:
        list: List of names of numeric columns.
    """
    number_regex = re.compile(r"-?\d+(\.\d+)?([eE][-+]?\d+)?")
    header = data[0]
    rows = data[1:]
    columns = []

    for i, column in enumerate(header):
        if column.lower() in {'id', 'index'}:
            continue
        if all(number_regex.fullmatch(row[i]) or row[i] == '' for row in rows) and not all(row[i] == '' for row in rows):
            columns.append(column)

    return columns


def calculate_count(data, columns):
    """Calculate the count of non-empty entries per column.

    Args:
        data (list): Dataset rows including header.
        columns (list): Column names to compute the count for.

    Returns:
        list: List of counts for each column.
    """
    return [sum(1 for row in data[1:] if row[i] != '') for i in range(len(columns))]


def calculate_mean(data, columns):
    """Calculate the mean of each numeric column.

    Args:
        data (list): Dataset rows including header.
        columns (list): Numeric column names.

    Returns:
        list: List of mean values for each column.
    """
    means = []
    for i, _ in enumerate(columns):
        valid_values = [float(row[i]) for row in data[1:] if row[i] != '']
        means.append(sum(valid_values) / len(valid_values))
    return means


def calculate_std(data, columns, means):
    """Calculate standard deviation for each numeric column.

    Args:
        data (list): Dataset rows including header.
        columns (list): Numeric column names.
        means (list): Precomputed mean values.

    Returns:
        list: Standard deviations.
    """
    stds = []
    for i, mean in enumerate(means):
        valid_values = [float(row[i]) for row in data[1:] if row[i] != '']
        variance = sum((val - mean) ** 2 for val in valid_values) / len(valid_values)
        stds.append(variance ** 0.5)
    return stds


def calculate_min(data, columns):
    """Calculate minimum value of each column.

    Args:
        data (list): Dataset rows.
        columns (list): Numeric column names.

    Returns:
        list: Minimum values.
    """
    mins = []
    for i, _ in enumerate(columns):
        valid_values = [float(row[i]) for row in data[1:] if row[i] != '']
        mins.append(min(valid_values) if valid_values else None)
    return mins


def calculate_max(data, columns):
    """Calculate maximum value of each column.

    Args:
        data (list): Dataset rows.
        columns (list): Numeric column names.

    Returns:
        list: Maximum values.
    """
    maxs = []
    for i, _ in enumerate(columns):
        valid_values = [float(row[i]) for row in data[1:] if row[i] != '']
        maxs.append(max(valid_values) if valid_values else None)
    return maxs


def calculate_percentile(values, percentile):
    """Calculate a percentile from a list of values.

    Args:
        values (list): List of numbers.
        percentile (float): Percentile (0-100).

    Returns:
        float: The calculated percentile value.
    """
    index = (percentile / 100) * (len(values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] + weight * (values[upper] - values[lower])


def calculate_percent(data, columns, stat):
    """Calculate 25%, 50%, or 75% percentile.

    Args:
        data (list): Dataset rows.
        columns (list): Column names.
        stat (str): One of '25%', '50%', '75%'.

    Returns:
        list: List of percentile values.
    """
    results = []
    percent_value = int(stat.strip('%'))
    for i, _ in enumerate(columns):
        values = sorted(float(row[i]) for row in data[1:] if row[i] != '')
        results.append(calculate_percentile(values, percent_value) if values else None)
    return results


def display_calc(values):
    """Display statistics values aligned.

    Args:
        values (list): List of values to display.
    """
    max_feature_length = 20
    for value in values:
        formatted_value = f"{value:.6f}" if value is not None else "NaN"
        print(f"{formatted_value:>{max_feature_length}}", end=" ")
    print()


def display_data(data):
    """Display summary statistics of dataset.

    Args:
        data (list): Full dataset with headers.
    """
    columns = filter_columns(data)
    stats = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    header = data[0]
    col_indices = [header.index(col) for col in columns]
    filtered_data = [[row[i] for i in col_indices] for row in data]
    truncated_headers = [col[:15] + '...' if len(col) > 15 else col for col in columns]

    print(f"{'':<15}", end=" ")
    for head in truncated_headers:
        print(f"{head:>20}", end=" ")
    print()

    count = calculate_count(filtered_data, columns)
    mean = calculate_mean(filtered_data, columns)
    std = calculate_std(filtered_data, columns, mean)
    min_values = calculate_min(filtered_data, columns)
    max_values = calculate_max(filtered_data, columns)

    for stat in stats:
        print(f"{stat:<15}", end=" ")
        match stat:
            case "Count":
                values = count
            case "Mean":
                values = mean
            case "Std":
                values = std
            case "Min":
                values = min_values
            case "Max":
                values = max_values
            case _:
                values = calculate_percent(filtered_data, columns, stat)
        display_calc(values)


def main():
    """
    Entry point of the program.
    Handles CSV loading, data validation, and display of statistics.
    """
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    data = read_csv(file_path)

    if not data:
        print(f"Failed to read data from {file_path}")
        sys.exit(1)

    columns = filter_columns(data)
    if not columns:
        print("No numeric columns found in the dataset.")
        sys.exit(1)

    display_data(data)


if __name__ == "__main__":
    main()
