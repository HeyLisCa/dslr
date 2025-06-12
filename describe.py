import csv
import os
import re
import sys
import signal


# Handle the SIGINT signal
signal.signal(signal.SIGINT, signal.SIG_DFL)


class InvalidDatasetError(Exception):
    """Custom exception raised for invalid datasets."""
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
    Filters columns that are numeric and not 'id' or 'index'.

    Args:
        data (list): A list of rows from the CSV file, where each row is a list of strings.

    Returns:
        list: A list of column names that are numeric and not 'id' or 'index'.
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


def calculate_count(data, columns):
    """
    Calculates the count of non-empty entries for each column.

    Args:
        data (list): A list of rows from the CSV file.
        columns (list): A list of column names to calculate the count for.

    Returns:
        list: A list of counts of non-empty entries for each column.
    """
    counts = []
    for i in range(len(columns)):
        counts.append(sum(1 for row in data[1:] if row[i] != ''))
    return counts


def calculate_mean(data, columns):
    """
    Calculates the mean for each numeric column.

    Args:
        data (list): A list of rows from the CSV file.
        columns (list): A list of numeric column names.

    Returns:
        list: A list of means for each numeric column.
    """
    means = []
    for i, _ in enumerate(columns):
        valid_values = [float(row[i]) for row in data[1:] if row[i] != '']
        mean = sum(valid_values) / len(valid_values)
        means.append(mean)
    return means


def calculate_std(data, columns, means):
    """
    Calculates the standard deviation for each numeric column.

    Args:
        data (list): A list of rows from the CSV file.
        columns (list): A list of numeric column names.
        means (list): A list of mean values for the corresponding columns.

    Returns:
        list: A list of standard deviations for each numeric column.
    """
    stds = []
    for i, mean in enumerate(means):
        valid_values = [float(row[i]) for row in data[1:] if row[i] != '']
        variance = sum((val - mean) ** 2 for val in valid_values) / len(valid_values)
        stds.append(variance ** 0.5)
    return stds


def calculate_min(data, columns):
    """
    Calculates the minimum value for each numeric column.

    Args:
        data (list): A list of rows from the CSV file.
        columns (list): A list of numeric column names.

    Returns:
        list: A list of minimum values for each numeric column.
    """
    mins = []
    for i, _ in enumerate(columns):
        valid_values = [float(row[i]) for row in data[1:] if row[i] != '']
        mins.append(valid_values[0] if valid_values else None)
        for val in valid_values[1:]:
            if val < mins[-1]:
                mins[-1] = val
    return mins


def calculate_max(data, columns):
    """
    Calculates the maximum value for each numeric column.

    Args:
        data (list): A list of rows from the CSV file.
        columns (list): A list of numeric column names.

    Returns:
        list: A list of maximum values for each numeric column.
    """
    maxs = []
    for i, _ in enumerate(columns):
        valid_values = [float(row[i]) for row in data[1:] if row[i] != '']
        maxs.append(valid_values[0] if valid_values else None)
        for val in valid_values[1:]:
            if val > maxs[-1]:
                maxs[-1] = val
    return maxs


def calculate_percentile(values, percentile):
    """
    Calculates a specific percentile from a list of values.

    Args:
        values (list): A list of numeric values.
        percentile (float): The desired percentile (between 0 and 100).

    Returns:
        float: The calculated percentile value.
    """
    index = (percentile / 100) * (len(values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] + weight * (values[upper] - values[lower])


def calculate_percent(data, columns, stat):
    """
    Calculates the 25th, 50th, or 75th percentile for each numeric column.

    Args:
        data (list): A list of rows from the CSV file.
        columns (list): A list of numeric column names.
        stat (str): The desired percentile ('25%', '50%', or '75%').

    Returns:
        list: A list of calculated percentile values for each numeric column.
    """
    results = []
    for i, _ in enumerate(columns):
        values = sorted(float(row[i]) for row in data[1:] if row[i] != '')
        if not values:
            results.append(None)
            continue
        if stat == '25%':
            results.append(calculate_percentile(values, 25))
        elif stat == '50%':
            results.append(calculate_percentile(values, 50))
        elif stat == '75%':
            results.append(calculate_percentile(values, 75))
    return results


def calculate_skewness(data, columns, means, stds):
    """
    Calculates the skewness for each numeric column.

    Args:
        data (list): A list of rows from the CSV file.
        columns (list): A list of numeric column names.
        means (list): A list of mean values for the corresponding columns.
        stds (list): A list of standard deviation values for the corresponding columns.

    Returns:
        list: A list of skewness values for each numeric column.
    """
    skewness = []
    for i, mean in enumerate(means):
        valid_values = [float(row[i]) for row in data[1:] if row[i] != '']
        n = len(valid_values)
        if n < 3 or stds[i] == 0:
            skewness.append(None)
            continue
        m3 = sum((x - mean) ** 3 for x in valid_values) / n
        skew = m3 / (stds[i] ** 3)
        skewness.append(skew)
    return skewness


def calculate_kurtosis(data, columns, means, stds):
    """
    Calculates the kurtosis for each numeric column.

    Args:
        data (list): A list of rows from the CSV file.
        columns (list): A list of numeric column names.
        means (list): A list of mean values for the corresponding columns.
        stds (list): A list of standard deviation values for the corresponding columns.

    Returns:
        list: A list of kurtosis values for each numeric column.
    """
    kurtosis = []
    for i, mean in enumerate(means):
        valid_values = [float(row[i]) for row in data[1:] if row[i] != '']
        n = len(valid_values)
        if n < 4 or stds[i] == 0:
            kurtosis.append(None)
            continue
        m4 = sum((x - mean) ** 4 for x in valid_values) / n
        kurt = m4 / (stds[i] ** 4) - 3
        kurtosis.append(kurt)
    return kurtosis


def calculate_missing(data, columns):
    """
    Calculates the percentage of missing values for each column.

    Args:
        data (list): A list of rows from the CSV file.
        columns (list): A list of column names to calculate the missing percentage for.

    Returns:
        list: A list of missing value percentages for each column.
    """
    missing = []
    total = len(data) - 1
    for i in range(len(columns)):
        count = sum(1 for row in data[1:] if row[i] == '')
        missing.append((count / total * 100) if total > 0 else None)
    return missing


def calculate_range(data, columns):
    """
    Calculates the range (max - min) for each numeric column.

    Args:
        data (list): A list of rows from the CSV file.
        columns (list): A list of numeric column names.

    Returns:
        list: A list of range values for each numeric column.
    """
    ranges = []
    for i, _ in enumerate(columns):
        valid_values = [float(row[i]) for row in data[1:] if row[i] != '']
        if not valid_values:
            ranges.append(None)
        else:
            ranges.append(max(valid_values) - min(valid_values))
    return ranges


def display_calc(values, columns):
    """
    Displays calculated statistics for the columns.

    Args:
        values (list): A list of calculated statistical values to display.
        columns (list): A list of column names to associate with the statistics.
    """
    max_feature_length = 20
    for i, value in enumerate(values):
        formatted_value = f"{value:.6f}" if value is not None else "NaN"
        print(f"{formatted_value:>{max_feature_length}}", end=" ")
    print()


def display_data(data):
    """
    Displays the calculated statistics for the dataset.

    Args:
        data (list): A list of rows from the CSV file.
    """
    columns = filter_columns(data)
    stats = [
        "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max",
        "Skewness", "Kurtosis", "Missing (%)", "Range"
    ]
    header = data[0]
    col_indices = [header.index(col) for col in columns]
    filtered_data = [[row[i] for i in col_indices] for row in data]

    truncated_headers = [col[:15] + '...' if len(col) > 15 else col for col in columns]

    print(f"{'':<15}", end=" ")
    for header in truncated_headers:
        print(f"{header:>20}", end=" ")
    print()

    count = calculate_count(filtered_data, columns)
    mean = calculate_mean(filtered_data, columns)
    std = calculate_std(filtered_data, columns, mean)
    min = calculate_min(filtered_data, columns)
    max = calculate_max(filtered_data, columns)
    skewness = calculate_skewness(filtered_data, columns, mean, std)
    kurtosis = calculate_kurtosis(filtered_data, columns, mean, std)
    missing = calculate_missing(filtered_data, columns)
    ranges = calculate_range(filtered_data, columns)

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
                values = min
            case "Max":
                values = max
            case "Skewness":
                values = skewness
            case "Kurtosis":
                values = kurtosis
            case "Missing (%)":
                values = missing
            case "Range":
                values = ranges
            case _:
                values = calculate_percent(filtered_data, columns, stat)
        display_calc(values, columns)


def main():
    """
    Entry point of the program.
    Handles argument validation, file loading and display.
    """
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    data = read_csv(file_path)

    if data:
        header = data[0]
        if "Hogwarts House" not in header:
            print("Error: 'Hogwarts House' column is missing from the dataset")
            sys.exit(1)
        house_idx = header.index("Hogwarts House")
        if all((len(row) <= house_idx or not row[house_idx].strip()) for row in data[1:]):
            print("Error: All values in 'Hogwarts House' column are empty")
            sys.exit(1)
        columns = filter_columns(data)
        if not columns:
            print("No numeric columns found in the dataset.")
        else:
            display_data(data)
    else:
        print(f"Failed to read data from {file_path}")


if __name__ == "__main__":
    main()
