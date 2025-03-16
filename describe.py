import csv
import os
import sys
import re
import math


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
        if all(number_regex.fullmatch(row[i]) or row[i] == '' for row in rows):
            columns.append(column)
    return columns


def calc_count(data, columns):
    values = []
    for i, column in enumerate(columns):
        count = sum(1 for row in data[1:] if row[i] != '')
        values.append(count)
    return values


def calc_mean(data, columns):
    values = []
    for i, column in enumerate(columns):
        mean = sum(float(row[i]) for row in data[1:] if row[i] != '') / calc_count(data, columns)[i]
        values.append(mean)
    return values


def calc_std(data, columns):
    values = []
    for i, column in enumerate(columns):
        mean = calc_mean(data, columns)[i]
        variance = sum((float(row[i]) - mean) ** 2 for row in data[1:] if row[i] != '') / calc_count(data, columns)[i]
        std = math.sqrt(variance)
        values.append(std)
    return values


def calc_min(data, columns):
    values = []
    for i, column in enumerate(columns):
        if data[1][i] == '':
            minimum = float('inf')
        else:
            minimum = float(data[1][i])
        for row in data[1:]:
            if row[i] != '' and float(row[i]) < minimum:
                minimum = float(row[i])
        if minimum == float('inf'):
            minimum = None
        values.append(minimum)
    return values


def calculate_percentile(values, percentile):
    index = (percentile / 100) * (len(values) - 1)
    lower_index = int(index)
    upper_index = min(lower_index + 1, len(values) - 1)
    weight = index - lower_index
    return values[lower_index] + weight * (values[upper_index] - values[lower_index])


def calc_percent(data, columns, stat):
    results = []
    for i, column in enumerate(columns):
        values = [float(row[i]) for row in data[1:] if row[i] != '']
        values.sort()
        if stat == '25%':
            results.append(calculate_percentile(values, 25))
        elif stat == '50%':
            results.append(calculate_percentile(values, 50))
        elif stat == '75%':
            results.append(calculate_percentile(values, 75))
    return results


def calc_max(data, columns):
    values = []
    for i, column in enumerate(columns):
        if data[1][i] == '':
            maximum = -float('inf')
        else:
            maximum = float(data[1][i])
        for row in data[1:]:
            if row[i] != '' and float(row[i]) > maximum:
                maximum = float(row[i])
        if maximum == -float('inf'):
            maximum = None
        values.append(maximum)
    return values


def display_calc(values, columns):
    for i, column in enumerate(columns):
        col_width = len(column)
        value = values[i]

        if i == 0:
            padding = 0
        else:
            formatted_value = f"{value:.6f}"
            count_width = len(formatted_value)
            
            if col_width < count_width:
                padding = 5 - (count_width - col_width)
            else:
                padding = 5
            
        print(" " * padding + f"{value:.6f}".rjust(col_width), end="")
    print()


def display_data(data):
    columns = filter_columns(data)
    stats = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    header = data[0]
    col_indices = [i for i, col in enumerate(header) if col in columns]
    filtered_data = [[row[i] for i in col_indices] for row in data]

    print(" " * 10, end="")
    for col in columns:
        col_width = len(col) + 5
        print(f"{col: <{col_width}}", end="")
    print()

    rows = filtered_data[1:]

    for stat in stats:
        print(f"{stat: <10}", end="")
        match stat:
            case "Count":
                values = calc_count(filtered_data, columns)
            case "Mean":
                values = calc_mean(filtered_data, columns)
            case "Std":
                values = calc_std(filtered_data, columns)
            case "Min":
                values = calc_min(filtered_data, columns)
            case "Max":
                values = calc_max(filtered_data, columns)
            case _:
                values = calc_percent(filtered_data, columns, stat)
        display_calc(values, columns)
        print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset.csv>")
    else:
        data = read_csv(sys.argv[1])
        if data:
            display_data(data)