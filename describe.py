import csv
import math
import sys
import os

# Helper functions
def read_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: '{file_path}'")
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"Cannot read file: '{file_path}'")

    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        try:
            headers = next(reader)
        except StopIteration:
            raise ValueError(f"The file '{file_path}' is empty or does not contain headers.")
        for row in reader:
            data.append(row)
        if not data:
            raise ValueError(f"The file '{file_path}' contains headers but no data rows.")
    return headers, data

def is_numeric(value):
    """Checks if the given value is numeric."""
    try:
        float(value)
        return True
    except ValueError:
        return False

def calculate_count(data):
    """Returns the number of elements in the given data."""
    return len(data)

def calculate_mean(data):
    """Calculates the mean of the given data."""
    return sum(data) / calculate_count(data)

def calculate_std(data, mean):
    """Calculates the standard deviation of the given data."""
    variance = sum((x - mean) ** 2 for x in data) / calculate_count(data)
    return math.sqrt(variance)

def calculate_min_max(data):
    """Calculates the minimum and maximum values in the given data."""
    return min(data), max(data)

def calculate_percentiles(data, percentile):
    """Calculates a percentile of the given data."""
    data_sorted = sorted(data)
    index = int(percentile * len(data_sorted))
    return data_sorted[index]

# Main function
def main(file_path):
    """Main entry point of the program."""
    headers, data = read_csv(file_path)
    transposed_data = list(zip(*data))

    numeric_data = []
    numeric_headers = []
    
    columns_to_ignore = ['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand']
    
    for i, col in enumerate(transposed_data):
        col_filtered = [float(val) for val in col if is_numeric(val)]

        if col_filtered and headers[i] not in columns_to_ignore:
            numeric_data.append(col_filtered)
            numeric_headers.append(headers[i])

    # Limit the length of feature names
    max_feature_length = 15  # Maximum length for feature names
    truncated_headers = [header[:max_feature_length-3] + '...' if len(header) > max_feature_length else header for header in numeric_headers]
    
    # Print the headers for the numerical features
    print(f"{'':<{max_feature_length}}", end=" ")
    for header in truncated_headers:
        print(f"{header:>20}", end=" ")
    print()  # New line after headers

    # Print the statistics
    stats = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]

    for stat in stats:
        if stat == "Count":
            values = [calculate_count(col) for col in numeric_data]
        elif stat == "Mean":
            values = [calculate_mean(col) for col in numeric_data]
        elif stat == "Std":
            values = [calculate_std(col, calculate_mean(col)) for col in numeric_data]
        elif stat == "Min":
            values = [calculate_min_max(col)[0] for col in numeric_data]
        elif stat == "Max":
            values = [calculate_min_max(col)[1] for col in numeric_data]
        else:  # Percentiles
            percentile_index = {"25%": 0.25, "50%": 0.50, "75%": 0.75}[stat]
            values = [calculate_percentiles(col, percentile_index) for col in numeric_data]

        # Print the stat values
        print(f"{stat:<{max_feature_length}}", end=" ")
        for value in values:
            print(f"{value:>20.6f}", end=" ")
        print()  # New line after each stat

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe.py dataset_train.csv")
        sys.exit(1)

    main(sys.argv[1])
