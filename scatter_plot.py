import csv, os, re, sys
import matplotlib.pyplot as plt
import numpy as np


colors = {
    "Gryffindor": "#f32100",
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
        if all(number_regex.fullmatch(row[i]) or row[i] == '' for row in rows):
            columns.append(column)
    return columns


def extract_column_data(data):
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
    header = data[0]
    col_indices = [i for i, col in enumerate(header) if col in columns]
    house_data = {house: {col: [] for col in columns} for house in colors.keys()}

    for row in data[1:]:
        house = row[1]
        if house in house_data:
            for i, column in enumerate(columns):
                if row[col_indices[i]] != '':
                    house_data[house][column].append(float(row[col_indices[i]]))
    
    return house_data


def display_scatter_plots(data, subject):
    column_data = extract_column_data(data)
    columns = list(column_data.keys())
    
    if subject not in columns:
        print(f"Error: '{subject}' is not a valid subject")
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
                ax.scatter(x_values, y_values, alpha=0.8, s=5, label=house, color=colors.get(house, 'gray'))
        
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
    columns = filter_columns(data)
    return columns


def user_input_subject(valid_subjects):
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
                print("Invalid choice, please try again")
        except ValueError:
            print("Invalid input, please enter a number")


def pearson_correlation(x, y):
    n = len(x)
    if n == 0:
        return 0  

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denom_x = sum((x[i] - mean_x) ** 2 for i in range(n)) ** 0.5
    denom_y = sum((y[i] - mean_y) ** 2 for i in range(n)) ** 0.5

    if denom_x == 0 or denom_y == 0:
        return 0  

    return num / (denom_x * denom_y)


def find_strongest_correlations(result):
    correlation_scores = []

    for subject1 in result:
        for subject2 in result[subject1]:
            correlations = result[subject1][subject2].values()
            avg_correlation = sum(abs(corr) for corr in correlations) / len(correlations)
            correlation_scores.append(((subject1, subject2), avg_correlation))

    correlation_scores.sort(key=lambda x: x[1], reverse=True)

    return correlation_scores[0]


def check_similar_features(data):
    column_data = extract_column_data(data)
    columns = list(column_data.keys())
    
    house_data = extract_house_data(data, columns)
    
    result = {col_gen: {col: {house: [] for house in colors.keys()} for col in columns if col != col_gen} 
              for col_gen in columns}
    
    for col_gen in columns:
        for col in columns:
            if col_gen == col:
                continue

            for house, house_values in house_data.items():
                x_values = house_values[col_gen]
                y_values = house_values[col]
                
                mask = [x is not None and y is not None for x, y in zip(x_values, y_values)]
                x_values = [x for x, m in zip(x_values, mask) if m]
                y_values = [y for y, m in zip(y_values, mask) if m]

                correlation = pearson_correlation(x_values, y_values)
                result[col_gen][col][house] = correlation

    top_correlations = find_strongest_correlations(result)

    print(f"The strongest correlation is between {top_correlations[0][0]} and {top_correlations[0][1]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py <dataset.csv>")
    else:
        data = read_csv(sys.argv[1])
    if data:
        valid_subjects = get_valid_subjects(data)
        if valid_subjects:
            subject = user_input_subject(valid_subjects)
            if subject == "Similar Features":
                check_similar_features(data)
            elif subject == "Exit":
                print("Exiting program")
            else:
                display_scatter_plots(data, subject)
        else:
            print("No valid numeric subjects found in the dataset")
    else:
        print("No valid data loaded")
