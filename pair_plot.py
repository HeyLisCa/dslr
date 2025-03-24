import sys, os, csv, re
import matplotlib.pyplot as plt


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
        if all(number_regex.fullmatch(row[i]) or row[i] == '' for row in rows) and not all(row[i] == '' for row in rows):
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



def display_pair_plot(data):
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
                        ax.hist(values, bins=15, color=colors.get(house, 'gray'), alpha=0.5, label=house, edgecolor='black')
            else:
                for house, house_values in house_data.items():
                    x_values = house_values[col_x]
                    y_values = house_values[col_y]

                    mask = [x is not None and y is not None for x, y in zip(x_values, y_values)]
                    x_values = [x for x, m in zip(x_values, mask) if m]
                    y_values = [y for y, m in zip(y_values, mask) if m]

                    if x_values and y_values:
                        ax.scatter(x_values, y_values, alpha=0.5, s=3, label=house, color=colors.get(house, 'gray'))


            ax.set_xticks([])
            ax.set_yticks([])

            if j == 0:
                ax.set_ylabel(col_x[:16], fontsize=7)
            if i == n - 1:
                ax.set_xlabel(col_y[:16], fontsize=7)

            ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset.csv>")
    else:
        data = read_csv(sys.argv[1])
        if data:
            display_pair_plot(data)
        else:
            print("No numeric columns found in the dataset")