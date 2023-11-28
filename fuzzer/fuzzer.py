import random
import string
import os
import re
import pandas as pd

# Define grammar elements
scalers = ["MinMax", "Standard", "AbsMax"]
kernels = ["linear", "poly", "rbf", "sigmoid", "precomputed"]
criteria = ["gini", "entropy", "log_loss"]
splitters = ["best", "random"]

# Track generated data and algo block names
generated_data_blocks = []
generated_algo_blocks = []

# Function to generate random strings
def random_string():
    return ''.join(random.choices(string.ascii_letters, k=random.randint(3, 8)))

# Available datasets
datasets_dir = './datasets'
datasets = [os.path.join(datasets_dir, d) for d in os.listdir(datasets_dir)]

# Function to generate a valid Data block
def generate_data():
    data_name = random_string()
    while(data_name in generated_data_blocks):
        data_name = random_string()
    
    dataset_path = random.choice(datasets)
    dataset_name = os.path.basename(dataset_path)
    label = re.search('^.*__(\w+)\.csv$', dataset_name).group(1)
    data = pd.read_csv(dataset_path)
    columns = list(data.columns)

    data_block = f"data {data_name} {{\n    source = '{dataset_path}'"
    if random.choice([True, False]):
        data_block += f"\n    label = '{label}'"
    if random.choice([True, False]):
        num_cols_to_drop = random.randint(1, min(2, len(columns)-2))
        columns_to_drop = random.sample(columns, num_cols_to_drop)
        if(label in columns_to_drop):
            columns_to_drop.remove(label)
        quoted_columns = [f"'{col}'" for col in columns_to_drop]
        data_block += f"\n    drop = {' '.join(quoted_columns)}"
    if random.choice([True, False]):
        data_block += f"\n    scaler = {random.choice(scalers)}"
    data_block += "\n}"
    return data_block

# Function to generate a valid Algo block
def generate_algo():
    algo_name = random_string()
    while(algo_name in generated_algo_blocks):
        algo_name = random_string()
    
    algo_type = random.choice(["svm", "knn", "decisionTree", "mlp"])

    algo_block = f"algo {algo_name} {algo_type} {{\n"
    if algo_type == "svm":
        algo_block += f"    C = {random.uniform(0, 1)}\n" if random.choice([True, False]) else ""
        algo_block += f"    kernel = '{random.choice(kernels)}'\n" if random.choice([True, False]) else ""
    elif algo_type == "knn":
        algo_block += f"    n_neighbors = {random.randint(1, 10)}\n" if random.choice([True, False]) else ""
        algo_block += f"    weights = '{random.choice(['uniform', 'distance'])}'\n" if random.choice([True, False]) else ""
    elif algo_type == "decisionTree":
        algo_block += f"    criterion = '{random.choice(criteria)}'\n" if random.choice([True, False]) else ""
        algo_block += f"    splitter = '{random.choice(splitters)}'\n" if random.choice([True, False]) else ""
        algo_block += f"    max_depth = {random.randint(5, 15)}\n" if random.choice([True, False]) else ""
    elif algo_type == "mlp":
        algo_block += f"    hidden_layer_sizes = {' '.join([str(random.randint(4, 10)) for _ in range(random.randint(1, 3))])}\n" if random.choice([True, False]) else ""
    algo_block += "}"
    return algo_block

# Function to generate a valid Trainer block referencing existing Data and Algo blocks
def generate_trainer():
    data_ref = random.choice(generated_data_blocks)
    algo_ref = random.choice(generated_algo_blocks)

    trainer_block = f"trainer {{\n    data = data.{data_ref}\n    model = algo.{algo_ref}"
    trainer_block += f"\n    train_test_split = {random.uniform(0, 1)}" if random.choice([True, False]) else ""
    trainer_block += f"\n    show_metrics = {'true' if random.choice([True, False]) else 'false'}"
    trainer_block += "\n}"
    return trainer_block

# Function to generate a Model block
def generate_model():
    model_block = ""
    for _ in range(random.randint(1, 6)):
        choice = random.choice([generate_data, generate_algo])
        if choice == generate_data:
            data_block = generate_data()
            model_block += data_block + "\n\n"
            # Extract data block name and add to the list of generated data blocks
            data_name = data_block.split()[1]
            generated_data_blocks.append(data_name)
        elif choice == generate_algo:
            algo_block = generate_algo()
            model_block += algo_block + "\n\n"
            # Extract algo block name and add to the list of generated algo blocks
            algo_name = algo_block.split()[1]
            generated_algo_blocks.append(algo_name)

    for _ in range(random.randint(1, 4)):
        # Only generate a trainer if there are existing data and algo blocks
        if generated_data_blocks and generated_algo_blocks:
            model_block += generate_trainer() + "\n\n"
    return model_block

# Generate a program using Model entry
generated_program = generate_model()

file_id = random_string() + random_string()
file_name = "./generated_programs/gen_" + file_id +".neoml"
with open(file_name, "w") as file:
    file.write(generated_program)

print(f"Generated program saved in '{file_name}'")