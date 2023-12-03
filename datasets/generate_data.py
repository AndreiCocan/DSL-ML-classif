import csv
import random

def generate_fake_data():
    id1 = random.randint(0, 10)
    id2 = random.randint(0, 10)
    fake_data = random.randint(1, 100)
    class_value = random.randint(0, 1)
    return id1, id2, fake_data, class_value

def generate_csv_dataset(filename, num_lines):
    with open(filename, 'w') as csvfile:
        fieldnames = ['Id1', 'Id2', 'FakeData', 'Class']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for _ in range(num_lines):
            id1, id2, fake_data, class_value = generate_fake_data()
            writer.writerow({'Id1': id1, 'Id2': id2, 'FakeData': fake_data, 'Class': class_value})

if __name__ == "__main__":
    dataset_filename = "random_dataset.csv"
    num_lines = 700
    generate_csv_dataset(dataset_filename, num_lines)
    print("CSV dataset generated and saved")
