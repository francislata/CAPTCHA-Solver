import os
import pandas as pd

TRAIN_PATH = "train"
TEST_PATH = "test"

def generate_dataset(dataset_path, csv_filename):
    inputs = []
    labels = []

    for filename in os.listdir(dataset_path):
        if filename.endswith(".png"):
            inputs.append(filename)
            labels.append(filename[:3])

    dataset = {"inputs": inputs, "labels": labels}
    dataset_df = pd.DataFrame(dataset)
    dataset_df.to_csv("{}/{}".format(dataset_path, csv_filename, index=False))

if __name__ == "__main__":
    print("Creating training dataset...")
    generate_dataset(TRAIN_PATH, "train.csv")
    print("Created training dataset!\n")

    print("Creating test dataset...")
    generate_dataset(TEST_PATH, "test.csv")
    print("Created test dataset!")
