import pandas as pd
from dataset.datasets import TrainingDataset, ValidationDataset, TestDataset
from dataset.vocabulary import Vocabulary
from sklearn.model_selection import train_test_split
import torch

TRAIN_FILEPATH = "dataset/train/{}"
TEST_FILEPATH = "dataset/test/{}"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def open_create_datasets(vocab):
    training_ds_df = pd.read_csv(TRAIN_FILEPATH.format("train.csv"))
    test_ds_df = pd.read_csv(TEST_FILEPATH.format("test.csv"))

    # Split training set to training and validation sets
    training_inputs, validation_inputs, training_labels, validation_labels = train_test_split(training_ds_df.inputs.values.tolist(), training_ds_df.labels.values.tolist(), test_size=0.1, random_state=42)

    training_ds = TrainingDataset(training_inputs, training_labels, vocab, TRAIN_FILEPATH, DEVICE)
    validation_ds = ValidationDataset(validation_inputs, validation_labels, vocab, TRAIN_FILEPATH, DEVICE)
    test_ds = TestDataset(test_ds_df.inputs.values.tolist(), test_ds_df.labels.values.tolist(), vocab, TEST_FILEPATH, DEVICE)

    return training_ds, validation_ds, test_ds

if __name__ == "__main__":
    print("[NOTE] Current device being used: {}\n".format(DEVICE))
    
    print("[SETUP] Setting up datasets...\n")

    print("Creating training, validation, and test datasets...")
    vocab = Vocabulary()
    training_ds, validation_ds, test_ds = open_create_datasets(vocab)
    print("Done creating training, validation, and test datasets!\n")

    print("The training dataset has {} samples.".format(len(training_ds)))
    print("The validation dataset has {} samples.".format(len(validation_ds)))
    print("The test dataset has {} samples.\n".format(len(test_ds)))

    print("[SETUP] Datasets are setup!\n")
