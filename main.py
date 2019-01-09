import pandas as pd
from dataset.datasets import TrainingDataset, ValidationDataset, TestDataset
from dataset.vocabulary import Vocabulary
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader
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

def train_validate_model(model, optimizer, criterion, training_ds, validation_ds, batch_size=128, num_epochs=10):
    # Create dataloader for batching support of every dataset
    training_dl = DataLoader(training_ds, batch_size=batch_size)
    validation_dl = DataLoader(validation_ds, batch_size=batch_size)

    for epoch in range(1, num_epochs + 1):
        print("Starting epoch {}...".format(epoch))

        training_loss = 0.0
        validation_loss = 0.0

        for inputs, labels in tqdm(training_dl, desc="[TRAINING]"):
            break

        for inputs, labels in tqdm(validation_dl, desc="[VALIDATION]"):
            break

        print("Completed epoch {}!\n".format(epoch))

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

    print("[SETUP] Completed!\n")

    print("[TRAINING & VALIDATION] Starting training and validation of model...\n")
    train_validate_model(None, None, None, training_ds, validation_ds)
    print("[TRAINING & VALIDATION] Completed!\n")
