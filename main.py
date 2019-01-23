import pandas as pd
from dataset.datasets import *
from dataset.vocabulary import Vocabulary
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.model import *
import utils
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

TRAIN_FILEPATH = "dataset/train/{}"
TEST_FILEPATH = "dataset/test/{}"
CAPTCHACNNCLASSIFIER_MODEL_CHECKPOINT_FILEPATH = "models/checkpoints/CAPTCHACNNClassifier_epoch{}.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def open_create_datasets(vocab):
    training_ds_df = pd.read_csv(TRAIN_FILEPATH.format("train.csv"))
    test_ds_df = pd.read_csv(TEST_FILEPATH.format("test.csv"))

    # Split training set to training and validation sets
    training_inputs, validation_inputs, training_labels, validation_labels = train_test_split(training_ds_df.inputs.values.tolist(), training_ds_df.labels.values.tolist(), test_size=0.1, random_state=42)

    training_ds = TrainingDataset(training_inputs, training_labels, vocab, TRAIN_FILEPATH)
    validation_ds = ValidationDataset(validation_inputs, validation_labels, vocab, TRAIN_FILEPATH)
    test_ds = TestDataset(test_ds_df.inputs.values.tolist(), test_ds_df.labels.values.tolist(), vocab, TEST_FILEPATH)

    return training_ds, validation_ds, test_ds

def train_validate_model(model, optimizer, criterion, training_ds, validation_ds, training_ds_batch_size=64, validation_ds_batch_size=64, num_epochs=10):
    # Create dataloader for batching support of every dataset
    training_dl = DataLoader(training_ds, batch_size=training_ds_batch_size, num_workers=4)
    validation_dl = DataLoader(validation_ds, batch_size=validation_ds_batch_size)

    for epoch in range(1, num_epochs + 1):
        print("Starting epoch {}...".format(epoch))

        training_loss = 0.0
        training_accuracies = []
        
        # Start training model
        model.train()

        for inputs, labels in tqdm(training_dl, desc="[TRAINING]"):
            optimizer.zero_grad()

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            predictions = model(inputs)
            predictions = predictions.view(-1, predictions.size(2), predictions.size(1))
            loss = criterion(predictions, labels)

            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            training_accuracies.append(calculate_accuracy(predictions, labels).item())
            
        print("[TRAINING] Epoch {} loss is {:.4f}\n".format(epoch, training_loss / len(training_dl)))
        print("[TRAINING] Epoch {} accuracy is {:.4f}\n".format(epoch, sum(training_accuracies) / len(training_accuracies)))

        validation_loss = 0.0
        validation_accuracies = []

        # Evaluate the model
        model.eval()

        for inputs, labels in tqdm(validation_dl, desc="[VALIDATION]"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            predictions = model(inputs)
            predictions = predictions.view(-1, predictions.size(2), predictions.size(1))
            loss = criterion(predictions, labels)

            validation_loss += loss.item()
            validation_accuracies.append(calculate_accuracy(predictions, labels).item())

        print("[VALIDATION] Epoch {} loss is {:.4f}\n".format(epoch, validation_loss / len(validation_dl)))
        print("[VALIDATION] Epoch {} accuracy is {:.4f}\n".format(epoch, sum(validation_accuracies) / len(validation_accuracies)))

        print("Completed epoch {}!\n".format(epoch))

def calculate_accuracy(predictions, labels):
    predictions = nn.LogSoftmax(dim=1)(predictions)
    predictions_index = torch.max(predictions, dim=1)[1].view(-1)
    labels = labels.view(-1)
    
    return torch.sum(predictions_index == labels.data).float() / len(labels.data)

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

    print("[TRAINING & VALIDATION] Starting training and validation of encoder classifier model...\n")
    enc_model = CAPTCHACNNClassifier(3, len(vocab.labels_to_idx), DEVICE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(enc_model.parameters(), lr=1e-1, momentum=0.9)

    train_validate_model(enc_model, optimizer, criterion, training_ds, validation_ds, training_ds_batch_size=1024, num_epochs=30)
    print("[TRAINING & VALIDATION] Completed!")
