import torch
import string
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

IMG_SIZE = (100, 100)

class TrainingDataset(Dataset):
    def __init__(self, inputs, labels, vocabulary, filepath):
        self.inputs = inputs
        self.labels = labels
        self.vocabulary = vocabulary

        self.filepath = filepath

        self.transforms = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def __len__(self): 
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.convert_sample_to_tensor(idx)

    def convert_sample_to_tensor(self, idx):
        input = self.inputs[idx]
        input = Image.open(self.filepath.format(input))
        input = self.transforms(input)

        label = self.labels[idx]
        label = [self.vocabulary.labels_to_idx[char] for char in label]
        label = torch.LongTensor(label)

        return input, label

class ValidationDataset(TrainingDataset):
    def __init__(self, inputs, labels, vocabulary, filepath):
        super(ValidationDataset, self).__init__(inputs, labels, vocabulary, filepath)

class TestDataset(TrainingDataset):
    def __init__(self, inputs, labels, vocabulary, filepath):
        super(TestDataset, self).__init__(inputs, labels, vocabulary, filepath)
