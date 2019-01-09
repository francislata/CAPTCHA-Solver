import torch
import string
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

IMG_SIZE = (100, 100)

class TrainingDataset(Dataset):
    def __init__(self, inputs, labels, vocabulary, filepath, device):
        self.inputs = inputs
        self.labels = labels
        self.vocabulary = vocabulary

        self.filepath = filepath
        self.device = device

        self.transforms = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor()
        ])

    def __len__(self): 
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.convert_sample_to_tensor(idx)

    def convert_sample_to_tensor(self, idx):
        input = self.inputs[idx]
        input = Image.open(self.filepath.format(input))
        input = self.transforms(input).to(self.device)

        label = self.labels[idx]
        label = [self.vocabulary.labels_to_idx[char] for char in label]
        label = torch.FloatTensor(label, device=self.device)

        return input, label

class ValidationDataset(TrainingDataset):
    def __init__(self, inputs, labels, vocabulary, filepath, device):
        super(ValidationDataset, self).__init__(inputs, labels, vocabulary, filepath, device)

class TestDataset(TrainingDataset):
    def __init__(self, inputs, labels, vocabulary, filepath, device):
        super(TestDataset, self).__init__(inputs, labels, vocabulary, filepath, device)
