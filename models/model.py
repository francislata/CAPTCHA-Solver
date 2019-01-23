import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = self.create_encoder()

    def forward(self, x):
        x = self.encoder(x)
        return x

    def create_encoder(self):
        return nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, x, h):
        return None 

class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()

    def forward(self, x, h):
        return None

class CAPTCHACNNClassifier(nn.Module):
    def __init__(self, num_letters, num_classes, device):
        super(CAPTCHACNNClassifier, self).__init__()

        self.num_letters = num_letters
        self.encoder = Encoder()
        self.linear = nn.Sequential(*[
            nn.Dropout(),
            nn.Linear(128 * 10 * 10, 512),
            nn.ReLU()
        ])
        self.linear_1 = nn.Sequential(*[
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU()
        ])
        self.linear_2 = nn.Sequential(*[
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU()
        ])
        self.linear_3 = nn.Sequential(*[
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU()
        ])
        self.classifier = nn.Sequential(*[
            nn.Dropout(), 
            nn.Linear(256, num_classes)
        ])

    def forward(self, x):
        x = self.encoder(x).view(x.size(0), -1)
        x = self.linear(x)
        x = torch.stack([self.linear_1(x), self.linear_2(x), self.linear_3(x)], dim=1)
        x = self.classifier(x)
        return x
