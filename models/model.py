import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, num_classes):
        super(Encoder, self).__init__()
        self.encoder = self.create_encoder()
        self.classifier = nn.Linear(60, )

    def forward(self, img):
        img = self.encoder(img)
        return img

    def create_encoder(self):
        return nn.Sequential(
            nn.Conv2d(3, 40, (3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),

            nn.Conv2d(40, 60, (3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
        )

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, x, h):
        return None

class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()

    def forward(self, img, h):
        return None
