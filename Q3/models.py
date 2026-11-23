import torch.nn as nn


class ConvAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.to_code = nn.Linear(128 * 4 * 4, d)
        self.from_code = nn.Sequential(nn.Linear(d, 128 * 4 * 4), nn.ReLU())
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def encode(self, x):
        return self.to_code(self.encoder_conv(x))

    def decode(self, z):
        return self.decoder_conv(self.from_code(z).view(-1, 128, 4, 4))

    def forward(self, x):
        return self.decode(self.encode(x))
