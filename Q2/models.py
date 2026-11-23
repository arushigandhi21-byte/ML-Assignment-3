import torch.nn as nn

STAGES = ((32,), (64, 64), (128, 128), (320, 320))


class CNN(nn.Module):
    def __init__(self, stages=STAGES, n_classes=10):
        super().__init__()
        layers, cin = [], 3
        for i, stage in enumerate(stages):
            for cout in stage:
                layers += [nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU()]
                cin = cout
            if i < len(stages) - 1:
                layers.append(nn.MaxPool2d(2))
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(cin, n_classes)
        self.stages = stages

    def embed(self, x):
        return self.features(x)

    def forward(self, x):
        return self.head(self.features(x))
