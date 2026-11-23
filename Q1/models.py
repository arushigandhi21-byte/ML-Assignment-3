import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, hidden, in_dim=3 * 32 * 32, n_classes=10):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)
        self.hidden = list(hidden)

    def forward(self, x):
        return self.net(x.flatten(1))


CONFIGS = {
    "shallow_wide": [512],
    "deep_narrow": [350] * 5,
}


def build(name):
    return MLP(CONFIGS[name])
