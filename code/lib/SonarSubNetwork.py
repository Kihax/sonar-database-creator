import torch
import torch.nn as nn


class SonarSubNetwork(nn.Module):
    """Branche unique (Backbone) chargée d'extraire l'empreinte (embedding) d'une imagette."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}

        depth = int(self.config.get("depth", 3))
        base_channels = int(self.config.get("base_channels", 16))
        kernel_size = int(self.config.get("kernel_size", 3))
        fc_hidden_dim = int(self.config.get("fc_hidden_dim", 256))
        embedding_dim = int(self.config.get("embedding_dim", 128))
        meta_feature_count = int(self.config.get("meta_feature_count", 5))

        layers = []
        in_channels = 2
        current_channels = base_channels

        for block_idx in range(depth):
            layers.extend([
                nn.Conv2d(in_channels, current_channels, kernel_size=kernel_size, padding=1),
                nn.BatchNorm2d(current_channels),
                nn.ReLU(),
            ])
            if block_idx < depth - 1:
                layers.append(nn.MaxPool2d(2, 2))
            in_channels = current_channels
            current_channels *= 2

        layers.append(nn.AdaptiveAvgPool2d((8, 8)))
        self.cnn = nn.Sequential(*layers)

        last_channels = base_channels * (2 ** (depth - 1))
        self.fc = nn.Sequential(
            nn.Linear(last_channels * 8 * 8 + meta_feature_count, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, embedding_dim),
        )

    def forward(self, x_img, x_meta):
        features = self.cnn(x_img)
        features = features.view(features.size(0), -1)
        merged = torch.cat((features, x_meta), dim=1)
        embedding = self.fc(features) ## a changer pour remettre les features  
        return embedding