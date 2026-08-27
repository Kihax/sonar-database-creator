import torch
import torch.nn as nn

class SiameseSonarNetwork(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        
        self.depth = model_config.get("depth", 3)
        base_channels = model_config.get("base_channels", 16)
        kernel_size = model_config.get("kernel_size", 3)
        fc_hidden_dim = model_config.get("fc_hidden_dim", 256)
        self.embedding_dim = model_config.get("embedding_dim", 128)
        self.meta_feature_count = model_config.get("meta_feature_count", 0)

        # 1. Extrait visuel CNN (sur les images sonar)
        layers = []
        in_channels = len(model_config.get("data", {}).get("channel", []))
        for i in range(self.depth):
            out_channels = base_channels * (2 ** i)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        self.cnn_encoder = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Dim de sortie du CNN plat
        cnn_out_dim = in_channels * 4 * 4

        # 2. Dense Layers avec fusion des métadonnées
        total_input_dim = cnn_out_dim + self.meta_feature_count
        
        self.fc = nn.Sequential(
            nn.Linear(total_input_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(fc_hidden_dim, self.embedding_dim)
        )

    def forward(self, img: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """
        Calcule l'embedding unique (Image + Métadonnées).
        👉 Utilisé directement par InfoNCE.
        """
        feat_img = self.cnn_encoder(img)
        feat_img = self.global_pool(feat_img)
        feat_img = torch.flatten(feat_img, 1)

        # Fusion avec les métadonnées si présentes
        if self.meta_feature_count > 0 and meta is not None:
            # S'assurer que 'meta' a la bonne dimension (batch_size, meta_feature_count)
            if meta.dim() == 1:
                meta = meta.unsqueeze(1)
            x = torch.cat((feat_img, meta), dim=1)
        else:
            x = feat_img

        # Projection dans l'espace d'embedding
        embedding = self.fc(x)
        return embedding