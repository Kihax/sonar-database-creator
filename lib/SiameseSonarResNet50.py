import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class SiameseSonarResNet50(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        
        fc_hidden_dim = model_config.get("fc_hidden_dim", 256)
        self.embedding_dim = model_config.get("embedding_dim", 128)
        self.meta_feature_count = model_config.get("meta_feature_count", 0)
        pretrained = model_config.get("pretrained", False)

        # 1. Chargement du squelette ResNet-50 officiel
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)

        # Adaptation de la première couche pour 1 canal sonar (au lieu de 3 RGB)
        if pretrained:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))
            backbone.conv1 = new_conv
        else:
            backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Extracteur de caractéristiques (jusqu'au pooling global)
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,  # 3 blocs Bottleneck  -> 256 canaux
            backbone.layer2,  # 4 blocs Bottleneck  -> 512 canaux
            backbone.layer3,  # 6 blocs Bottleneck  -> 1024 canaux
            backbone.layer4,  # 3 blocs Bottleneck  -> 2048 canaux
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Le ResNet-50 extrait un vecteur visuel de 2048 dimensions
        cnn_out_dim = 2048

        # 2. Tête de projection et fusion des métadonnées
        total_input_dim = cnn_out_dim + self.meta_feature_count
        
        self.fc = nn.Sequential(
            nn.Linear(total_input_dim, fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(fc_hidden_dim, self.embedding_dim)
        )

    def forward(self, img: torch.Tensor, meta: torch.Tensor = None) -> torch.Tensor:
        feat_img = self.encoder(img)
        feat_img = torch.flatten(feat_img, 1)

        # Fusion avec métadonnées si présentes
        if self.meta_feature_count > 0 and meta is not None:
            if meta.dim() == 1:
                meta = meta.unsqueeze(1)
            x = torch.cat((feat_img, meta), dim=1)
        else:
            x = feat_img

        return self.fc(x)