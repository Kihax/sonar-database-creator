import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# 1. Module d'Attention Spatiale & Canaux (CBAM)
#    Filtre le bruit de speckle sonar et rehausse les structures d'intérêt.
# ─────────────────────────────────────────────────────────────────────────────
class SpatialChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        # 1. Channel Attention (AvgPool + MaxPool)
        red_channels = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, red_channels, bias=False),
            nn.ReLU(),
            nn.Linear(red_channels, channels, bias=False)
        )
        # 2. Spatial Attention (Conv 7x7)
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        
        # Channel Attention
        avg_pool = torch.mean(x, dim=(2, 3))
        max_pool = torch.max(torch.max(x, dim=2)[0], dim=2)[0]
        channel_weight = torch.sigmoid(self.fc(avg_pool) + self.fc(max_pool)).view(b, c, 1, 1)
        x = x * channel_weight

        # Spatial Attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial = torch.max(x, dim=1, keepdim=True)[0]
        spatial_weight = torch.sigmoid(self.conv_spatial(torch.cat([avg_spatial, max_spatial], dim=1)))
        
        return x * spatial_weight


# ─────────────────────────────────────────────────────────────────────────────
# 2. Module de Cross-Attention Inter-Imagettes (Optionnel)
# ─────────────────────────────────────────────────────────────────────────────
class SonarCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor):
        # Cross Attention A -> B
        attn_out1, _ = self.mha(query=feat1, key=feat2, value=feat2)
        out1 = self.norm(feat1 + attn_out1)

        # Cross Attention B -> A
        attn_out2, _ = self.mha(query=feat2, key=feat1, value=feat1)
        out2 = self.norm(feat2 + attn_out2)

        return out1, out2


# ─────────────────────────────────────────────────────────────────────────────
# 3. Réseau Siamois Sonar Principal
# ─────────────────────────────────────────────────────────────────────────────
class AttentionSiameseSonarNetwork(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        
        self.depth = model_config.get("model", {}).get("depth", 3)
        base_channels = model_config.get("model", {}).get("base_channels", 16)
        kernel_size = model_config.get("model", {}).get("kernel_size", 3)
        fc_hidden_dim = model_config.get("model", {}).get("fc_hidden_dim", 256)
        self.embedding_dim = model_config.get("model", {}).get("embedding_dim", 128)
        self.meta_feature_count = model_config.get("data", {}).get("meta", []).__len__()
        in_channels = model_config.get("data", {}).get("channel", []).__len__() 

        # 1. Encodeur CNN + Attention Spatiale (CBAM)
        layers = []
        
        for i in range(self.depth):
            out_channels = base_channels * (2 ** i)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(SpatialChannelAttention(out_channels))  # Filtrage du bruit sonar
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        self.cnn_encoder = nn.Sequential(*layers)
        
        # Adaptatif 4x4 pour fixer la taille de sortie visuelle
        self.grid_size = 4
        self.global_pool = nn.AdaptiveAvgPool2d((self.grid_size, self.grid_size))
        
        self.feature_dim = in_channels
        cnn_out_dim = self.feature_dim * self.grid_size * self.grid_size

        # Module de Cross-Attention (pour forward_pair si besoin)
        self.cross_attention = SonarCrossAttention(dim=self.feature_dim, num_heads=4)

        # 2. Fully Connected Layers + Fusion Métadonnées
        total_input_dim = cnn_out_dim + self.meta_feature_count
        
        self.fc = nn.Sequential(
            nn.Linear(total_input_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(fc_hidden_dim, self.embedding_dim)
        )

    def extract_features(self, img: torch.Tensor) -> torch.Tensor:
        """Passe l'image dans le CNN avec attention et sort le feature map plati."""
        feat = self.cnn_encoder(img)
        feat = self.global_pool(feat)
        return feat

    def forward(self, img: torch.Tensor, meta: torch.Tensor = None) -> torch.Tensor:
        """
        Passage standard à une seule image (100% compatible avec utility_training.py).
        Appelé via : model(img_A, meta_A)
        """
        # 1. Extraction visuelle avec attention CBAM
        feat_img = self.extract_features(img)
        feat_img = torch.flatten(feat_img, 1)

        # 2. Fusion avec les métadonnées (profondeur, angle, etc.)
        if self.meta_feature_count > 0 and meta is not None:
            if meta.dim() == 1:
                meta = meta.unsqueeze(1)
            x = torch.cat((feat_img, meta), dim=1)
        else:
            x = feat_img

        # 3. Projection et Normalisation L2 (essentiel pour InfoNCE / Cosine Loss)
        embedding = self.fc(x)
        return F.normalize(embedding, p=2, dim=1)

    def forward_pair(self, img1: torch.Tensor, img2: torch.Tensor, meta1: torch.Tensor = None, meta2: torch.Tensor = None):
        """
        Passage conjoint en paire avec Cross-Attention (A <-> B).
        À utiliser si vous faites évoluer utility_training.py.
        """
        # Extrait les features (B, C, 4, 4)
        feat1 = self.extract_features(img1)
        feat2 = self.extract_features(img2)

        # Transformation en tokens (B, 16, C) pour le transformer
        b, c, h, w = feat1.size()
        t1 = feat1.view(b, c, h * w).permute(0, 2, 1)
        t2 = feat2.view(b, c, h * w).permute(0, 2, 1)

        # Alignment par Cross-Attention
        t1_att, t2_att = self.cross_attention(t1, t2)

        # Re-platitude des tokens
        f1_flat = torch.flatten(t1_att, 1)
        f2_flat = torch.flatten(t2_att, 1)

        # Fusion meta + FC pour Image 1
        if self.meta_feature_count > 0 and meta1 is not None:
            if meta1.dim() == 1: meta1 = meta1.unsqueeze(1)
            x1 = torch.cat((f1_flat, meta1), dim=1)
        else: x1 = f1_flat

        # Fusion meta + FC pour Image 2
        if self.meta_feature_count > 0 and meta2 is not None:
            if meta2.dim() == 1: meta2 = meta2.unsqueeze(1)
            x2 = torch.cat((f2_flat, meta2), dim=1)
        else: x2 = f2_flat

        emb1 = F.normalize(self.fc(x1), p=2, dim=1)
        emb2 = F.normalize(self.fc(x2), p=2, dim=1)

        return emb1, emb2