import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

class SonarSubNetwork(nn.Module):
    """Branche unique (Backbone) chargée d'extraire l'empreinte (embedding) d'une imagette"""
    def __init__(self):
        super().__init__()
        # Entrée : 3 canaux (Image Sonar, Pitch 2D, Heading 2D)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 256x256 -> 128x128
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 128x128 -> 64x64
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)) # Écrase la géométrie spatiale à une taille fixe (8, 8)
        )
        
        # Taille en sortie du CNN aplati : 64 canaux * 8 * 8 pixels = 4096
        # On y ajoute les 3 caractéristiques globales (depth, sin_heading, cos_heading) -> 4096 + 3 = 4099
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8 + 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128) # Vecteur caractéristique final (Embedding) de taille 128
        )
        
    def forward(self, x_img, x_meta):
        features = self.cnn(x_img)
        features = features.view(features.size(0), -1) # Aplatissement en (Batch, 4096)
        
        # Fusion par concaténation : Données Visuelles + Métadonnées Capteurs
        merged = torch.cat((features, x_meta), dim=1) # Forme : (Batch, 4099)
        
        embedding = self.fc(merged)
        return embedding