from .SonarSubNetwork import SonarSubNetwork
import torch
import torch.nn as nn

class SiameseSonarNetwork(nn.Module):
    """Réseau global instanciant la structure siamoise"""
    def __init__(self):
        super().__init__()
        # Une seule instance de sous-réseau : les poids sont partagés et identiques pour les deux bras
        self.branch = SonarSubNetwork() 
        
    def forward(self, img_A, meta_A, img_B, meta_B):
        emb_A = self.branch(img_A, meta_A)
        emb_B = self.branch(img_B, meta_B)
        
        # Calcul de la distance euclidienne entre les deux signatures vectorielles
        distance = torch.norm(emb_A - emb_B, dim=1)
        return distance