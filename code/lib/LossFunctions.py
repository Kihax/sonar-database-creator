import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb_A: torch.Tensor, emb_B: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # 1. NORMALISATION L2 OBLIGATOIRE
        # Projetant les embeddings sur la sphère unité, empêche l'effondrement des poids.
        emb_A = F.normalize(emb_A, p=2, dim=-1)
        emb_B = F.normalize(emb_B, p=2, dim=-1)

        # 2. Calcul de la distance euclidienne (sera maintenant entre 0.0 et 2.0)
        distance = torch.pairwise_distance(emb_A, emb_B, p=2)

        # 3. Calcul des pertes
        # Si label == 1 (positif/similaire) -> on veut distance -> 0
        # Si label == 0 (négatif/dissimilaire) -> on veut distance >= margin
        loss_pos = label * torch.pow(distance, 2)
        loss_neg = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        
        loss = torch.mean(loss_pos + loss_neg)

        if torch.isnan(loss):
            print("⚠️ Warning: NaN detected in loss, replacing with safe value")
            loss = torch.tensor(1.0, device=distance.device, dtype=distance.dtype)

        return loss

class InfoNCELoss(nn.Module):
    """
    InfoNCE adaptée aux datasets de paires (Positive/Negative).
    Pour chaque paire positive du batch, on maximise sa similarité
    par rapport aux autres embeddings du batch.
    Les paires négatives servent de négatifs dans le dénominateur.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_A: torch.Tensor, emb_B: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        emb_A = F.normalize(emb_A, p=2, dim=-1)
        emb_B = F.normalize(emb_B, p=2, dim=-1)

        if label.dim() != 1:
            label = label.view(-1)

        sim_matrix = torch.matmul(emb_A, emb_B.T) / self.temperature
        positive_idx = torch.nonzero(label == 1.0, as_tuple=True)[0]

        if positive_idx.numel() == 0:
            return torch.tensor(0.0, device=emb_A.device, dtype=emb_A.dtype)

        log_prob_ab = F.log_softmax(sim_matrix, dim=1)
        pos_log_prob_ab = torch.diag(log_prob_ab)[positive_idx]
        loss_ab = -pos_log_prob_ab.mean()

        # Symétrisation : calculer également la perte dans l'autre sens.
        log_prob_ba = F.log_softmax(sim_matrix.T, dim=1)
        pos_log_prob_ba = torch.diag(log_prob_ba)[positive_idx]
        loss_ba = -pos_log_prob_ba.mean()

        return (loss_ab + loss_ba) / 2.0

def calculate_step_loss(model: nn.Module, criterion: nn.Module, img_A, meta_A, img_B, meta_B, labels):
    emb_A = model.forward(img_A, meta_A)
    emb_B = model.forward(img_B, meta_B)
    return criterion(emb_A, emb_B, labels)


def evaluate_loss(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    loss_total = 0.0

    with torch.no_grad():
        for img_A, meta_A, img_B, meta_B, labels in dataloader:
            img_A, meta_A = img_A.to(device), meta_A.to(device)
            img_B, meta_B = img_B.to(device), meta_B.to(device)
            labels = labels.to(device)

            loss = calculate_step_loss(model, criterion, img_A, meta_A, img_B, meta_B, labels)
            loss_total += loss.item() * img_A.size(0)

    return loss_total / len(dataloader.dataset)