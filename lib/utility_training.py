import numpy as np
import torch
import random
import csv
from pathlib import Path
import copy
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

# Configuration pour éviter la fragmentation mémoire sous CUDA
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from lib.LossFunctions import evaluate_loss

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_history_to_csv(history_rows: list[dict], output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "train_loss", "val_loss", "lr"]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs: int = 15, print_progress: bool = True, best_model:bool=True):
    history_rows = []
    best_val_loss = float("inf")
    best_model_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0.0

        progress_bar = tqdm(
            train_loader, 
            desc=f"Époque {epoch:02d}/{epochs:02d}", 
            unit="batch",
            disable=not print_progress
        )

        for img_A, meta_A, img_B, meta_B, labels in progress_bar:
            img_A, meta_A = img_A.to(device, non_blocking=True), meta_A.to(device, non_blocking=True)
            img_B, meta_B = img_B.to(device, non_blocking=True), meta_B.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            emb_A = model.forward(img_A, meta_A)
            emb_B = model.forward(img_B, meta_B)
            
            loss = criterion(emb_A, emb_B, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  Batch skipped : loss invalide = {loss.item()}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_loss = loss.item()
            epoch_train_loss += batch_loss * img_A.size(0)

            if print_progress:
                progress_bar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})

            # 🛠️ FIX 1 : Libération explicite des tenseurs du batch pour libérer la VRAM
            del img_A, meta_A, img_B, meta_B, labels, emb_A, emb_B, loss

        epoch_train_loss /= len(train_loader.dataset)
        
        # Évaluation en mode no_grad
        epoch_val_loss = evaluate_loss(model, val_loader, criterion, device)

        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if best_model and epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            # 🛠️ FIX 2 : Déplacer le state_dict sur le CPU avant la copie pour ne pas surcharger la VRAM
            best_model_weights = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            msg_save = " 💾 [Meilleur modèle mis à jour]"
        else:
            msg_save = ""

        if not best_model:
            best_model_weights = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

        history_rows.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 6),
            "val_loss": round(epoch_val_loss, 6),
            "lr": round(current_lr, 8),
        })

        if print_progress:
            tqdm.write(f"Époque {epoch:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | LR: {current_lr:.6f}{msg_save}")

        # Nettoyage du cache CUDA entre les époques
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_model_weights, best_val_loss, history_rows

def save_model(checkpoint_path, model, recommanded_threshold):
    # Sauvegarde sur le CPU pour éviter tout problème de désérialisation GPU
    checkpoint = {
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "best_threshold": recommanded_threshold,
    }
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)

def compute_predictions(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    distances_list = []
    labels_list = []

    with torch.no_grad():
        for img_A, meta_A, img_B, meta_B, labels in tqdm(dataloader, desc="Calcul des prédictions", unit="batch"):
            img_A, meta_A = img_A.to(device, non_blocking=True), meta_A.to(device, non_blocking=True)
            img_B, meta_B = img_B.to(device, non_blocking=True), meta_B.to(device, non_blocking=True)

            emb_A = model(img_A, meta_A)
            emb_B = model(img_B, meta_B)

            emb_A = F.normalize(emb_A, p=2, dim=-1)
            emb_B = F.normalize(emb_B, p=2, dim=-1)

            distance = torch.pairwise_distance(emb_A, emb_B, p=2)

            distances_list.append(distance.cpu().numpy())
            labels_list.append(labels.numpy())

            del img_A, meta_A, img_B, meta_B, emb_A, emb_B, distance

    return np.concatenate(distances_list), np.concatenate(labels_list)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_overlapping_histogram(distances, labels, split_name, threshold=0.5, bins=30):
    """
    Affiche la distribution des distances des paires positives et négatives
    sur un même graphique avec transparence.
    """
    # Convertir en tableaux NumPy
    distances = np.array(distances)
    labels = np.array(labels)

    # Séparation selon le label (1 = Positif / Similaire, 0 = Négatif / Dissimilaire)
    pos_distances = distances[labels == 1]
    neg_distances = distances[labels == 0]

    # Création d'une figure unique
    plt.figure(figsize=(9, 5.5))

    # 1. Histogramme des paires positives
    plt.hist(
        pos_distances, 
        bins=bins, 
        color="skyblue", 
        edgecolor="navy", 
        alpha=0.6, 
        label="Positifs (Paires similaires)"
    )

    # 2. Histogramme des paires négatives (superposé)
    plt.hist(
        neg_distances, 
        bins=bins, 
        color="coral", 
        edgecolor="darkred", 
        alpha=0.6, 
        label="Négatifs (Paires dissimilaires)"
    )

    # 3. Ligne verticale du seuil
    plt.axvline(
        threshold, 
        color="red", 
        linestyle="--", 
        linewidth=2, 
        label=f"Seuil de décision ({threshold})"
    )

    # Mise en forme du graphique
    plt.title(f"Distribution des distances - {split_name}", fontsize=14, fontweight="bold")
    plt.xlabel("Distance", fontsize=11)
    plt.ylabel("Nombre d'échantillons", fontsize=11)
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    plt.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(labels, predictions, split_name):
    """
    Calcule et affiche la matrice de confusion détaillée (VP, VN, FP, FN).
    """
    cm = confusion_matrix(labels, predictions, labels=[1, 0])
    # Structure de confusion_matrix avec labels=[1, 0] :
    # [ [VP, FN],
    #   [FP, VN] ]
    
    tp, fn = cm[0, 0], cm[0, 1]
    fp, tn = cm[1, 0], cm[1, 1]

    print(f"\n--- Matrice de Confusion [{split_name}] ---")
    print(f"Vrais Positifs  (VP / TP) : {tp}")
    print(f"Vrais Négatifs  (VN / TN) : {tn}")
    print(f"Faux Positifs   (FP / FP) : {fp}")
    print(f"Faux Négatifs   (FN / FN) : {fn}")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Similaire (1)", "Dissimilaire (0)"],
        yticklabels=["Similaire (1)", "Dissimilaire (0)"],
        ax=ax
    )
    ax.set_title(f"Matrice de Confusion - {split_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Prédiction du Modèle")
    ax.set_ylabel("Vrai Label (Vérité Terrain)")
    
    plt.tight_layout()
    plt.show()