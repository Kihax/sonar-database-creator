"""
    Ce script permet d'analyser les performances du modèles et d analyser précisément chaque cas pour chaque couple
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from lib.SonarPairDataset import build_dataloader, prepare_datasets
from lib.get_model import get_model
from lib.utility_training import ROOT_DIR, compute_predictions, set_seed
from lib.SonarPairViewer import SonarPairViewer
from lib.utility_training import plot_overlapping_histogram, plot_confusion_matrix



# config based on last training
from configs_experience.experience16 import config as CONFIG


if __name__ == "__main__":
    name = CONFIG.get("name", "default_model_name")

    set_seed(CONFIG.get("seed", 43))
    train_groups, val_groups, test_groups = prepare_datasets(CONFIG["data"])

    if(train_groups):
        train_loader, train_dataset = build_dataloader(train_groups, CONFIG["data"], shuffle=True)
    if(val_groups):
        val_loader, val_dataset = build_dataloader(val_groups, CONFIG["data"], shuffle=False)
    if(test_groups):
        test_loader, test_dataset = build_dataloader(test_groups, CONFIG["data"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(CONFIG, device)

    print(ROOT_DIR / "model/" / f"{name}.pt")

    checkpoint = torch.load(ROOT_DIR / "model/" / f"{name}.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    threshold = float(checkpoint.get("best_threshold", 0.5))

    # Calcul des prédictions
    dataset_splits = {}
    if(train_groups):
        train_distances, train_labels = compute_predictions(model, train_loader, device)
        dataset_splits["Train"] = (train_distances, train_labels, train_dataset)
    if(val_groups):
        val_distances, val_labels = compute_predictions(model, val_loader, device)
        dataset_splits["Validation"] = (val_distances, val_labels, val_dataset)
    if(test_groups):
        test_distances, test_labels = compute_predictions(model, test_loader, device)
        dataset_splits["Test"] = (test_distances, test_labels, test_dataset)

    # Affichage des matrices de confusion et des histogrammes
    for split_name, (dist, lbl, ds) in dataset_splits.items():
        preds = (dist < threshold).astype(float)
        acc = (preds == lbl).mean() * 100
        print(f"\n================ Split: {split_name} (Accuracy: {acc:.2f}%) ================")
        
        # 1. Histogramme des distances
        plot_overlapping_histogram(dist, lbl, split_name=split_name, threshold=threshold)
        
        # 2. Matrice de confusion (VP, VN, FP, FN)
        plot_confusion_matrix(lbl, preds, split_name=split_name)

    # 3. Visualisation des couples d'images pour le jeu de Test (VP, VN, FP, FN)
    SonarPairViewer(
        dataset=test_dataset,
        distances=test_distances,
        labels=test_labels,
        threshold=threshold,
        category="ALL",    # 'FP', 'FN', 'VP', 'VN' ou 'ALL'
        page_size=3,     # Affiche 3 couples par page
        split_name="Test"
    )