import torch
import copy


from lib.SonarPairDataset import build_dataloader, prepare_datasets
from lib.SiameseSonarNetwork import SiameseSonarNetwork
from lib.LossFunctions import ContrastiveLoss, evaluate_loss
from lib.utility_training import save_history_to_csv, ROOT_DIR, train, save_model, compute_predictions, set_seed
from lib.SiameseSonarResNet50 import SiameseSonarResNet50
from lib.AttentionSiameseSonarNetwork import AttentionSiameseSonarNetwork


#config based on last training
from train_nn_resnet50 import CONFIG

import matplotlib.pyplot as plt
import numpy as np

def plot_split_histograms(distances, labels, split_name, threshold=0.5, bins=30):
    """
    Affiche deux histogrammes des distances pour un split donné :
    1. Pour les labels positifs (label == 1)
    2. Pour les labels négatifs (label == 0)
    """
    # S'assurer que les données sont sous forme de tableaux NumPy
    distances = np.array(distances)
    labels = np.array(labels)

    # Séparation selon le label (1 = Positif / Similaire, 0 = Négatif / Dissimilaire)
    pos_distances = distances[labels == 1]
    neg_distances = distances[labels == 0]

    # Création d'une figure avec 2 sous-graphiques côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Distribution des distances - {split_name}", fontsize=14, fontweight="bold")

    # 1. Histogramme des Labels Positifs
    axes[0].hist(pos_distances, bins=bins, color="skyblue", edgecolor="black", alpha=0.7)
    axes[0].axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Seuil ({threshold})")
    axes[0].set_title("Labels Positifs (Paires similaires)")
    axes[0].set_xlabel("Distance")
    axes[0].set_ylabel("Nombre d'échantillons")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # 2. Histogramme des Labels Négatifs
    axes[1].hist(neg_distances, bins=bins, color="coral", edgecolor="black", alpha=0.7)
    axes[1].axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Seuil ({threshold})")
    axes[1].set_title("Labels Négatifs (Paires dissimilaires)")
    axes[1].set_xlabel("Distance")
    axes[1].set_ylabel("Nombre d'échantillons")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    
    model_name = CONFIG.get("model", {}).get("model_name", "") + "_" + CONFIG.get("dataset", "dataset")

    set_seed(CONFIG.get("seed", 42))
    train_groups, val_groups, test_groups = prepare_datasets({
        "epochs": 6,
        "learning_rate": 1e-4,
        "margin": 1.0,
        "model": {
            "model_name": "SiameseNetwork-o-img-No_Meta-cl_lecun",
            "depth": 3,
            "base_channels": 16,
        },
        "batch_size": 64,
        "dataset": "Grid-All-eq-sf100-wn",
        "dataset_folder": "../",
        "target_size": (3000, 100),
        "train_ratio": 0.60,
        "val_ratio": 0.20,
        "test_ratio": 0.20,
        "min_imagettes": 15,
    })

    train_loader, train_dataset = build_dataloader(train_groups, CONFIG, shuffle=True)
    val_loader, val_dataset = build_dataloader(val_groups, CONFIG, shuffle=False)
    test_loader, test_dataset = build_dataloader(test_groups, CONFIG, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionSiameseSonarNetwork(CONFIG.get("model", {})).to(device)
    print(ROOT_DIR / "model/" / f"{model_name}.pt")

    checkpoint = torch.load(ROOT_DIR / "model/" / f"{model_name}.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    threshold = float(checkpoint.get("best_threshold", 0.5))

    train_distances, train_labels = compute_predictions(model, train_loader, device)

    predictions = (train_distances < threshold).astype(float)
    accuracy = (predictions == train_labels).mean() * 100
    print(accuracy)

    val_distances, val_labels = compute_predictions(model, val_loader, device)

    predictions = (val_distances < threshold).astype(float)
    accuracy = (predictions == val_labels).mean() * 100
    print(accuracy)

    test_distances, test_labels = compute_predictions(model, test_loader, device)

    predictions = (test_distances < threshold).astype(float)
    accuracy = (predictions == test_labels).mean() * 100
    print(accuracy)

    dataset_splits = {
        "Train": (train_distances, train_labels),
        "Validation": (val_distances, val_labels),
        "Test": (test_distances, test_labels)
    }

    for split_name, (dist, lbl) in dataset_splits.items():
        plot_split_histograms(dist, lbl, split_name=split_name, threshold=threshold)

