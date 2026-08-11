import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from lib.SonarPairDataset import build_dataloader, prepare_datasets
from lib.SiameseSonarNetwork import SiameseSonarNetwork
from lib.LossFunctions import ContrastiveLoss, evaluate_loss
from lib.utility_training import save_history_to_csv, ROOT_DIR, train, save_model, compute_predictions, set_seed
from lib.SiameseSonarResNet50 import SiameseSonarResNet50
from lib.AttentionSiameseSonarNetwork import AttentionSiameseSonarNetwork
from SonarPairViewer import SonarPairViewer


# config based on last training
from train_nn_resnet50 import CONFIG


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

def plot_split_histograms(distances, labels, split_name, threshold=0.5, bins=30):
    """Affiche deux histogrammes des distances pour un split donné."""
    distances = np.array(distances)
    labels = np.array(labels)

    pos_distances = distances[labels == 1]
    neg_distances = distances[labels == 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Distribution des distances - {split_name}", fontsize=14, fontweight="bold")

    axes[0].hist(pos_distances, bins=bins, color="skyblue", edgecolor="black", alpha=0.7)
    axes[0].axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Seuil ({threshold})")
    axes[0].set_title("Labels Positifs (Paires similaires)")
    axes[0].set_xlabel("Distance")
    axes[0].set_ylabel("Nombre d'échantillons")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].hist(neg_distances, bins=bins, color="coral", edgecolor="black", alpha=0.7)
    axes[1].axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Seuil ({threshold})")
    axes[1].set_title("Labels Négatifs (Paires dissimilaires)")
    axes[1].set_xlabel("Distance")
    axes[1].set_ylabel("Nombre d'échantillons")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)

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
        "dataset": "Grid-All-eq-sf100-wn-a",
        "dataset_folder": "../",
        "target_size": (3000, 100),
        "train_ratio": 0.0,
        "val_ratio": 0.0,
        "test_ratio": 1.0,
        "min_imagettes": 0,
    })

    if(train_groups):
        train_loader, train_dataset = build_dataloader(train_groups, CONFIG, shuffle=True)
    if(val_groups):
        val_loader, val_dataset = build_dataloader(val_groups, CONFIG, shuffle=False)
    if(test_groups):
        test_loader, test_dataset = build_dataloader(test_groups, CONFIG, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionSiameseSonarNetwork(CONFIG.get("model", {})).to(device)

    checkpoint = torch.load(ROOT_DIR / "model/" / f"{model_name}.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    threshold = float(checkpoint.get("best_threshold", 0.5))

    # Calcul des prédictions
    #train_distances, train_labels = compute_predictions(model, train_loader, device)
    #val_distances, val_labels = compute_predictions(model, val_loader, device)
    test_distances, test_labels = compute_predictions(model, test_loader, device)

    dataset_splits = {
        #"Train": (train_distances, train_labels, train_dataset),
        #"Validation": (val_distances, val_labels, val_dataset),
        "Test": (test_distances, test_labels, test_dataset)
    }

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
        category="VP",    # 'FP', 'FN', 'VP', 'VN' ou 'ALL'
        page_size=3,     # Affiche 3 couples par page
        rotate=True,     # Rotation 90°
        split_name="Test"
    )