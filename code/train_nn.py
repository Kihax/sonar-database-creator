import copy
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader

from lib.file_management import get_tree_from_file
from lib.ReadDatabaseImagette import ReadDatabaseImagette
from lib.SiameseSonarNetwork import SiameseSonarNetwork
from lib.SonarPairDataset import SonarPairDataset
from lib.VisualiseurResultatSiamois import VisualiseurResultatsSiamois
from lib.FuzzyDistanceClassifier import FuzzyDistanceClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_NAME = "S3CNN-No-Meta"
DATASET_NAME = "Grid-All-eq-sf100-wn"
EPOCHS = 5
DATASET_FILE = f"{DATASET_NAME}.nc"
INFO= "o-h"

CONFIG = {
    "dataset_file": DATASET_FILE,
    "dataset_folder": "../",
    "target_size": (3000, 100),
    "batch_size": 16,
    "learning_rate": 1e-4,
    "margin": 1.0,
    "epochs": EPOCHS,
    "seed": 42,
    "train_ratio": 0.60,
    "val_ratio": 0.20,
    "test_ratio": 0.20,
    "show_images": True,
    # Configuration floue (N catégories)
    "fuzzy_categories": [
        "Sûr (Même Objet)",
        "Presque Sûr (Même Objet)",
        "Moyennement Sûr / Incertain",
        "Presque Sûr (Différent)",
        "Sûr (Différent)",
    ],
    # Les chemins sont générés dynamiquement via f-strings
    "history_csv": ROOT_DIR / f"epochs/{MODEL_NAME}-{EPOCHS}-{DATASET_NAME}-{INFO}.csv",
    "checkpoint_path": ROOT_DIR / f"model/{MODEL_NAME}-{EPOCHS}-{DATASET_NAME}-{INFO}.pt",
    "model": {
        "depth": 3,
        "base_channels": 16,
        "kernel_size": 3,
        "fc_hidden_dim": 256,
        "embedding_dim": 128,
        "meta_feature_count": 0,
        "depth_normalization": 50.0,
        "detection_range_normalization": 1.0,
        "roll_normalization": 30.0,
    },
}


# ==========================================
# FONCTIONS UTILITAIRES DE TRAIN / EVAL
# ==========================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(groups: dict, config: dict, batch_size: int | None = None, shuffle: bool = False):
    target_size = tuple(config.get("target_size", (256, 256)))
    effective_batch_size = batch_size if batch_size is not None else int(config.get("batch_size", 16))
    dataset = SonarPairDataset(groups, target_size=target_size, meta_config=config.get("model", {}))
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    return loader, dataset


def compute_predictions(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    distances_list = []
    labels_list = []

    with torch.no_grad():
        for img_A, meta_A, img_B, meta_B, labels in dataloader:
            img_A, meta_A = img_A.to(device), meta_A.to(device)
            img_B, meta_B = img_B.to(device), meta_B.to(device)

            distances = model(img_A, meta_A, img_B, meta_B)
            distances_list.append(distances.cpu().numpy())
            labels_list.append(labels.numpy())

    return np.concatenate(distances_list), np.concatenate(labels_list)


def evaluate_loss(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    loss_total = 0.0

    with torch.no_grad():
        for img_A, meta_A, img_B, meta_B, labels in dataloader:
            img_A, meta_A = img_A.to(device), meta_A.to(device)
            img_B, meta_B = img_B.to(device), meta_B.to(device)
            labels = labels.to(device)

            distances = model(img_A, meta_A, img_B, meta_B)
            loss = criterion(distances, labels)
            loss_total += loss.item() * img_A.size(0)

    return loss_total / len(dataloader.dataset)


def find_best_threshold(distances: np.ndarray, labels: np.ndarray, min_val: float = 0.0, max_val: float = 2.0, step: float = 0.05):
    best_accuracy = 0.0
    best_threshold = min_val

    for threshold in np.arange(min_val, max_val, step):
        predictions = (distances < threshold).astype(float)
        accuracy = (predictions == labels).mean() * 100
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy


def print_confusion_matrix(true_labels: np.ndarray, predictions: np.ndarray, title: str):
    cm = confusion_matrix(true_labels, predictions, labels=[0.0, 1.0])
    tn, fp, fn, tp = cm.ravel()

    print(f"  Vrais Négatifs (Différents bien classés) : {tn}")
    print(f"  Faux Positifs (Différents confondus)     : {fp}")
    print(f"  Faux Négatifs (Mêmes non reconnus)       : {fn}")
    print(f"  Vrais Positifs (Mêmes bien classés)      : {tp}")

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Différents (0)", "Même Objet (1)"])
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")
    plt.title(title)
    plt.show()


def aggregate_cases(distances: np.ndarray, labels: np.ndarray, predictions: np.ndarray):
    cases = []
    count_vp = count_vn = count_fp = count_fn = 0

    for index, (dist, actual, pred) in enumerate(zip(distances, labels, predictions)):
        cases.append((index, dist, actual, pred))
        if pred == 1.0 and actual == 1.0:
            count_vp += 1
        elif pred == 0.0 and actual == 0.0:
            count_vn += 1
        elif pred == 1.0 and actual == 0.0:
            count_fp += 1
        elif pred == 0.0 and actual == 1.0:
            count_fn += 1

    cases.sort(key=lambda x: x[2] == x[3])
    return cases, count_vp, count_vn, count_fp, count_fn


def save_history_to_csv(history_rows: list[dict], output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "train_loss", "val_loss", "lr"]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def display_split_results(
    split_name: str,
    distances: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    fuzzy_classifier: FuzzyDistanceClassifier | None = None,
    dataset=None,
    show_images: bool = False,
):
    predictions = (distances < threshold).astype(float)
    accuracy = (predictions == labels).mean() * 100

    print(f"\n--- 📊 {split_name.upper()} ---")
    print(f"Seuil binaire appliqué            : {threshold:.2f}")
    print(f"➡️ PRÉCISION GLOBALE SUR {split_name.upper()} : {accuracy:.1f}%")

    # Évaluation floue avec taux d'exactitude par catégorie
    if fuzzy_classifier is not None:
        fuzzy_stats = {
            cat: {"total": 0, "correct": 0} for cat in fuzzy_classifier.categories
        }

        for d, label in zip(distances, labels):
            res = fuzzy_classifier.predict(d)
            cat = res["dominant_category"]
            fuzzy_stats[cat]["total"] += 1

            # Règle de décision binaire implicite de la catégorie floue :
            # Si la catégorie est du côté "Même" (< S) ou "Différent" (> S)
            # On vérifie si la prédiction binaire correspond au vrai label
            pred_binaire = 1.0 if d < threshold else 0.0
            if pred_binaire == label:
                fuzzy_stats[cat]["correct"] += 1

        print("\n  🧠 Répartition et Précision par Catégorie Floue :")
        for cat, stats in fuzzy_stats.items():
            total_cat = stats["total"]
            pct_volume = (total_cat / len(distances)) * 100 if len(distances) > 0 else 0

            if total_cat > 0:
                cat_acc = (stats["correct"] / total_cat) * 100
                acc_str = f"| Bonnes prédictions : {cat_acc:5.1f}% ({stats['correct']}/{total_cat})"
            else:
                acc_str = "| Aucune donnée"

            print(f"    ├─ {cat:<30} : {total_cat:4d} couples ({pct_volume:5.1f}%) {acc_str}")

        # Affichage d'un échantillon
        print("\n  🔍 Exemples d'interprétation floue (3 premiers couples) :")
        for i in range(min(3, len(distances))):
            res = fuzzy_classifier.predict(distances[i])
            print(
                f"    - Couple #{i+1} | Dist: {distances[i]:.3f} -> {res['dominant_category']} (Certitude: {res['confidence']}%)"
            )

    cases, vp, vn, fp, fn = aggregate_cases(distances, labels, predictions)
    print(f"\nTotal {split_name.title()} : {len(labels)} couples")
    print(f"  ├─ Corrects : {vp} VP (Mêmes) et {vn} VN (Diffs)")
    print(f"  └─ Erreurs  : {fp} FP (Confondus) et {fn} FN (Ratés)")

    print_confusion_matrix(labels, predictions, f"Matrice de Confusion - {split_name.title()}\nSeuil: {threshold:.2f}")

    if show_images and dataset is not None:
        display_cases = [(case[0], case[1]) for case in cases]
        print(f"Ouvrez la fenêtre Matplotlib pour le JEU DE {split_name.upper()}.")
        VisualiseurResultatsSiamois(display_cases, dataset, threshold)

    return predictions, accuracy


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, distance: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        distance = torch.clamp(distance, min=1e-6, max=100.0)
        loss_pos = label * torch.pow(distance, 2)
        loss_neg = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        loss = torch.mean(loss_pos + loss_neg)

        if torch.isnan(loss):
            print("⚠️  Warning: NaN detected in loss, replacing with safe value")
            loss = torch.tensor(1.0, device=distance.device, dtype=distance.dtype)

        return loss


def prepare_datasets(config: dict):
    print("--- Début du chargement de la base de données Sonar ---")

    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception as error:
        print(f"⚠️  Impossible de changer la stratégie de partage multiprocessing: {error}")

    set_seed(int(config.get("seed", 42)))

    dt = get_tree_from_file(config["dataset_file"], config["dataset_folder"])
    dts = [dt]

    rdi = ReadDatabaseImagette(dts)
    rdi.extract()
    global_imagettes = rdi.pos_imagette

    total_imagettes = sum(len(images) for images in global_imagettes.values())
    print(f"Nombre total d'objets imagettes : {total_imagettes}")

    matching_groups = {coord: images for coord, images in global_imagettes.items() if len(images) >= 2}
    all_coords = list(matching_groups.keys())
    random.shuffle(all_coords)

    total_groups = len(all_coords)
    train_groups_count = int(config.get("train_ratio", 0.80) * total_groups)
    val_groups_count = int(config.get("val_ratio", 0.10) * total_groups)

    train_coords = all_coords[:train_groups_count]
    val_coords = all_coords[train_groups_count : train_groups_count + val_groups_count]
    test_coords = all_coords[train_groups_count + val_groups_count :]

    train_groups = {coord: matching_groups[coord] for coord in train_coords}
    val_groups = {coord: matching_groups[coord] for coord in val_coords}
    test_groups = {coord: matching_groups[coord] for coord in test_coords}

    print("\nRépartition des points d'intérêt (Objets physiques) :")
    print(f"  ├─ Train : {len(train_groups)} objets")
    print(f"  ├─ Val   : {len(val_groups)} objets")
    print(f"  └─ Test  : {len(test_groups)} objets")

    return train_groups, val_groups, test_groups


def main(config: dict | None = None):
    config = config or CONFIG
    print("Configuration du training :")
    for key, value in config.items():
        if key == "model":
            print("  model:")
            for model_key, model_value in value.items():
                print(f"    - {model_key}: {model_value}")
        else:
            print(f"  - {key}: {value}")

    train_groups, val_groups, test_groups = prepare_datasets(config)

    train_loader, train_dataset = build_dataloader(train_groups, config, shuffle=True)
    val_loader, val_dataset = build_dataloader(val_groups, config, shuffle=False)
    test_loader, test_dataset = build_dataloader(test_groups, config, shuffle=False)

    print("\nNombre final de couples générés :")
    print(f"  ├─ Entraînement (Train) : {len(train_dataset)} couples")
    print(f"  ├─ Validation (Val)     : {len(val_dataset)} couples")
    print(f"  └─ Évaluation (Test)    : {len(test_dataset)} couples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseSonarNetwork(config.get("model", {})).to(device)
    criterion = ContrastiveLoss(margin=float(config.get("margin", 1.0)))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-4)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_model_weights = None
    epochs = int(config.get("epochs", 15))

    history_rows = []
    print(f"\n--- 🏋️ Début de l'entraînement ({epochs} époques) ---")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0.0

        for img_A, meta_A, img_B, meta_B, labels in train_loader:
            img_A, meta_A = img_A.to(device), meta_A.to(device)
            img_B, meta_B = img_B.to(device), meta_B.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            distances = model(img_A, meta_A, img_B, meta_B)
            loss = criterion(distances, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  Batch skipped : loss invalide = {loss.item()}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item() * img_A.size(0)

        epoch_train_loss /= len(train_loader.dataset)
        epoch_val_loss = evaluate_loss(model, val_loader, criterion, device)

        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            msg_save = " 💾 [Meilleur modèle mis à jour]"
        else:
            msg_save = ""

        history_rows.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 6),
            "val_loss": round(epoch_val_loss, 6),
            "lr": round(current_lr, 8),
        })
        save_history_to_csv(history_rows, config.get("history_csv", ROOT_DIR / "training_history.csv"))

        print(f"Époque {epoch:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | LR: {current_lr:.6f}{msg_save}")

    if best_model_weights is not None:
        print(f"\n🔄 Chargement des meilleurs poids (Val Loss min: {best_val_loss:.4f}) pour l'évaluation finale.")
        model.load_state_dict(best_model_weights)

    print("\n--- 📊 CALCUL ET RECHERCHE DU SEUIL OPTIMAL ET INITIALISATION FLOU ---")
    val_distances, val_labels = compute_predictions(model, val_loader, device)
    best_threshold, best_accuracy = find_best_threshold(val_distances, val_labels)

    print(f"Seuil de décision binaire optimal (Validation) : {best_threshold:.2f}")
    print(f"➡️ POURCENTAGE DE BONS RÉSULTATS (VAL)          : {best_accuracy:.1f}%")

    pos_distances = val_distances[val_labels == 1.0]
    neg_distances = val_distances[val_labels == 0.0]
    max_d = max(np.max(val_distances), 1.2) if len(val_distances) > 0 else 1.2

    if len(pos_distances) > 0:
        print(f"Moyenne des distances (Même objet)            : {np.mean(pos_distances):.4f}")
    if len(neg_distances) > 0:
        print(f"Moyenne des distances (Objets diff)            : {np.mean(neg_distances):.4f}")

    # Initialisation du classifieur flou
    fuzzy_classifier = FuzzyDistanceClassifier(
        categories=config.get("fuzzy_categories", []),
        min_dist=0.0,
        max_dist=1.0  # Marge de la Contrastive Loss
    )

    display_split_results("train", *compute_predictions(model, train_loader, device), best_threshold, fuzzy_classifier, train_dataset, show_images=False)
    display_split_results(
        "validation", val_distances, val_labels, best_threshold, fuzzy_classifier, val_dataset, show_images=bool(config.get("show_images", False))
    )

    print("\n--- 💾 ENREGISTREMENT DU MODÈLE ---")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "best_threshold": best_threshold,
        "best_accuracy": best_accuracy,
    }
    checkpoint_path = config.get("checkpoint_path", ROOT_DIR / "basic_100.pt")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"✅ Modèle sauvegardé sous : '{checkpoint_path}'")

    print("\n--- 🔒 ÉVALUATION SUR LE JEU DE TEST (INÉDIT) ---")
    test_distances, test_labels = compute_predictions(model, test_loader, device)
    display_split_results(
        "test", test_distances, test_labels, best_threshold, fuzzy_classifier, test_dataset, show_images=bool(config.get("show_images", False))
    )

    print("\n--- Fin de l'exécution complète ---")


if __name__ == "__main__":
    main()