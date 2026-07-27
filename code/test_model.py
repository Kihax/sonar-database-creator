import torch
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

# --- AJOUT DE LA CORRECTION D'IMPORT ET DE SÉCURITÉ ICI ---
import torch.serialization
try:
    torch.serialization.add_safe_globals([
        np._core.multiarray.scalar,
        np.dtype
    ])
except AttributeError:
    pass
# ---------------------------------------------------------

from lib.ReadDatabaseImagette import ReadDatabaseImagette
from lib.SiameseSonarNetwork import SiameseSonarNetwork
from lib.SonarPairDataset import SonarPairDataset
from lib.file_management import get_tree_from_file

# =====================================================================
# CONFIGURATION
# =====================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = "../HP-Centered-100.nc"
CHECKPOINT_PATH = ROOT_DIR / "model/S3CNN-15-Grid-All-eq-sf100.pt"

# Active ou désactive l'affichage de CHAQUE couple d'images sonar en cours de route
VISUALIZE_EACH_PAIR = True

MODEL_CONFIG = {
    "depth": 3,
    "base_channels": 16,
    "kernel_size": 3,
    "fc_hidden_dim": 256,
    "embedding_dim": 128,
    "meta_feature_count": 0,
    "depth_normalization": 50.0,
    "detection_range_normalization": 1.0,
    "roll_normalization": 30.0,
}

TARGET_SIZE = (3000, 200)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil utilisé : {device}")

    # 1. Chargement du modèle
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Le fichier de modèle '{CHECKPOINT_PATH}' est introuvable.")
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = SiameseSonarNetwork(MODEL_CONFIG).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    saved_threshold = checkpoint.get("best_threshold", 1.0)

    # 2. Chargement de la base d'imagettes
    print(f"\n--- Chargement de la base d'images : {DATABASE_PATH} ---")
    dt = get_tree_from_file(DATABASE_PATH, "")
    rdi = ReadDatabaseImagette([dt])
    rdi.extract()
    global_imagettes = rdi.pos_imagette  # Dictionnaire complet

    # 3. Création du dataset GLOBAL (Mélange Positifs/Négatifs automatique)
    # En passant tout d'un coup, len(coords) >= 2 devient vrai et génère les couples aléatoires négatifs.
    dataset = SonarPairDataset(global_imagettes, target_size=TARGET_SIZE, meta_config=MODEL_CONFIG)
    print(f"Nombre total de paires générées (mélangées) : {len(dataset)}")

    pos_distances = []
    neg_distances = []
    correct_predictions = 0

    print("\n--- Début de l'inférence sur les couples mélangés ---")
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            img_A, meta_A, img_B, meta_B, label_tensor = dataset[idx]
            label = float(label_tensor.item()) # 1.0 = Même Objet, 0.0 = Objets Différents

            img_A_t = img_A.unsqueeze(0).to(device)
            meta_A_t = meta_A.unsqueeze(0).to(device)
            img_B_t = img_B.unsqueeze(0).to(device)
            meta_B_t = meta_B.unsqueeze(0).to(device)
            
            distance = model(img_A_t, meta_A_t, img_B_t, meta_B_t).item()
            
            # Classification selon le seuil
            pred_similaire = distance < saved_threshold
            is_correct = (pred_similaire and label == 1.0) or (not pred_similaire and label == 0.0)
            if is_correct:
                correct_predictions += 1

            if label == 1.0:
                pos_distances.append(distance)
                type_str = "Même Objet (Positif)"
            else:
                neg_distances.append(distance)
                type_str = "Objets Différents (Négatif)"

            status_prediction = "Bonne Décision ✅" if is_correct else "Erreur ❌"
            print(f"Paire {idx+1:03d} | Type: {type_str:<28} | Dist: {distance:.4f} (Seuil: {saved_threshold:.2f}) -> {status_prediction}")
            
            # --- AFFICHAGE FACULTATIF DE CHAQUE COUPLE ---
            if VISUALIZE_EACH_PAIR:
                sonar_A = img_A[0].cpu().numpy()
                sonar_B = img_B[0].cpu().numpy()
                
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                color_status = "green" if is_correct else "red"
                decision_str = "SIMILAIRE" if pred_similaire else "DIFFÉRENT"
                
                fig.suptitle(
                    f"Paire {idx+1} | Attendu: {int(label)} ({type_str}) | Prédit: {decision_str}\n"
                    f"Distance Siamois : {distance:.4f} -> {status_prediction}",
                    color=color_status, fontsize=12, fontweight='bold'
                )
                
                axes[0].imshow(sonar_A, cmap='gray', aspect='auto')
                axes[0].set_title("Imagette A")
                axes[0].axis('off')
                
                axes[1].imshow(sonar_B, cmap='gray', aspect='auto')
                axes[1].set_title("Imagette B")
                axes[1].axis('off')
                
                plt.tight_layout()
                plt.show()

    # =====================================================================
    # 4. AFFICHAGE DES RÉSULTATS ET STATISTIQUES GLOBALES
    # =====================================================================
    print("\n" + "="*80)
    print(f"{'BILAN DE L\'ÉVALUATION ALÉATOIRE':^80}")
    print("="*80)
    accuracy = (correct_predictions / len(dataset)) * 100
    print(f"Précision globale du modèle sur ce vrac : {accuracy:.2f}%")
    print(f"Nombre total de paires positives (Mêmes) : {len(pos_distances)}")
    print(f"Nombre total de paires négatives (Diffs) : {len(neg_distances)}")
    
    if pos_distances:
        print(f"Moyenne des distances pour les Mêmes Objets   : {np.mean(pos_distances):.4f} (std: {np.std(pos_distances):.4f})")
    if neg_distances:
        print(f"Moyenne des distances pour les Objets Différents : {np.mean(neg_distances):.4f} (std: {np.std(neg_distances):.4f})")
    print("-" * 80)

    # =====================================================================
    # 5. VISUALISATION GRAPHIQUE FINALE (Matplotlib)
    # =====================================================================
    plt.figure(figsize=(12, 6))
    
    # Boxplot comparatif Positifs vs Négatifs
    plt.subplot(1, 2, 1)
    plot_data = [pos_distances, neg_distances]
    plt.boxplot(plot_data, labels=["Mêmes Objets (1)", "Objets Différents (0)"], patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2))
    plt.axhline(y=saved_threshold, color='r', linestyle='--', linewidth=1.5, label=f'Seuil ({saved_threshold:.2f})')
    plt.title("Séparabilité des classes (Positifs vs Négatifs)")
    plt.ylabel("Distance calculée par le réseau")
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend()

    # Histogramme superposé
    plt.subplot(1, 2, 2)
    if pos_distances:
        plt.hist(pos_distances, bins=15, alpha=0.6, color='g', edgecolor='green', label='Mêmes Objets (1)')
    if neg_distances:
        plt.hist(neg_distances, bins=15, alpha=0.6, color='r', edgecolor='red', label='Objets Différents (0)')
    plt.axvline(x=saved_threshold, color='black', linestyle='--', linewidth=1.5, label=f'Seuil ({saved_threshold:.2f})')
    plt.title("Superposition des distributions de distances")
    plt.xlabel("Distance")
    plt.ylabel("Nombre de paires")
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()