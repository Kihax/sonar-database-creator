"""
Ce script permet de mesurer la similarité entre des paires d'imagettes sonar en utilisant un modèle de réseau de neurones.
"""

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
DATABASE_PATH = "../sensitivity-6areas-100-4t-300-30.nc" 
CHECKPOINT_PATH = ROOT_DIR / "model/S3CNN-No-Meta-Grid-All-eq-sf100-wn.pt"

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

TARGET_SIZE = (3000, 100)

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

    # 2. Chargement de la base d'imagettes augmentées
    print(f"\n--- Chargement de la base d'images augmentées : {DATABASE_PATH} ---")
    dt = get_tree_from_file(DATABASE_PATH, "")
    rdi = ReadDatabaseImagette([dt])
    rdi.extract()
    global_imagettes = rdi.pos_imagette

    # 3. Inférence et regroupement par objet physique
    distances_par_objet = {}
    all_augmentations_distances = []
    
    print("\n--- Calcul des distances pour les mêmes objets (Augmentations) ---")
    
    with torch.no_grad():
        for coord, imagettes in global_imagettes.items():
            if len(imagettes) < 2:
                continue
            
            single_group = {coord: imagettes}
            group_dataset = SonarPairDataset(single_group, target_size=TARGET_SIZE, meta_config=MODEL_CONFIG)
            
            distances_par_objet[coord] = []
            
            for idx in range(len(group_dataset)):
                img_A, meta_A, img_B, meta_B, label = group_dataset[idx]
                
                if label == 1.0:
                    img_A_t = img_A.unsqueeze(0).to(device)
                    meta_A_t = meta_A.unsqueeze(0).to(device)
                    img_B_t = img_B.unsqueeze(0).to(device)
                    meta_B_t = meta_B.unsqueeze(0).to(device)
                    
                    distance = model(img_A_t, meta_A_t, img_B_t, meta_B_t).item()
                    
                    distances_par_objet[coord].append(distance)
                    all_augmentations_distances.append(distance)
                    
                    # --- AFFICHAGE DE CHAQUE COUPLE ---
                    if VISUALIZE_EACH_PAIR:
                        # Dé-packaging des images 2D (Canal 0 = Intensité Sonar brute)
                        sonar_A = img_A[0].cpu().numpy()
                        sonar_B = img_B[0].cpu().numpy()
                        
                        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                        status = "SIMILAIRE" if distance < saved_threshold else "DIFFÉRENT"
                        color_status = "green" if distance < saved_threshold else "red"
                        
                        # Titre de la figure globale avec la distance
                        fig.suptitle(
                            f"Position: {coord}\n"
                            f"Paire {idx+1} | Distance Siamois : {distance:.4f} (Seuil: {saved_threshold:.2f}) -> {status}",
                            color=color_status, fontsize=12, fontweight='bold'
                        )
                        
                        axes[0].imshow(sonar_A, cmap='gray', aspect='auto')
                        axes[0].set_title("Imagette A")
                        axes[0].axis('off')
                        
                        axes[1].imshow(sonar_B, cmap='gray', aspect='auto')
                        axes[1].set_title("Imagette B")
                        axes[1].axis('off')
                        
                        plt.tight_layout()
                        plt.show() # Le script se met en pause ici jusqu'à la fermeture de la fenêtre graphique

    # =====================================================================
    # 4. COMPARAISON ET AFFICHAGE DES RÉSULTATS REGROUPÉS
    # =====================================================================
    print("\n" + "="*80)
    print(f"{'COMPARAISON DES DISTANCES PAR OBJET PHYSIQUE':^80}")
    print("="*80)
    print(f"Seuil de décision du modèle : {saved_threshold:.4f}\n")
    
    coords_list = list(distances_par_objet.keys())
    plot_data = []
    labels_plot = []
    
    for idx, coord in enumerate(coords_list):
        dists = distances_par_objet[coord]
        if not dists:
            continue
            
        mean_dist = np.mean(dists)
        max_dist = np.max(dists)
        min_dist = np.min(dists)
        std_dist = np.std(dists)
        recon_rate = (np.array(dists) < saved_threshold).mean() * 100
        
        print(f"📍 Objet {idx+1} | Coordonnées: (E: {coord[0]:.1f}, N: {coord[1]:.1f})")
        print(f"   ├─ Nombre de paires comparées : {len(dists)}")
        print(f"   ├─ Distance moyenne           : {mean_dist:.4f} (std: {std_dist:.4f})")
        print(f"   ├─ Extrêmes [Min - Max]       : [{min_dist:.4f} - {max_dist:.4f}]")
        print(f"   └─ Taux de reconnaissance     : {recon_rate:.1f}% similaires (sous le seuil)")
        print("-" * 80)
        
        plot_data.append(dists)
        labels_plot.append(f"Objet {idx+1}\n({coord[0]:.1f})")

    # =====================================================================
    # 5. VISUALISATION GRAPHIQUE FINALE (Matplotlib)
    # =====================================================================
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.boxplot(plot_data, labels=labels_plot, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2))
    plt.axhline(y=saved_threshold, color='r', linestyle='--', linewidth=1.5, label=f'Seuil ({saved_threshold:.2f})')
    plt.title("Dispersion des distances par objet physique\n(Plus c'est bas, plus c'est similaire)")
    plt.ylabel("Distance calculée par le réseau")
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(all_augmentations_distances, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=saved_threshold, color='r', linestyle='--', linewidth=1.5, label=f'Seuil ({saved_threshold:.2f})')
    plt.title("Distribution globale des distances\n(Toutes augmentations confondues)")
    plt.xlabel("Distance")
    plt.ylabel("Nombre de paires")
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()