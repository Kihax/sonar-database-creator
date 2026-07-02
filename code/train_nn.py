from lib.file_management import get_files, get_tree_from_file
from lib.DataManagement import DataManagement
from lib.Sonar import Sonar
from matplotlib import pyplot as plt
from lib.Point import Point
from lib.DatabaseCreatorImagette import DatabaseCreatorImagette
from lib.ReadDatabasePing import ReadDatabasePing
from lib.vizualize_imagette import vizualize_imagette
from lib.Imagette import Imagette
from typing import List
import numpy as np
from lib.ReadDatabaseImagette import ReadDatabaseImagette
from lib.ScrollImageViewer import ScrollImageViewer
import math
from lib.SiameseSonarNetwork import SiameseSonarNetwork
from lib.SonarPairDataset import SonarPairDataset
from torch.utils.data import DataLoader, random_split
import torch

print("--- Début du chargement de la base de données Sonar ---")
    
    # Récupération et extraction via votre code original
dt = get_tree_from_file("database_interest_point.nc", "../")
dts = [dt]

rdi = ReadDatabaseImagette(dts)
rdi.extract()
global_imagettes = rdi.pos_imagette
    
    # Simulation d'un point cible pour utiliser votre fonction de filtrage géographique
    # (Remplacez par votre logique ou passez directement tout le dictionnaire si pertinent)
print("Filtrage des groupes pertinents...")
test_point = Point(eastern=500000.0, northern=5000000.0) # À adapter à vos vraies coordonnées UTM
    
    # On extrait les groupes qui contiennent au moins 2 imagettes d'un même objet commun
matching_groups = rdi.get_groups_with_n_imagettes_containing_point(
    global_imagettes=global_imagettes, 
    target_point=test_point, 
    n=2
)
    
# Si vous préférez utiliser TOUS vos groupes contenant au moins 2 images sans filtrer par un point unique,
# décommentez la ligne suivante :
# matching_groups = {coord: imgs for coord, imgs in global_imagettes.items() if len(imgs) >= 2}

if not matching_groups:
    print("⚠️ Aucun groupe ne remplit les critères. Utilisation de la base complète pour le test.")
    matching_groups = {coord: imgs for coord, imgs in global_imagettes.items() if len(imgs) >= 2}

# Instanciation du Dataset global
try:
    dataset = SonarPairDataset(matching_groups, target_size=(256, 256))
    print(f"🎉 Dataset créé avec succès ! Nombre total de paires (Positives + Négatives) : {len(dataset)}")
        
    # Découpage classique Entraînement (80%) / Validation (20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
    # Création des DataLoaders de l'environnement PyTorch
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
        
        # --- PHASE DE TEST / VÉRIFICATION DU PIPELINE ---
    print("\n--- Test d'un passage de batch dans le modèle ---")
    model = SiameseSonarNetwork()
        
        # Choix du processeur disponible (GPU Cuda / MPS Apple / CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
        
    # Extraction du tout premier batch disponible
    for img_A, meta_A, img_B, meta_B, labels in train_loader:
        # Envoi des tenseurs sur le processeur graphique
        img_A, meta_A = img_A.to(device), meta_A.to(device)
        img_B, meta_B = img_B.to(device), meta_B.to(device)
            
        print(f"Format Tensor Image A : {img_A.shape}")  # Doit afficher torch.Size([16, 3, 256, 256])
        print(f"Format Tensor Meta A  : {meta_A.shape}")   # Doit afficher torch.Size([16, 3])
        print(f"Format Tenseur Labels : {labels.shape}")   # Doit afficher torch.Size([16])
            
        # Exécution (Forward pass) sans calcul de gradients
        with torch.no_grad():
            distances = model(img_A, meta_A, img_B, meta_B)
            print(f"Sortie du réseau (Distances calculées) : {distances.shape}") # torch.Size([16])
            print("Valeurs des premières distances obtenues :\n", distances.cpu().numpy()[:5])
            
        break # Fin de la vérification, le pipeline fonctionne de bout en bout !
            
except Exception as e:
    print(f"❌ Erreur lors de l'initialisation du pipeline PyTorch : {e}")