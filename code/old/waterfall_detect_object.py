from lib.file_management import get_files, get_tree_from_file
from lib.DataManagement import DataManagement
from lib.Sonar import Sonar
from matplotlib import pyplot as plt
from lib.Point import Point
from lib.DatabaseCreator import DatabaseCreator
from lib.ReadDatabasePing import ReadDatabasePing
from lib.vizualize_imagette import vizualize_imagette
from typing import List
from lib.SonarPing import SonarPing
import cv2
import numpy as np
import math

folder_path = "../refactored_data_survey1/"
files = get_files("./refactored_data_survey1/")
print(files)

total_pings_counter = 0
search_window = 40  # Taille de la fenêtre (+/- échantillons) pour chercher le maximum réel du fond

for idx_file, filename in enumerate(files): 
    dts = [get_tree_from_file(filename, folder_path)]

    rd = ReadDatabasePing(dts)
    sonarPings: List[SonarPing] = rd.get_sonarPings()
    
    if not sonarPings:
        continue
        
    waterfall = []
    waterfall_stbd_coor = []
    waterfall_stbd = [] 
    seafloor_indices = [] # Pour stocker les indices ajustés du fond marin
    
    sound_speed = 1475
    dt = sonarPings[0].delta_time
    
    for j in range(len(sonarPings)):
        waterfall.append(sonarPings[j].sample)   
        sonarPing_stbd = sonarPings[j].get_starboard()
        
        # 1. Récupération de l'index de bathymétrie théorique
        index_bathy = sonarPings[j].get_index_depth_relative()
        
        # 2. Recalage sur le maximum local (sommet de l'écho de sol)
        start_idx = max(0, index_bathy - search_window)
        end_idx = min(len(sonarPing_stbd), index_bathy + search_window)
        search_zone = sonarPing_stbd[start_idx:end_idx]
        
        if len(search_zone) > 0:
            index_seafloor = start_idx + np.argmax(search_zone)
        else:
            index_seafloor = index_bathy
            
        seafloor_indices.append(index_seafloor)
        
        # 3. Calculs géométriques et correction avec l'index recalé
        depth = index_seafloor * sound_speed * dt / 2
        intensity_max = sonarPing_stbd[index_seafloor]
        sonarPing_corrected = []

        for i, sample in enumerate(sonarPing_stbd):
            if i > index_seafloor:
                r = sound_speed * dt * i / 2
                
                # Sécurité géométrique pour éviter une division par zéro ou un arccos impossible
                if r > 0 and (depth / r) <= 1.0:
                    cos_theta = depth / r
                else:
                    cos_theta = 1.0
                
                # On bride le cosinus au loin pour éviter que la division n'explose vers l'infini
                cos_limite = max(0.25, cos_theta)
                
                # Application de ton modèle empirique stabilisé
                val_corrigee = (sonarPing_stbd[i] / (cos_limite)) - intensity_max
                
                # Protection pour rester dans la dynamique d'affichage [0, 255]
                val_corrigee = np.clip(val_corrigee, 0, 255)
                sonarPing_corrected.append(val_corrigee)
            else:
                # Avant le fond marin, on garde le signal brut (colonne d'eau)
                sonarPing_corrected.append(sonarPing_stbd[i])
                
        waterfall_stbd_coor.append(sonarPing_corrected)
        waterfall_stbd.append(sonarPing_stbd) 

        
    
    # Génération de l'axe Y pour le tracé de la ligne du fond marin
    ping_numbers = list(range(len(sonarPings)))

    # --- Création de la figure avec deux axes côte à côte ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    
    # Axe 1 : Waterfall Original (Starboard)
    im1 = ax1.imshow(waterfall_stbd, vmax=250, vmin=0, cmap='gray', aspect="auto")
    ax1.plot(seafloor_indices, ping_numbers, color='red', linestyle='--', linewidth=1.2, alpha=0.6, label='Recalated Seafloor')
    ax1.set_title("Waterfall Starboard (Original)")
    ax1.set_ylabel("Cumulated Ping Number")
    ax1.set_xlabel("Samples")
    ax1.legend(loc='upper right')
    
    # Axe 2 : Waterfall Corrigé
    im2 = ax2.imshow(waterfall_stbd_coor, vmax=250, vmin=0, cmap='gray', aspect="auto")
    ax2.plot(seafloor_indices, ping_numbers, color='red', linestyle='--', linewidth=1.2, alpha=0.6, label='Recalated Seafloor')
    ax2.set_title("Waterfall Starboard (Corrected)")
    ax2.set_xlabel("Samples")
    ax2.legend(loc='upper right')
    
    # On force les limites de l'axe X pour éviter les déformations dues au tracé de la ligne
    ax1.set_xlim(0, len(sonarPing_stbd))
    ax2.set_xlim(0, len(sonarPing_stbd))

    plt.tight_layout()
    plt.show()