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
import numpy as np
from skimage import exposure
from scipy.ndimage import median_filter
from scipy.ndimage import uniform_filter
import cv2

def lee_filter(img, size=5):
    """
    Applique un filtre de Lee pour réduire le speckle (bruit sonar/radar)
    """
    img_mean = uniform_filter(img, size)
    img_sqr_mean = uniform_filter(img**2, size)
    img_variance = img_sqr_mean - img_mean**2
    overall_variance = np.var(img)
    
    img_variance[img_variance == 0] = 1e-5
    k = img_variance / (img_variance + overall_variance)
    
    img_filtered = img_mean + k * (img - img_mean)
    return np.clip(img_filtered, 0.0, 1.0)

def extract_sonar_regions_exact(img_clean, shadow_block_size=101, shadow_constant=12, highlight_percentile=95, 
                                kernel_w=5, kernel_h=2, min_region_size=60):
    """
    Détecte les formes et remplit les bounding boxes de manière semi-transparente.
    """
    img_8u = (img_clean * 255).astype(np.uint8)
    
    # 1. Seuillages bruts
    shadows_raw = cv2.adaptiveThreshold(
        img_8u, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, shadow_block_size, shadow_constant
    )
    thresh_high = np.percentile(img_8u, highlight_percentile)
    _, highlights_raw = cv2.threshold(img_8u, thresh_high, 255, cv2.THRESH_BINARY)

    # 2. Application de l'ouverture morphologique (kernel_w, kernel_h)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    shadows_clean = cv2.morphologyEx(shadows_raw, cv2.MORPH_OPEN, kernel_clean)
    highlights_clean = cv2.morphologyEx(highlights_raw, cv2.MORPH_OPEN, kernel_clean)

    # 3. Masquage de la colonne d'eau centrale
    width = shadows_clean.shape[1]
    center_start, center_end = int(width * 0.38), int(width * 0.62)
    shadows_clean[:, center_start:center_end] = 0
    highlights_clean[:, center_start:center_end] = 0

    combined_mask = cv2.bitwise_or(shadows_clean, highlights_clean)

    kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_big)

    # 4. Analyse des composantes connexes (regionprops)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask)
    
    base_rgb = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2BGR)
    overlay = base_rgb.copy()

    objs = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area >= min_region_size:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            cx = int(x + w / 2)
            cy = int(y + h / 2)
            objs.append([cx, cy])
            
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), -1)
        
    output_rgb = cv2.addWeighted(base_rgb, 0.7, overlay, 0.3, 0)

    return output_rgb, objs

folder_path = "../refactored_data/"
files = get_files("./refactored_data/")

total_pings_counter = 0
objs_geo_x = []
objs_geo_y = []
ship_x = []
ship_y = []
ship_headings = []

for i, filename in enumerate(files):
    dts = [get_tree_from_file(filename, folder_path)]

    rd = ReadDatabasePing(dts)
    sonarPings: List[SonarPing] = rd.get_sonarPings()
    
    if not sonarPings:
        continue
        
    waterfall = []
    waterfall_corr = []    

    for j in range(len(sonarPings)):
        ping = sonarPings[j]
        waterfall.append(ping.sample)

        sound_speed = ping.sound_speed
        delta_time = ping.delta_time

        sample_stbd = np.array(ping.get_starboard())
        sample_port = np.array(ping.get_port())
        
        # Récupération position navire (X=Longitude ou Est, Y=Latitude ou Nord)
        # On suppose ici que point_ship possède des attributs x, y ou lat, lon
        if hasattr(ping.point_ship, 'x') and hasattr(ping.point_ship, 'y'):
            ship_x.append(ping.point_ship.x)
            ship_y.append(ping.point_ship.y)
        else:
            # S'il s'agit de coordonnées d'un objet Point classique (par exemple .longitude / .latitude)
            ship_x.append(ping.point_ship.eastern)
            ship_y.append(ping.point_ship.northern)
            
        # Récupération du cap (Heading) du ping (généralement en degrés ou radians)
        # Si absent, on peut le calculer à partir de la différence de position
        if hasattr(ping, 'heading'):
            ship_headings.append(ping.heading)
        else:
            ship_headings.append(0.0) # Valeur par défaut si non disponible

        indices = np.arange(len(sample_stbd))
        gain = (sound_speed * delta_time * indices / 2) ** 2
        
        sample_stbd_corr = sample_stbd * gain
        sample_port_corr = sample_port * gain

        sample_corr = np.concatenate((np.flip(sample_port_corr), sample_stbd_corr))
        waterfall_corr.append(sample_corr)
    
    current_file_pings = len(sonarPings)
    num_samples = len(sonarPings[0].sample)

    waterfall_matrix = np.array(waterfall_corr, dtype=float)
    
    p1, p99 = np.percentile(waterfall_matrix, (1, 99))
    waterfall_stretched = np.clip((waterfall_matrix - p1) / (p99 - p1), 0.0, 1.0)

    waterfall_clean = lee_filter(waterfall_stretched, size=7)
    
    y_start = total_pings_counter
    y_end = total_pings_counter + current_file_pings
    num_samples_corr = len(waterfall_corr[0])
    
    img_extent_raw = [0, num_samples, y_end, y_start]
    img_extent_corr = [0, num_samples_corr, y_end, y_start]

    image_bounding_boxes, objs = extract_sonar_regions_exact(
        waterfall_clean, 
        shadow_block_size=101, 
        shadow_constant=80,          
        highlight_percentile=95,     
        kernel_w=2,                  
        kernel_h=2,                  
        min_region_size=100           
    )

    
    objs_global_for_plot = []

    for obj in objs:
        x = obj[0]
        y_local = obj[1]

        eastern, nothern = sonarPings[y_local].get_position_from_index(x)

        y_global = y_start + y_local

        objs_geo_y.append(nothern)
        objs_geo_x.append(eastern)

    total_pings_counter += current_file_pings


fig2, ax3 = plt.subplots(figsize=(7, 7))
ax3.set_title(f"Ship Track & Heading Vectors\nFile: {filename}")

ax3.scatter(objs_geo_x, objs_geo_y, color='lime', edgecolors='black', marker='^', s=40, label="Detected Objects", zorder=4)

ship_x = np.array(ship_x)
ship_y = np.array(ship_y)
ship_headings = np.array(ship_headings)

step = max(1, len(ship_x) // 20) # Affiche environ 20 flèches le long du parcours
for idx in range(0, len(ship_x), step):
    # Conversion du cap en composantes de vecteur (U, V)
    # /!\ Attention si ton heading est en degrés (on passe en radians)
    angle_rad = np.radians(ship_headings[idx])
        
    # En navigation, 0° est souvent le Nord (axe Y). 
    # Si c'est le cas : U = sin(angle), V = cos(angle)
    u = np.sin(angle_rad)
    v = np.cos(angle_rad)
        
    # Tracé d'une flèche directionnelle (Quiver) en rouge
    ax3.quiver(ship_x[idx], ship_y[idx], u, v, color='red', scale=15, width=0.005)


ax3.set_xlabel("Position X (m or lon)")
ax3.set_ylabel("Position Y (m or lat)")
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.axis('equal') # Garde les proportions géographiques 1:1
ax3.legend()
plt.tight_layout()

# Affichage des deux blocs
plt.show()
    
plt.close(fig2)
    
#total_pings_counter += current_file_pings