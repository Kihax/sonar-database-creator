"""
Ce script créer une base de données augmentées en décallant les imagettes en translations, la translation n'étant pas possible du au carractère rectangulaire des imagettes.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from lib.file_management import get_files, get_tree_from_file
from lib.Point import Point
from lib.DatabaseCreatorImagette import DatabaseCreatorImagette
from lib.ReadDatabasePing import ReadDatabasePing
import numpy as np

folder_path = "./refactored_data/"

files = get_files("./refactored_data/")
dts = []

tgv_corr = False
width = 3000
height = 100

for i, filename in enumerate(files):
    if filename != "2024__0931101_Binned.nc":
        dt = get_tree_from_file(filename, folder_path)
        dt["filename"] = filename
        dts.append(dt)

rb = ReadDatabasePing(dts)
sonarPings = rb.get_sonarPings()

ship_x = []
ship_y = []

for ping in sonarPings:
    ship_x.append(ping.point_ship.eastern)
    ship_y.append(ping.point_ship.northern)

min_grid_x = min(ship_x)
max_grid_x = max(ship_x)

min_grid_y = min(ship_y)
max_grid_y = max(ship_y)

print(f"min_grid_x: {min_grid_x}, max_grid_x: {max_grid_x}")
print(f"min_grid_y: {min_grid_y}, max_grid_y: {max_grid_y}")

step = 30

x_range = np.arange(min_grid_x, max_grid_x + step, step)
y_range = np.arange(min_grid_y, max_grid_y + step, step)

X_grid, Y_grid = np.meshgrid(x_range, y_range) 

nb_lignes, nb_colonnes = X_grid.shape

print(f"Dimensions grille: {nb_lignes} x {nb_colonnes}")

dictionnaire_imagettes = {}

dci = DatabaseCreatorImagette("./database/Grid-All-eq-sf100-wn-a.nc")

total_groups = nb_lignes * nb_colonnes

dci.init_global_attributes(width, height, option={
    "TVG": tgv_corr
})

aug_list = [
    {"ping_offset": 5, "sample_offset": 0},
    {"ping_offset": -5, "sample_offset": 0},
    {"ping_offset": 0, "sample_offset": 50},
    {"ping_offset": 0, "sample_offset": -50},
]

for i in range(nb_lignes):
    for j in range(nb_colonnes):
        x = X_grid[i, j]
        y = Y_grid[i, j]

        # Extraction combinée : géolocalisée + augmentée
        imagettes_locat = rb.extract_augmented_imagette(
            sonarPings=sonarPings,
            target=Point(x, y, 0),
            height=height,
            width=width,
            tgv=tgv_corr,
            strict_single_file=True,
            only_centered=False,
            augmentations=aug_list,
            min_detection_range=0.3,
        )

        #imagettes_locat = [img for img in imagettes_locat if img.data.shape == (100, 3000)]


        print("nombre d imagette : ", len(imagettes_locat))

        # 3. Filtrage du groupe : on ignore les cases avec moins de 2 imagettes
        if len(imagettes_locat) < 5:
            continue

        # Écriture de la cellule filtrée sur disque
        dci.write_cell_data(x, y, imagettes_locat, i * nb_lignes + j * nb_colonnes)
        
        # Nettoyage explicite de la mémoire
        del imagettes_locat

# Fermeture finale
dci.close()
print("finish")