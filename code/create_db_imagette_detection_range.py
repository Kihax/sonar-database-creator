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

folder_path = "../refactored_data/"

files = get_files("./refactored_data/")
dts = []

tgv_corr = False

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

dci = DatabaseCreatorImagette("./Grid-All-eq-sf100-without-nadir.nc")
init_main = False

total_groups = nb_lignes * nb_colonnes

for i in range(nb_lignes):
    for j in range(nb_colonnes):
        x = X_grid[i, j]
        y = Y_grid[i, j]

        # 1. Extraction brute des imagettes pour la case (x, y)
        imagettes_locat = rb.extract_imagette(
            sonarPings, Point(x, y, 0), 100, 3000, 
            tgv=tgv_corr, strict_single_file=True, only_centered=False
        )

        print("nombre d imagette avant filtre : ", len(imagettes_locat))

        # 2. Filtrage : on ne conserve que les imagettes avec detection_range > 0.3
        imagettes_locat = [
            img for img in imagettes_locat 
            if img.getDetectionRange() > 0.3
        ]

        print("nombre d imagette apres filtre : ", len(imagettes_locat))

        # 3. Filtrage du groupe : on ignore les cases avec moins de 2 imagettes
        if len(imagettes_locat) < 2:
            continue

        # Initialisation unique des dimensions globales lors du premier groupe valide
        if not init_main:
            dci.init_global_attributes(imagettes_locat[0], nb_groups=total_groups, option={
                "TVG": tgv_corr
            })
            
            init_main = True
        
        # Écriture de la cellule filtrée sur disque
        dci.write_cell_data(x, y, imagettes_locat, i * nb_lignes + j * nb_colonnes)
        
        # Nettoyage explicite de la mémoire
        del imagettes_locat

# Fermeture finale
dci.close()
print("finish")