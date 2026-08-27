"""
Ce script permet de créer une base de données d'imagettes en prennant des zones  de 30x30 m à partir d'une grille
"""

# permet d importer les librairies depuis le dossier parent
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

for i, filename in enumerate(files):
    if(filename != "2024__0931101_Binned.nc"):
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

min_grid_x= min(ship_x)
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

print(nb_lignes, nb_colonnes)

dictionnaire_imagettes = {}

# création du fichier contennant la base de données d'imagettes
dci = DatabaseCreatorImagette("./database/Grid-All-eq-sf100.nc")
init_main = False

total_groups = nb_lignes * nb_colonnes

for i in range(nb_lignes):
    for j in range(nb_colonnes):
        x = X_grid[i, j]
        y = Y_grid[i, j]

        # On extrait les imagettes JUSTE pour cette case (x, y)
        imagettes_locat = rb.extract_imagette(sonarPings, Point(x, y, 0), 100, 3000, tgv=tgv_corr, strict_single_file=True, only_centered=False)

        if len(imagettes_locat) < 5:
            # on passe si moins de 5 imagettes
            continue

        # Initialisation unique des dimensions globales
        if not init_main:
            dci.init_global_attributes(3000, 100, option={
                "TVG": tgv_corr
            })
            init_main = True
        
        # Écriture immédiate sur disque
        dci.write_cell_data(x, y, imagettes_locat, i*nb_lignes+j*nb_colonnes)
        
        # Nettoyage explicite de la mémoire de la cellule courante
        del imagettes_locat

dci.close()
