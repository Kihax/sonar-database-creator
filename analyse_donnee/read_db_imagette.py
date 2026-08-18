"""
Ce script permet de lire les imagettes d'une base de données et de les afficher pour une analyse visuelle.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from lib.file_management import get_tree_from_file
from lib.ReadDatabaseImagette import ReadDatabaseImagette
from lib.ScrollImageViewer import ScrollImageViewer

dt = get_tree_from_file("HP-Centered-100.nc", "../database/")
dts = [dt]

rdi = ReadDatabaseImagette(dts)
rdi.extract()
min_imagettes = 0

global_imagettes = rdi.pos_imagette

matching_groups = {
    coord: images for coord, images in global_imagettes.items() if len(images) > min_imagettes
}

# Boucle principale : affiche une position après l'autre
for coord, list_imagettes in matching_groups.items():
    print(f"Affichage de la position {coord} ({len(list_imagettes)} imagettes).")
    viewer = ScrollImageViewer(coord, list_imagettes)
    viewer.show()

print("Toutes les positions ont été visionnées.")