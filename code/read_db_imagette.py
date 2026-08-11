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

dt = get_tree_from_file("HP-Centered-100.nc", "../")
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