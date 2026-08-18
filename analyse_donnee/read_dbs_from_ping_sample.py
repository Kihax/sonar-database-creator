"""
Ce script est la continuité de hand_picking_object, il permet de récupérer les coordonnées géographiques d'un points (à partir de son ping et sample) et de trouver les imagettes associées à ce point
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from lib.file_management import get_files, get_tree_from_file
from lib.Point import Point
from lib.ReadDatabasePing import ReadDatabasePing
from lib.vizualize_imagette import vizualize_imagette
from lib.Imagette import Imagette
from typing import List

folder_path = "../refactored_data/"

files = get_files("./refactored_data/")
print(len(files))
dts = []

for i, filename in enumerate(files):
    if(filename != "2024__0931101_Binned.nc"):
        dt = get_tree_from_file(filename, folder_path)
        dt["filename"] = filename
        dts.append(dt)

rb = ReadDatabasePing(dts)
sonarPings = rb.get_sonarPings()

ping = 1520
sample = 1400

(eastering, northering) = sonarPings[ping].get_position_from_index(sample)
print(f"target : ({eastering}, {northering})")
imagettes : List[Imagette] = rb.extract_imagette(sonarPings, Point(eastering, northering), max_dist=0.3, width=3000, height=200)

vizualize_imagette(imagettes, 3)