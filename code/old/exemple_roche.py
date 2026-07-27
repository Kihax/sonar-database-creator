from lib.file_management import get_files, get_tree_from_file
from lib.DataManagement import DataManagement
from lib.Sonar import Sonar
from matplotlib import pyplot as plt
from lib.Point import Point
from lib.DatabaseCreator import DatabaseCreator
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

sample = 4400
ping = 2670
(eastering, northering) = sonarPings[ping].get_position_from_index(sample)
print(f"target : ({eastering}, {northering})")
imagettes : List[Imagette] = rb.extract_imagette(sonarPings, Point(eastering, northering), max_dist=0.3, width=3000, height=200)

vizualize_imagette(imagettes, 3)