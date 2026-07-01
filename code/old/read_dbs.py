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
dts = []

for i, filename in enumerate(files):
    if(filename != "2024__0931101_Binned.nc"):
        dt = get_tree_from_file(filename, folder_path)
        dt["filename"] = filename
        dts.append(dt)

rb = ReadDatabasePing(dts)
sonarPings = rb.get_sonarPings()
#imagettes : List[Imagette] = rb.extract_imagette(sonarPings, Point(402614.75, 4654423.5), max_dist=0.1, width=300, height=300)

imagettes : List[Imagette] = rb.extract_imagette(sonarPings, Point(402653.375, 4654312.0), max_dist=0.1, width=300, height=300)

vizualize_imagette(imagettes)