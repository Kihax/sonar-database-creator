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
    if(filename != "2024__0931101_Binned.nc"):
        dt = get_tree_from_file(filename, folder_path)
        dt["filename"] = filename
        dts.append(dt)

rb = ReadDatabasePing(dts)
sonarPings = rb.get_sonarPings()

sample_interesting = [5300, 4400, 9900, 2600, 5400, 600, 10900, 4750, 2500, 1075]
ping_intesting = [3320, 2670, 5080, 6020, 3470, 13860, 34175, 19890]

positions_interrest = []

dictionnaire_imagettes = {}

for i in range(len(ping_intesting)):
    (eastering, northering) = sonarPings[ping_intesting[i]].get_position_from_index(sample_interesting[i])
    positions_interrest.append((eastering, northering))

    
dci = DatabaseCreatorImagette("./database_interest_point.nc")
init_main = False

total_groups = len(positions_interrest)

for idx, (eastering, northering) in enumerate(positions_interrest):

    imagettes_locat = rb.extract_imagette(sonarPings, Point(eastering, northering, 0), 200, 3000, tgv=tgv_corr, strict_single_file=True)

    if not init_main:
        dci.init_global_attributes(imagettes_locat[0], nb_groups=total_groups, option={
            "TVG": tgv_corr
        })
        init_main = True
        
    dci.write_cell_data(eastering, northering, imagettes_locat, idx)
        
    del imagettes_locat

# Fermeture finale indispensable
dci.close()