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


# ... vos imports restants inchangés ...

folder_path = "../refactored_data/"
files = get_files("./refactored_data/")
dts = [get_tree_from_file("2024__0931123_Binned.nc", folder_path)]

rd = ReadDatabasePing(dts)
sonarPings: List[SonarPing] = rd.get_sonarPings()
waterfall = []
for i in range(415, 715):
    waterfall.append(sonarPings[i].sample[11842:12142])
    
plt.imshow(waterfall, vmax=250, vmin=0, cmap='grey', aspect="auto")
plt.show()