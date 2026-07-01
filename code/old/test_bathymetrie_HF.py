from lib.file_management import get_files, get_tree_from_file
from lib.DataManagement import DataManagement
from lib.Sonar import Sonar
from matplotlib import pyplot as plt
from lib.Point import Point
from lib.DatabaseCreator import DatabaseCreator
import numpy as np

folder_path = "../Dataset Metric/"

files = get_files("./Dataset Metric/")
dts = []
i = 0
for filename in files:
    if(filename != "2024__1001410_0001.001_Binned" and i == 1):
        dt = get_tree_from_file(filename, "../Dataset Metric/")
        dts.append(dt)
        
        print(filename)
    i = i + 1

print(dts[0])
s = Sonar(dts)

s.extract_lines_HF()
sonarPings = s.storage_HF

for sonarPing in sonarPings:
    plt.plot(sonarPing.sample)
    (depth_port, depth_starboard) = sonarPing.get_index_depth_absolute()

    plt.axvline(x=depth_port, color='red', linestyle='--', linewidth=2, label='profondeur')
    plt.axvline(x=depth_starboard, color='red', linestyle='--', linewidth=2, label='profondeur')

    plt.show()
