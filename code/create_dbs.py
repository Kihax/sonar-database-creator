from lib.file_management import get_files, get_tree_from_file
from lib.DataManagement import DataManagement
from lib.Sonar import Sonar
from matplotlib import pyplot as plt
from lib.Point import Point
from lib.DatabaseCreator import DatabaseCreator

folder_path = "../Dataset Metric/"

files = get_files("./Dataset Metric/")
i = 0
dts = []
for filename in files:
    
    if(filename != "2024__1001410_0001.001_Binned" and "Binned" in filename):
        print(filename)
        dt = get_tree_from_file(filename, folder_path)
        dts.append(dt)
        
    i+=1

data = DataManagement(dts)

for i, dt in enumerate(dts):
    print(f"progress : {i}/{len(dts)}")
    s = Sonar([dt], data)
    filename = dt["filename"].values
    print(filename)
    s.extract_lines_LF()
    dc = DatabaseCreator(s.storage_LF, "./refactored_data/" + filename)
    dc.create()