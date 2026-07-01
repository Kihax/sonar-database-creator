from lib.file_management import get_files, get_tree_from_file
from lib.DataManagement import DataManagement
from lib.Sonar import Sonar
from matplotlib import pyplot as plt
from lib.Point import Point
from lib.DatabaseCreator import DatabaseCreator
from lib.vizualize_imagette import vizualize_imagette
import numpy as np

folder_path = "../Dataset Metric/"

files = get_files("./Dataset Metric/")
dts = []
i = 0
for filename in files:
    if(filename != "2024__1001410_0001.001_Binned"): 
        dt = get_tree_from_file(filename, "../Dataset Metric/")
        dts.append(dt)
        print(filename)
    i = i + 1

print(dts[0])
s = Sonar(dts)

s.extract_lines_LF()
sonarPings = s.storage_LF

objects = s.get_objects(sonarPings)
groupes_imagettes = s.extract_imagette_from_object(objects, sonarPings);

for id_objet, imagettes_de_l_objet in groupes_imagettes.items():
    print(f"--- Affichage des {len(imagettes_de_l_objet)} vue(s) pour l'objet ID: {id_objet} ---")
    
    vizualize_imagette(imagettes_de_l_objet)