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


folder_path = "../refactored_data_survey1/"
files = get_files("./refactored_data_survey1/")
dts = [get_tree_from_file(filename, folder_path) for filename in files]

rd = ReadDatabasePing(dts)
sonarPings: List[SonarPing] = rd.get_sonarPings()

# Boucle par pas de 300
for i in range(0, len(sonarPings), 300):
    waterfall = []
    
    # Déterminer la fin théorique du lot (maximum 300 pings)
    fin_lot = min(i + 300, len(sonarPings))
    
    # La longueur de référence sera celle du tout premier ping de ce lot
    longueur_reference = len(sonarPings[i].sample)
    
    for j in range(i, fin_lot):
        actuel_sample = sonarPings[j].sample
        
        # Si la longueur change, on arrête immédiatement de remplir ce waterfall
        if len(actuel_sample) != longueur_reference:
            print(f"Changement de taille détecté au ping {j} ({len(actuel_sample)} vs {longueur_reference}). Arrêt du lot.")
            break
            
        waterfall.append(actuel_sample)
    
    # On n'affiche le waterfall que s'il contient des données
    if waterfall:
        print(f"Affichage du waterfall (Pings {i} à {i + len(waterfall) - 1})")
        plt.imshow(waterfall, vmax=250, vmin=0, cmap='grey', aspect="auto")
        plt.show()