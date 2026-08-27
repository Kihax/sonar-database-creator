"""
Ce script permet de créer une base de données avec des points d'intérêt en les séléctionnants manuellement (voir le fichier hand_picking_object.py).
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from lib.file_management import get_files, get_tree_from_file
from lib.Point import Point
from lib.DatabaseCreatorImagette import DatabaseCreatorImagette
from lib.ReadDatabasePing import ReadDatabasePing
import math

# base de données tempon
folder_path = "./refactored_data/"
files = get_files("./refactored_data/")
dts_pings = []

for filename in files:
    if filename != "2024__0931101_Binned.nc":
        dt = get_tree_from_file(filename, folder_path)
        dt["filename"] = filename
        dts_pings.append(dt)

# On récupère les pings de la base de données
rb = ReadDatabasePing(dts_pings)
sonarPings = rb.get_sonarPings()

#sample_interesting = [5300, 9900, 600, 10900, 4750, 2500, 1075, 14100, 9800, 4050, 1500, 3800, 14000, 11700, 11290, 400, 11000, 2550, 10000, 1000]
#ping_intesting = [3320, 5080, 13860, 34175, 19890, 597, 1257, 1680, 2040, 2540, 2915, 3675, 4640, 5390, 6945, 8200, 8970, 9110]

# On défini les points d'intérêt à partir des pings et samples choisis manuellement (hand_picking_object.py)
ping_intesting = [40, 287, 410, 473, 670, 495, 55,              198, 330, 600, 680, 700, 838,           900, 997, 1395, 1525, 1545, 875, 1170,      1190, 1345, 1570, 1695, 1890, 1900, 2045, 2180,         1695, 1780, 1955, 2075, 2140, 2410, 2325, 2430  ]
sample_interesting = [4210, 4975, 4500, 5575, 5050, 400, 11150, 10900, 11175, 14200, 9950, 13250, 4560, 500, 3200, 5050, 1200, 4450, 10150, 13200,  10400, 10375, 10900, 4950, 3400, 300, 3950, 4350,       11000, 10800, 10700, 9800, 12400, 4600, 10300, 12400   ]

if(len(ping_intesting) != len(sample_interesting)): # permet de vérifier qu'on a pas faire d'erreur lors de la copie des données
    print(len(ping_intesting), len(sample_interesting))
    raise ValueError("Les listes ping_intesting et sample_interesting doivent avoir la même longueur.")

# On extrait les coordonnées géographiques des points choisis à partir des pings et samples
points_interest = []
for i in range(len(ping_intesting)):
    eastering, northering = sonarPings[ping_intesting[i]].get_position_from_index(sample_interesting[i])
    points_interest.append(Point(eastering, northering, 0))

# On vérifie que les points d'intérêt extrait sont à plus de 30 m les uns des autres pour éviter de fausses l algorithmes
proches_trouves = False
for i in range(len(points_interest)):
    for j in range(i + 1, len(points_interest)):
        p1 = points_interest[i]
        p2 = points_interest[j]
        
        distance = math.sqrt((p1.eastern - p2.eastern)**2 + (p1.northern - p2.northern)**2)
        if distance < 30:
            print(f"⚠️ Groupe {i} et Groupe {j} sont très proches ! Distance : {distance:.2f}m")
            print(f"   -> Groupe {i} (Ping {ping_intesting[i]}, Sample {sample_interesting[i]})")
            print(f"   -> Groupe {j} (Ping {ping_intesting[j]}, Sample {sample_interesting[j]})")
            proches_trouves = True

dci = DatabaseCreatorImagette("./model/HP-Centered-100.nc") # création du fichier de la base de données
init_main = False
total_groups = len(points_interest)

for idx, target_point in enumerate(points_interest):
    imagettes_valides = rb.extract_imagette(sonarPings, target_point, 100, 3000, only_centered=True, strict_single_file=True) # extraction des imagettes
    
    if imagettes_valides:
        # Initialisation du fichier NetCDF au premier groupe valide trouvé
        if not init_main:
            # On tente de récupérer la correction TVG de la base d'origine, sinon False
            tgv_corr = getattr(imagettes_valides[0], 'tgv', False) 
            
            # APPEL DE TA NOUVELLE FONCTION DÉDIÉE À IMAGETTEDATABASE
            dci.init_global_attributes(100, 3000, option={
                "TVG": tgv_corr
            })
            init_main = True
            
        # ÉCRITURE VIA LA FONCTION DÉDIÉE À IMAGETTEDATABASE
        dci.write_cell_data(target_point.eastern, target_point.northern, imagettes_valides, idx)
    else:
        print(f"❌ Aucun groupe ne contient ce point à plus de 75% dans la base globale.")

# Fermeture finale indispensable
dci.close()
