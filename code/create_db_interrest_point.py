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
import math

# Import de ta classe de lecture d'imagettes
from lib.ReadDatabaseImagette import ReadDatabaseImagette

# =====================================================================
# 2. CALCUL DES POINTS D'INTÉRÊT (CIBLES) VIA LES PINGS
# =====================================================================
print("\n--- Calcul des positions d'intérêt depuis les pings ---")
folder_path = "../refactored_data/"
files = get_files("./refactored_data/")
dts_pings = []

for filename in files:
    if filename != "2024__0931101_Binned.nc":
        dt = get_tree_from_file(filename, folder_path)
        dt["filename"] = filename
        dts_pings.append(dt)

rb = ReadDatabasePing(dts_pings)
sonarPings = rb.get_sonarPings()

#sample_interesting = [5300, 9900, 600, 10900, 4750, 2500, 1075, 14100, 9800, 4050, 1500, 3800, 14000, 11700, 11290, 400, 11000, 2550, 10000, 1000]
#ping_intesting = [3320, 5080, 13860, 34175, 19890, 597, 1257, 1680, 2040, 2540, 2915, 3675, 4640, 5390, 6945, 8200, 8970, 9110]

ping_intesting = [1520, 1190, 1160, 2140, 1880, 2430, 4185, 6535, 7590, 735, 888, 1956, 2783, 3549, 4860, 5360, 10090, 11420, 11910, 12605, 13805, 13908, 16012, 21945]
sample_interesting = [1400, 10400, 13500, 12400, 3150, 12400, 4200, 11360, 10100, 10600, 5300, 10750, 4400, 3925, 13800, 5400, 3600, 10950, 13800, 10600, 4200, 12100, 11600, 11000]

if(len(ping_intesting) != len(sample_interesting)):
    print(len(ping_intesting), len(sample_interesting))
    raise ValueError("Les listes ping_intesting et sample_interesting doivent avoir la même longueur.")

# obj recalé : (sample : 5400 - ping : 3470)

# On crée directement des objets Point pour alimenter ta fonction de filtrage
points_interest = []
for i in range(len(ping_intesting)):
    eastering, northering = sonarPings[ping_intesting[i]].get_position_from_index(sample_interesting[i])
    points_interest.append(Point(eastering, northering, 0))

print("\n--- Vérification des distances entre les groupes ---")
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

if not proches_trouves:
    print("✅ Aucun groupe n'est à moins de 30 mètres d'un autre.")

# =====================================================================
# 3. FILTRAGE ET ÉCRITURE DANS LA NOUVELLE BASE DE DONNÉES
# =====================================================================
dci = DatabaseCreatorImagette("./HP-Centered-100.nc")
init_main = False
total_groups = len(points_interest)

for idx, target_point in enumerate(points_interest):
    print(f"\nRecherche pour le Point {idx+1}/{total_groups} : ({target_point.eastern:.2f}, {target_point.northern:.2f})")
    
    # Ta fonction de filtrage adaptée qui renvoie le meilleur groupe d'ImagetteDatabase
    imagettes_valides = rb.extract_imagette(sonarPings, target_point, 100, 3000, only_centered=True, strict_single_file=True)
    
    if imagettes_valides:
        # Initialisation du fichier NetCDF au premier groupe valide trouvé
        if not init_main:
            # On tente de récupérer la correction TVG de la base d'origine, sinon False
            tgv_corr = getattr(imagettes_valides[0], 'tgv', False) 
            
            # APPEL DE TA NOUVELLE FONCTION DÉDIÉE À IMAGETTEDATABASE
            dci.init_global_attributes(imagettes_valides[0], nb_groups=total_groups, option={
                "TVG": tgv_corr
            })
            init_main = True
            
        # ÉCRITURE VIA LA FONCTION DÉDIÉE À IMAGETTEDATABASE
        dci.write_cell_data(target_point.eastern, target_point.northern, imagettes_valides, idx)
    else:
        print(f"❌ Aucun groupe ne contient ce point à plus de 75% dans la base globale.")

# Fermeture finale indispensable
dci.close()
print("\n🎉 Nouvelle base d'intérêt générée avec succès.")