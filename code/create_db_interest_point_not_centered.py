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

# Import de ta classe modifiée
from lib.ReadDatabaseImagette import ReadDatabaseImagette

# =====================================================================
# 1. CHARGEMENT DE LA BASE D'IMAGETTES GLOBALE
# =====================================================================
print("--- Lecture de la base de données d'imagettes globale ---")
dt_imagettes = get_tree_from_file("database_3000x100_full.nc", "../")
rdi = ReadDatabaseImagette([dt_imagettes])
rdi.extract() # Remplit rdi.pos_imagette

# =====================================================================
# 2. CALCUL DES POINTS D'INTÉRÊT (CIBLES) VIA LES PINGS
# =====================================================================
print("\n--- Calcul des positions d'intérêt ---")
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

sample_interesting = [5300, 9900, 5400, 600, 10900, 4750, 2500, 1075]
ping_intesting = [3320, 5080, 3470, 13860, 34175, 19890]

# Création des vrais objets 'Point' attendus par ta fonction
points_interest = []
for i in range(len(ping_intesting)):
    eastering, northering = sonarPings[ping_intesting[i]].get_position_from_index(sample_interesting[i])
    # Instanciation de Point(easting, northing, altitude/depth=0)
    points_interest.append(Point(eastering, northering, 0))

# =====================================================================
# 3. FILTRAGE GÉOMÉTRIQUE ET CRÉATION DE LA NOUVELLE BASE
# =====================================================================
print("\n--- Filtrage (Seuil 75%) et Écriture ---")
dci = DatabaseCreatorImagette("./database_interest_point_not_centered_from_3000x100.nc")
init_main = False
total_groups = len(points_interest)

for idx, target_point in enumerate(points_interest):
    print(f"\nRecherche pour le Point {idx+1}/{total_groups} : ({target_point.eastern:.2f}, {target_point.northern:.2f})")
    
    # Appel de ta fonction adaptée
    coord_groupe, imagettes_valides = rdi.get_groups_with_75_percent_containing_point(target_point)
    
    if imagettes_valides:
        # Initialisation du fichier NetCDF si premier groupe valide trouvé
        if not init_main:
            # On récupère le paramètre de correction TVG de la première imagette
            tgv_corr = getattr(imagettes_valides[0], 'tgv', False) 
            dci.init_global_attributes_from_database(imagettes_valides[0], nb_groups=total_groups, option={"TVG": tgv_corr})
            init_main = True
            
        # Écriture du groupe complet d'imagettes dans ta base d'intérêt
        dci.write_cell_data_from_database(target_point.eastern, target_point.northern, imagettes_valides, idx)
    else:
        print(f"❌ Aucun groupe ne contient ce point à plus de 75%.")

# Fermeture finale du fichier NetCDF
dci.close()
print("\n🎉 Base de données d'intérêt 'database_interest_point_not_centered_from_3000x100.nc' créée avec succès !")