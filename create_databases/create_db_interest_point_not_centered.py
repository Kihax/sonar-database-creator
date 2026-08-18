"""
Ce script permet de créer une base de données d'imagettes non-centrées mais avec des points d'inérêt choisie et en sélectionnant les endroits où ils sont présents dans la grille.
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
from lib.ReadDatabaseImagette import ReadDatabaseImagette

# ouverture de la base de données et extraction des imagettes
dt_imagettes = get_tree_from_file("Grid-All-eq-sf100", "./database/")
rdi = ReadDatabaseImagette([dt_imagettes])
rdi.extract()

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

# Récupération des points d'intérêt à partir des pings et samples spécifiés
points_interest = []
for i in range(len(ping_intesting)):
    eastering, northering = sonarPings[ping_intesting[i]].get_position_from_index(sample_interesting[i])
    points_interest.append(Point(eastering, northering, 0))

dci = DatabaseCreatorImagette("./Grid-All-eq-sf100-ip.nc") # Grille, sf = un seul waterfall, ip = interest point
init_main = False
total_groups = len(points_interest)

for idx, target_point in enumerate(points_interest):    
    coord_groupe, imagettes_valides = rdi.get_groups_with_75_percent_containing_point(target_point) # Récupération des groupes contenant le point d'intérêt à plus de 75%
    
    if imagettes_valides:
        # Initialisation du fichier NetCDF si premier groupe valide trouvé
        if not init_main:
            # On récupère le paramètre de correction TVG de la première imagette
            tgv_corr = getattr(imagettes_valides[0], 'tgv', False) 
            dci.init_global_attributes_from_database(100, 3000, option={"TVG": tgv_corr})
            init_main = True
            
        # Écriture du groupe complet d'imagettes dans ta base d'intérêt
        dci.write_cell_data_from_database(target_point.eastern, target_point.northern, imagettes_valides, idx)
    else:
        print(f"❌ Aucun groupe ne contient ce point à plus de 75%.")

dci.close() # Fermeture finale du fichier NetCDF