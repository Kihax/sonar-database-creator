"""
Ce script permet d'évaluer la sensibilité à la translation d'un réseau à partir d'imagette identiques mais décalées en translation.
Il permet de vérifier si le modèle est capable de détecter un objet même si celui-ci est légèrement décalé par rapport à la position d'origine.
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

# Indices d'intérêt d'origine
sample_interesting = [5300, 9900, 600, 10900, 4750, 11290]
ping_intesting = [3320, 5080, 13860, 34175, 19890, 6945]

# On crée directement des objets Point pour alimenter ta fonction de filtrage
points_interest = []
for i in range(len(ping_intesting)):
    eastering, northering = sonarPings[ping_intesting[i]].get_position_from_index(sample_interesting[i])
    points_interest.append(Point(eastering, northering, 0))

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

# configuration des augmentations à appliquer pour chaque point d'intérêt
augmentations_config = [
    {"ping_offset": 0, "sample_offset": 300, "rotation": 0.0},    # Translation droite
    {"ping_offset": 0, "sample_offset": -300, "rotation": 0.0},   # Translation gauche
    {"ping_offset": 30, "sample_offset": 0, "rotation": 0.0},     # Translation bas
    {"ping_offset": -30, "sample_offset": 0, "rotation": 0.0},    # Translation haut
]

dci = DatabaseCreatorImagette("./test-sensitivity-translation.nc")
init_main = False
total_groups = len(points_interest)

for idx, target_point in enumerate(points_interest):
    p_idx = ping_intesting[idx]
    s_idx = sample_interesting[idx]
        
    # extraction des imagettes originales et augmentées pour le point d'intérêt
    imagettes_valides = rb.extract_augmented_imagettes_with_indices(
        sonarPings=sonarPings,
        ping_idx=p_idx,
        sample_idx=s_idx,
        height=100,
        width=3000,
        tgv=False,
        augmentations=augmentations_config
    )
    
    if imagettes_valides:        
        # Initialisation du fichier NetCDF
        if not init_main:
            # On tente de récupérer la correction TVG de la base d'origine, sinon False
            tgv_corr = getattr(imagettes_valides[0], 'tgv', False) 
            
            # APPEL DE TA NOUVELLE FONCTION DÉDIÉE À IMAGETTEDATABASE
            dci.init_global_attributes(100, 3000, option={
                "TVG": tgv_corr
            })
            init_main = True
            
        # ÉCRITURE VIA LA FONCTION DÉDIÉE À IMAGETTEDATABASE
        # La liste 'imagettes_valides' contient désormais toutes les versions altérées et l'originale
        dci.write_cell_data(target_point.eastern, target_point.northern, imagettes_valides, idx)
    else:
        print(f"❌ Aucun groupe valide n'a pu être extrait pour les indices ({p_idx}, {s_idx}).")

dci.close()
