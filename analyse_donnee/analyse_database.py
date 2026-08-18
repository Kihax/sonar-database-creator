"""
    Le but de ce script est d'analyser l'impact du filtrage du nadir sur le nombre d'imagettes par zones
    Et ainsi savoir si le nombre d'imagettes par classe ou zones est suffisant pour l'apprentissage du modèle.
    Il permet également d'afficher les imagettes filtrées pour vérifier visuellement le résultat du filtrage.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
from typing import List

from lib.file_management import get_tree_from_file
from lib.ReadDatabaseImagette import ReadDatabaseImagette
from lib.ScrollImageViewer import ScrollImageViewer

# 1. Chargement du fichier NetCDF
dt = get_tree_from_file("Grid-All-eq-sf100.nc", "./database/")
dts = [dt]

# 2. Extraction des imagettes
rdi = ReadDatabaseImagette(dts)
rdi.extract()

global_imagettes = rdi.pos_imagette

detection_ranges: List[float] = []
filtered_imagettes = {}

total_imagettes = 0
nadir_count = 0

nadir_threshold = 0.35  # Seuil pour considérer une imagette proche du Nadir

for coord, list_imagettes in global_imagettes.items():
    nadir_list_for_coord = []
    
    for img in list_imagettes:
        total_imagettes += 1
        
        # Récupération de la portée sur l'imagette ou son premier ping
        rng = None
        if hasattr(img, "detection_range") and img.detection_range is not None:
            rng = img.detection_range
        elif (
            hasattr(img, "sliced_pings")
            and img.sliced_pings
            and hasattr(img.sliced_pings[0], "detection_range")
        ):
            rng = img.sliced_pings[0].detection_range

        if rng is not None:
            detection_ranges.append(rng)
            # Filtrage pour l'affichage
            if rng < nadir_threshold:
                nadir_list_for_coord.append(img)
                nadir_count += 1

    if nadir_list_for_coord:
        filtered_imagettes[coord] = nadir_list_for_coord

print(f"Nombre total d'imagettes récupérées : {total_imagettes}")
print(f"Nombre d'imagettes proches du Nadir (> {nadir_threshold}) : {nadir_count} ({(nadir_count / total_imagettes * 100) if total_imagettes else 0:.1f}%)")

# 4. Affichage de l'histogramme
if detection_ranges:
    detection_ranges_arr = np.array(detection_ranges)

    mean_val = float(np.mean(detection_ranges_arr))
    median_val = float(np.median(detection_ranges_arr))
    near_nadir_pct = (
        float(np.sum(detection_ranges_arr > nadir_threshold)) / len(detection_ranges_arr)
    ) * 100.0

    print(f"Portée moyenne : {mean_val:.3f}")
    print(f"Portée médiane : {median_val:.3f}")

    plt.figure(figsize=(9, 5))

    counts, bins, patches = plt.hist(
        detection_ranges_arr,
        bins=30,
        range=(0.0, 1.0),
        edgecolor="black",
        alpha=0.75,
        color="#2b5c8f",
    )

    for bin_left, patch in zip(bins[:-1], patches):
        if bin_left < 0.15:
            patch.set_facecolor("#d9534f")

    plt.axvline(
        mean_val,
        color="orange",
        linestyle="dashed",
        linewidth=2,
        label=f"Moyenne ({mean_val:.2f})",
    )
    plt.axvline(
        median_val,
        color="green",
        linestyle="dashdot",
        linewidth=2,
        label=f"Médiane ({median_val:.2f})",
    )

    plt.title(
        "Distribution de la portée de détection ($Detection\\ Range$)",
        fontsize=12,
    )
    plt.xlabel("Portée relative (0.0 = Nadir / Centre, 1.0 = Fin de fauchée)")
    plt.ylabel("Nombre d'imagettes")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()

# 5. Affichage interactif des imagettes < 0.15 avec ScrollImageViewer
if filtered_imagettes:
    print("\nLancement du visionneur pour les imagettes avec range > 0.4...")
    for coord, list_imagettes in filtered_imagettes.items():
        print(f"Affichage de la position {coord} ({len(list_imagettes)} imagette(s) > 0.4).")
        viewer = ScrollImageViewer(coord, list_imagettes)
        viewer.show()
    print("Toutes les imagettes proches du Nadir ont été visionnées.")
else:
    print("Aucune imagette avec detection_range > 0.4 n'a été trouvée.")