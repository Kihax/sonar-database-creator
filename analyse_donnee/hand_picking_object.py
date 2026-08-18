"""
    Le but de ce script est d afficher les waterfalls sonar avec les pings associés 
    et incrémenté pour choisir les objets à récupérer dans la base de données et avoir leurs locations
    sous forme de ping/sample ce qui peut etre converti en coordonnées géographiques par la suite.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from lib.file_management import get_files, get_tree_from_file
from lib.ReadDatabasePing import ReadDatabasePing
from matplotlib import pyplot as plt
import numpy as np

folder_path = "../refactored_data/"
files = get_files("./refactored_data/")

total_pings_counter = 0

for filename in files:
    dts = [get_tree_from_file(filename, folder_path)]
    rd = ReadDatabasePing(dts)
    sonarPings = rd.get_sonarPings()

    if not sonarPings:
        continue

    waterfall = [ping.sample for ping in sonarPings]
    waterfall_matrix = np.array(waterfall, dtype=float)

    if waterfall_matrix.size == 0:
        continue

    p1, p99 = np.percentile(waterfall_matrix, (1, 99))
    if p99 > p1:
        waterfall_stretched = np.clip((waterfall_matrix - p1) / (p99 - p1), 0.0, 1.0)
    else:
        waterfall_stretched = np.zeros_like(waterfall_matrix)

    current_file_pings = len(sonarPings)
    y_start = total_pings_counter
    y_end = total_pings_counter + current_file_pings
    num_samples = waterfall_matrix.shape[1]
    img_extent = [0, num_samples, y_end, y_start]

    plt.figure(figsize=(10, 6))
    plt.imshow(waterfall_stretched, cmap="gray", aspect="auto", extent=img_extent, vmin=0.0, vmax=1.0)
    plt.title(f"Waterfall de base - {filename}")
    plt.xlabel("Sample")
    plt.ylabel("Ping")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.show()

    total_pings_counter += current_file_pings
