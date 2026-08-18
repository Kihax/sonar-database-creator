"""
Le but de ce script est d'afficher la trajectoire du navire à partir des coordonnées de chaque ping.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from lib.file_management import get_files, get_tree_from_file
from lib.ReadDatabasePing import ReadDatabasePing
import matplotlib.pyplot as plt

# 1. Chargement rapide des données
folder_path = "../refactored_data/"
files = get_files("./refactored_data/")
dts = []

for filename in files:
    if filename != "2024__0931101_Binned.nc":
        dt = get_tree_from_file(filename, folder_path)
        dt["filename"] = filename
        dts.append(dt)

rb = ReadDatabasePing(dts)
sonarPings = rb.get_sonarPings()

# 2. Extraction des coordonnées
ship_x = [ping.point_ship.eastern for ping in sonarPings]
ship_y = [ping.point_ship.northern for ping in sonarPings]

# 3. Affichage des stats dans le terminal
x_min, x_max = min(ship_x), max(ship_x)
y_min, y_max = min(ship_y), max(ship_y)

print("--- STATISTIQUES DES COORDONNÉES ---")
print(f"Nombre total de pings : {len(sonarPings)}")
print(f"X (Eastern)  -> Min: {x_min:.2f} | Max: {x_max:.2f} | Écart: {x_max - x_min:.2f} m")
print(f"Y (Northern) -> Min: {y_min:.2f} | Max: {y_max:.2f} | Écart: {y_max - y_min:.2f} m")
print("-------------------------------------")

# 4. Génération du graphique
plt.figure(figsize=(10, 6))
plt.plot(ship_x, ship_y, label="Trajectoire du navire", color="blue", linewidth=1.5)
plt.scatter(ship_x, ship_y, c="red", s=1, alpha=0.5, label="Pings")

plt.title("Visualisation de la position du navire (chaque ping)")
plt.xlabel("Eastern (m)")
plt.ylabel("Northern (m)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.axis("equal")  # Garde les proportions géographiques réelles

print("Génération du graphique en cours...")
plt.show()