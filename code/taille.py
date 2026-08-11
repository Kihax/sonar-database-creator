from pathlib import Path
import matplotlib.pyplot as plt
import netCDF4 as nc

DATASET_NAME = "Grid-All-eq-sf100-wn.nc"
DATASET_PATH = Path("./") / DATASET_NAME

if not DATASET_PATH.exists():
    print(f"❌ Fichier non trouvé : {DATASET_PATH.resolve()}")
else:
    with nc.Dataset(DATASET_PATH, "r") as root_ds:
        counts = []

        for group_name, group_obj in root_ds.groups.items():
            if group_name.startswith("area_"):
                if "nb_imagette" in group_obj.ncattrs():
                    c = int(group_obj.getncattr("nb_imagette"))
                else:
                    c = len([g for g in group_obj.groups.keys() if g.startswith("imagette_")])
                counts.append(c)

        print(f"Nombre total de zones : {len(counts)}")
        print(f"Nombre total d'imagettes : {sum(counts)}")

        # Histogramme classique
        plt.figure(figsize=(8, 5))
        
        # bins automatiques basés sur la valeur max
        max_val = max(counts) if counts else 1
        bins = range(1, max_val + 2)

        plt.hist(counts, bins=bins, align="left", rwidth=0.8, color="steelblue", edgecolor="black")

        plt.xlabel("Nombre d'imagettes par zone")
        plt.ylabel("Nombre de zones")
        plt.title(f"Distribution du nombre d'imagettes par zone ({DATASET_NAME})")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()