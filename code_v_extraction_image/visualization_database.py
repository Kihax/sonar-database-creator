import xarray as xr
import os
from lib import vizualize_imagette

path = os.path.dirname(__file__)
dt = xr.open_datatree(path + "/../test.nc")

imagettes_set = []

for i in range(len(dt["image"])):
    imagette_data = {
        "image": dt["image"].values[i],
        "roll": dt["roll"].values[i],
        "pitch": dt["pitch"].values[i],
        "yaw": dt["yaw"].values[i],
        "heave": dt["heave"].values[i],
        "heading": dt["heading"].values[i],
        "easting": dt["easting"].values[i],
        "northing": dt["northing"].values[i],
        "easting_max": dt["easting_max"].values[i],
        "northing_max": dt["northing_max"].values[i],
        "easting_min": dt["easting_min"].values[i],
        "northing_min": dt["northing_min"].values[i],
        "timestamp": dt["timestamp"].values[i],
    }
    imagettes_set.append(imagette_data)

vizualize_imagette(imagettes_set, initial_index=0)