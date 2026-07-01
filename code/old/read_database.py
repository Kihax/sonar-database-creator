import os
import xarray as xr
from lib.SonarPing import SonarPing
from lib.Point import Point
from code.lib.ReadDatabasePing import ReadDatabase
from lib.vizualize_imagette import vizualize_imagette

rd = ReadDatabase("./refactored_data")
sonarPings = rd.get_sonarPings()
print(len(sonarPings))
imagettes = rd.extract_imagette(sonarPings, Point(402508.16, 4654165.0), max_dist=0.3, width=200, height=200)
vizualize_imagette(imagettes)