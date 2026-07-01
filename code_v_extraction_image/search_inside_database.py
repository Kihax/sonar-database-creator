import xarray as xr
import os
from lib import vizualize_imagette
from lib import search_coord

path = os.path.dirname(__file__)
dt = xr.open_datatree(path + "/../test.nc")

# search_coord(dt, easting, northing)
# note : les titres de la visualisation affichent d'abord N puis E
# donc attention à bien inverser les valeurs lors de la saisie
#imagettes_set = search_coord(dt, 402538.78, 4654381)
imagettes_set = search_coord(dt, 402508.16, 4654165.0)

vizualize_imagette(imagettes_set, initial_index=0, target_coord=(402508.16, 4654165.0))