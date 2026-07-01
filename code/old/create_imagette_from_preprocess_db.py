import os
import xarray as xr
from lib.SonarPing import SonarPing
from lib.Point import Point
from code.lib.ReadDatabasePing import ReadDatabase
from lib.vizualize_imagette import vizualize_imagette
from lib.Sonar import Sonar

rd = ReadDatabase("./code/test.nc")

sonarPings = rd.get_sonarPings()
s = Sonar([]);

objects = s.get_objects(sonarPings)
groupes_imagettes = s.extract_imagette_from_object(objects, sonarPings);

print(len(groupes_imagettes))
for id_objet, imagettes_de_l_objet in groupes_imagettes.items():
    print(f"--- Affichage des {len(imagettes_de_l_objet)} vue(s) pour l'objet ID: {id_objet} ---")
    
    vizualize_imagette(imagettes_de_l_objet)