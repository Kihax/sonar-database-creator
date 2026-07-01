from lib import DataManagement, Sonar
from lib.file_management import get_all_latitude_longitude, get_files, get_tree_from_index, get_tree_from_file
from lib.DatabaseCreator import DatabaseCreator

filenames = get_files('./Dataset Metric/') # dépend d ou on ouvre les données
dts = []

for filename in filenames:
    if(filename != "2024__1001410_0001.001_Binned.nc"):
        dt = get_tree_from_file(filename)
        dts.append(dt)


s = Sonar(dts)

imagettes = s.extract_imagette_HF(None, 100);

database = DatabaseCreator(imagettes, "./test.nc")

database.create()