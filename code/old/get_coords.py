from lib.file_management import get_files, get_tree_from_file
from lib.DataManagement import DataManagement
from lib.Sonar import Sonar
from matplotlib import pyplot as plt
from lib.Point import Point
from lib.DatabaseCreator import DatabaseCreator
from lib.ReadDatabasePing import ReadDatabasePing
from lib.vizualize_imagette import vizualize_imagette
from typing import List
from lib.SonarPing import SonarPing


folder_path = "../refactored_data/"
files = get_files("./refactored_data/")
dts = [get_tree_from_file(filename, folder_path) for filename in files]

rd = ReadDatabasePing(dts)
sonarPings: List[SonarPing] = rd.get_sonarPings()

nb_ping_waterfall = 300;
waterfall = 0;
sample = 11300
ping = 64

print(sonarPings[waterfall*nb_ping_waterfall + ping].filename)
(eastering, northering) = sonarPings[waterfall*nb_ping_waterfall + ping].get_position_from_index(sample)

print(f"eastering : {eastering} and northering : {northering} -> ({eastering}, {northering})")