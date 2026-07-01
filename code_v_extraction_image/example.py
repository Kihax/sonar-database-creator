import math
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpp
import matplotlib.dates as mpd
import pandas

import netCDF4 as nc

import xarray as xr

from lib import DataManagement, Sonar
from lib.file_management import get_all_latitude_longitude, get_files, get_tree_from_index
from lib.DatabaseCreator import DatabaseCreator

FILE_INDEX = 1

dt = get_tree_from_index(FILE_INDEX)
#get_property_from_timestamp("/Platform/Position/latitude", "2020-01-01T00:00:00Z", dt)

dt, filename = get_tree_from_index(FILE_INDEX)
print(filename)
dt1, filename1 = get_tree_from_index(FILE_INDEX+1)

data = DataManagement([dt, dt1])

latitude = data.latitude.get_value_from_timestamp(1712858300567000000)
longitude = data.longitude.get_value_from_timestamp(1712858300567000000)
sound_speed = data.sound_speed.get_value_from_timestamp(1712858300567000000)
height = data.height.get_value_from_timestamp(1712858300567000000)
roll = data.roll.get_value_from_timestamp(1712858300567000000)
pitch = data.pitch.get_value_from_timestamp(1712858300567000000)
yaw = data.yaw.get_value_from_timestamp(1712858300567000000)
heave = data.heave.get_value_from_timestamp(1712858300567000000)
depth = data.depth.get_value_from_timestamp(1712858300567000000)
heading = data.heading.get_value_from_timestamp(1712858300567000000)
altitude = data.altitude.get_value_from_timestamp(1712858300567000000)
northing = data.northing.get_value_from_timestamp(1712858300567000000)
easting = data.easting.get_value_from_timestamp(1712858300567000000)


print("depth : ", depth)

#print(dt["/Platform/Position/northing"])