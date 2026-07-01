#
# This is the trajectory from the ship with latitude and longitude given by the gps.
# The gps is on the top of a mast so the latitude and longitude might be corrected
#

import matplotlib.pyplot as plt

from lib.file_management import get_all_latitude_longitude, get_files, get_tree_from_index


latitude, longitude = get_all_latitude_longitude()

plt.plot(latitude, longitude, "x")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.title("Position of the platform")
plt.show()