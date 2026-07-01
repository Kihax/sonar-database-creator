import os
import xarray as xr
from lib.SonarPing import SonarPing
from lib.Point import Point
from code.lib.ReadDatabasePing import ReadDatabase
from lib.vizualize_imagette import vizualize_imagette
from matplotlib import pyplot as plt

rd = ReadDatabase("./code/test.nc")
sonarPings = rd.get_sonarPings()
for sonarPing in sonarPings[10000:11000]:
    depth = sonarPing.depth
    dt = 1.152e-5

    index_depth = -depth/(1470*dt)
    print(depth)
    sample = sonarPing.sample
    middle = int(len(sample)/2)
    print(sonarPing.flat_seafloor_indicator)
    plt.axvline(x=middle-index_depth, color='red', linestyle='--', linewidth=1, label='seafloor')
    plt.axvline(x=middle+index_depth, color='red', linestyle='--', linewidth=1, label='seafloor')


    plt.plot(sample)
    plt.show()