from lib.file_management import get_files, get_tree_from_file
from lib.DataManagement import DataManagement
from lib.Sonar import Sonar
from matplotlib import pyplot as plt
from lib.Point import Point
from lib.DatabaseCreator import DatabaseCreator

folder_path = "../Dataset Metric/"

files = get_files("./Dataset Metric/")
dts = []
i = 0
for filename in files:
    
    if(filename != "2024__1001410_0001.001_Binned" and i > 10 and i < 20):
        dt = get_tree_from_file(filename, "../Dataset Metric/")
        dts.append(dt)
    i+=1

s = Sonar(dts)


s.extract_lines_LF()

size = len(s.storage_LF[0].sample)
print("original size : ", size)
j = 0
for i, ping in enumerate(s.storage_LF):
    if(len(ping.sample) != size):
        print(i)
        print("taille trouvée : ", len(ping.sample))
        if(j > 10):
            break;
        j+=1
        

dc = DatabaseCreator(s.storage_LF, "./test.nc")
dc.create()



#imagettes = s.extract_imagette([s.storage_HF], 100, 0, Point(402508.16, 4654165.0))

#print(imagettes)

#plt.plot(ping_sonar.x_bathy, ping_sonar.z_bathy)

#print("profondeur : ", ping_sonar.depth, "m, floatness : ", ping_sonar.flat_seafloor_indicator, ", roughness : ", ping_sonar.rough_seafloor_indicator)

#plt.show()