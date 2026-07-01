import netCDF4 as nc4
import numpy as np
from .Imagette import Imagette
from typing import List, Dict

class DatabaseCreator:
    def __init__(self, imagettes : List[Imagette], path):
        self.imagettes : List[Imagette] = imagettes
        self.datasetNetCDF = nc4.Dataset(path, "w", format="NETCDF4", encoding='latin-1')
        
    def create(self):
        ds = self.datasetNetCDF
        image, roll, pitch, yaw, heave, heading, eastering, nothering, nothering_min, nothering_max, eastering_min, eastering_max, timestamp, x_bathymetry, z_bathymetry = self.extract_data()

        print(f"Structure de la matrice timestamp : {(len(timestamp), len(timestamp[0]) if timestamp else 0)}")
        print("Format : (Nombre d'imagettes, Nombre de pings par imagette)")
        print("-" * 60)


        ds.createDimension("index_imagette", len(image))
        ds.createDimension("nb_ping_imagette", len(image[0]))
        ds.createDimension("nb_sample_imagette", len(image[0][0]))

        var_image = ds.createVariable("image", "f4", ("index_imagette", "nb_ping_imagette", "nb_sample_imagette"))

        var_roll = ds.createVariable("roll", "f4", ("index_imagette", "nb_ping_imagette"))
        var_pitch = ds.createVariable("pitch", "f4", ("index_imagette", "nb_ping_imagette"))
        var_yaw = ds.createVariable("yaw", "f4", ("index_imagette", "nb_ping_imagette"))

        var_heave = ds.createVariable("heave", "f4", ("index_imagette", "nb_ping_imagette"))
        var_heading = ds.createVariable("heading", "f4", ("index_imagette", "nb_ping_imagette"))

        var_eastering = ds.createVariable("easting", "f4", ("index_imagette", "nb_ping_imagette"))
        var_nothering = ds.createVariable("northing", "f4", ("index_imagette", "nb_ping_imagette"))

        var_timestamp = ds.createVariable("timestamp", "i8", ("index_imagette", "nb_ping_imagette"))

        var_nothering_min = ds.createVariable("northing_min", "f4", ("index_imagette"))
        var_nothering_max = ds.createVariable("northing_max", "f4", ("index_imagette"))

        var_eastering_min = ds.createVariable("easting_min", "f4", ("index_imagette"))
        var_eastering_max = ds.createVariable("easting_max", "f4", ("index_imagette"))

        vlen_type = ds.createVLType(np.float32, 'vlen_float32')
        var_x_bathymetry = ds.createVariable("x_bathymetry", vlen_type, ("index_imagette", "nb_ping_imagette"))
        var_z_bathymetry = ds.createVariable("z_bathymetry", vlen_type, ("index_imagette", "nb_ping_imagette"))


        var_image[:, :, :] = image;
        var_roll[:, :] = roll
        var_pitch[:, :] = pitch
        var_yaw[:, :] = yaw
        var_heave[:, :] = heave
        var_heading[:, :] = heading
        var_eastering[:, :] = eastering
        var_nothering[:, :] = nothering
        timestamp = np.array(timestamp, dtype=np.int64)
        var_timestamp[:, :] = timestamp
        var_nothering_min[:] = nothering_min
        var_nothering_max[:] = nothering_max
        var_eastering_min[:] = eastering_min
        var_eastering_max[:] = eastering_max
        
        for i in range(len(x_bathymetry)):
            for j in range(len(x_bathymetry[i])):
                xp = x_bathymetry[i][j]
                zp = z_bathymetry[i][j]
                if xp is None:
                    var_x_bathymetry[i, j] = np.array([], dtype=np.float32)
                else:
                    var_x_bathymetry[i, j] = np.array(xp, dtype=np.float32)

                if zp is None:
                    var_z_bathymetry[i, j] = np.array([], dtype=np.float32)
                else:
                    var_z_bathymetry[i, j] = np.array(zp, dtype=np.float32)



        ds.close()

    def extract_data(self):
        # extracting data
        image = []
        roll = []
        pitch = []
        yaw = []
        heave = []
        heading = []
        nothering = []
        eastering = []

        nothering_min = []
        nothering_max = []

        eastering_min = []
        eastering_max = []
        timestamp = []

        x_bathymetry = []
        z_bathymetry = []

        for i in range(len(self.imagettes)):
            imagette  = self.imagettes[i]

            image.append(imagette.value) # [ [  [1, 2, 3], [1, 25, 5], ... ], ...  ]

            roll.append(imagette.roll) # [ [1, 5, 8], [6, 4, 9], ... ]
            pitch.append(imagette.pitch)
            yaw.append(imagette.yaw)
            heave.append(imagette.heave)
            heading.append(imagette.heading)

            eastering.append([])
            nothering.append([])

            for pos in imagette.ship_positions:
                eastering[i].append(pos.eastern)
                nothering[i].append(pos.nothern)

            nothering_min.append(imagette.nothern_min)
            nothering_max.append(imagette.nothern_max)

            eastering_min.append(imagette.eastern_min)
            eastering_max.append(imagette.eastern_max)

            timestamp.append(imagette.timestamp)

            x_bathymetry.append(imagette.x_bathymetry)
            z_bathymetry.append(imagette.z_bathymetry)

        return (image, roll, pitch, yaw, heave, heading, eastering, nothering, nothering_min, nothering_max, eastering_min, eastering_max, timestamp, x_bathymetry, z_bathymetry)
    
        