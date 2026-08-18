import netCDF4 as nc4
import numpy as np
from typing import List
from lib.SonarPing import SonarPing

class DatabaseCreator:
    def __init__(self, pings : List[SonarPing], path):
        """
            This class create a database in NetCDF format with ping information and other useful information like depth from given pingZ
        """
        self.pings = pings
        self.datasetNetCDF = nc4.Dataset(path, "w", format="NETCDF4", encoding='latin-1')

    def create(self):
        """
            Create database structure and insert data
        """
        ds = self.datasetNetCDF
        samples, pitch, roll, yaw, heave, heading, depth, nothering, eastering, nothering_starboard, eastering_starboard, nothering_port, eastering_port, flat_seafloor_indicator, rough_seafloor_indicator, timestamp, freq, delta_time = self.extract_data()

        if(len(samples) == 0):
            ds.close()
            return;

        ds.createDimension("nb_ping", len(samples))
        ds.createDimension("nb_sample", len(samples[0]))
        ds.createDimension("string_len", 3)

        var_samples = ds.createVariable("samples", "f4", ("nb_ping", "nb_sample"))

        var_pitch = ds.createVariable("pitch", "f4", ("nb_ping"))
        var_roll = ds.createVariable("roll", "f4", ("nb_ping"))
        var_yaw = ds.createVariable("yaw", "f4", ("nb_ping"))

        var_heave = ds.createVariable("heave", "f4", ("nb_ping"))
        var_heading = ds.createVariable("heading", "f4", ("nb_ping"))
        var_depth = ds.createVariable("depth", "f4", ("nb_ping"))

        var_nothering = ds.createVariable("nothering", "f8", ("nb_ping"))
        var_eastering = ds.createVariable("eastering", "f8", ("nb_ping"))

        var_nothering_starboard = ds.createVariable("nothering_starboard", "f8", ("nb_ping"))
        var_eastering_starboard = ds.createVariable("eastering_starboard", "f8", ("nb_ping"))

        var_nothering_port = ds.createVariable("nothering_port", "f8", ("nb_ping"))
        var_eastering_port = ds.createVariable("eastering_port", "f8", ("nb_ping"))

        var_flat_seafloor_indicator = ds.createVariable("flat_seafloor_indicator", "f4", ("nb_ping"))
        var_rough_seafloor_indicator = ds.createVariable("rough_seafloor_indicator", "f4", ("nb_ping"))

        var_timestamp = ds.createVariable("timestamp", "i8", ("nb_ping"))

        var_delta_time = ds.createVariable("delta_time", "f4", ("nb_ping"))

        #var_freq = ds.createVariable("freq", "c", ("nb_ping", "string_len"))

        var_samples[:, :] = samples;
        var_pitch[:] = pitch
        var_roll[:] = roll
        var_yaw[:] = yaw;
        var_heave[:] = heave;
        var_heading[:] = heading
        var_depth[:] = depth

        var_nothering[:] = nothering
        var_eastering[:] = eastering

        var_nothering_starboard[:] = nothering_starboard
        var_eastering_starboard[:] = eastering_starboard

        var_nothering_port[:] = nothering_port
        var_eastering_port[:] = eastering_port

        var_flat_seafloor_indicator[:] = flat_seafloor_indicator
        var_rough_seafloor_indicator[:] = rough_seafloor_indicator

        var_timestamp[:] = np.array(timestamp, dtype=np.int64)

        var_delta_time[:] = delta_time

        #var_freq[:] = freq

        ds.close()

    def extract_data(self):
        """
            Put all data from pings into lists
        """
        samples = []
        pitch = []
        roll = []
        yaw = []

        heave = []
        heading = []
        depth = []

        nothering = []
        eastering = []

        nothering_starboard = []
        eastering_starboard = []

        nothering_port = []
        eastering_port = []

        flat_seafloor_indicator = []
        rough_seafloor_indicator = []

        timestamp = []
        delta_time = []
        freq = []

        for ping in self.pings:
            samples.append(ping.sample)

            pitch.append(ping.pitch)
            roll.append(ping.roll)
            yaw.append(ping.yaw)

            heading.append(ping.heading)

            depth.append(ping.depth)

            nothering.append(ping.point_ship.northern)
            eastering.append(ping.point_ship.eastern)

            nothering_starboard.append(ping.point_starboard.northern)
            eastering_starboard.append(ping.point_starboard.eastern)

            nothering_port.append(ping.point_port.northern)
            eastering_port.append(ping.point_port.eastern)

            flat_seafloor_indicator.append(ping.flat_seafloor_indicator)
            rough_seafloor_indicator.append(ping.rough_seafloor_indicator)

            heave.append(ping.heave)
            timestamp.append(ping.timestamp)
            delta_time.append(ping.delta_time)
            freq.append(ping.freq)
        
        return samples, pitch, roll, yaw, heave, heading, depth, nothering, eastering, nothering_starboard, eastering_starboard, nothering_port, eastering_port, flat_seafloor_indicator, rough_seafloor_indicator, timestamp, freq, delta_time