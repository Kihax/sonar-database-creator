from .DataManagement import DataManagement
from .Imagette import Imagette
from .timetag_to_timestamp import timetag_to_timestamp
from .Point import Point
from .lateral_points_pos_sonar import lateral_points_pos_sonar
from .SonarPing import SonarPing
from typing import List
from .measure_depth import measure_depth

import numpy as np
import math

class Sonar:
    """
        Class to extract data from the sonar
        
        Args :
            - dts (List<DataTree>) : List of netCDF files
    """
    def __init__(self, dts):
        self.dts=dts
        self.storage_HF : List[SonarPing] = []
        self.storage_LF : List[SonarPing] = []

    def compute_bathymetry(self, ping_timetag, angle_bathymetry_port, angle_bathymetry_starboard, time_bathymetry_port, time_bathymetry_starboard, quality_bathymetry_starboard, quality_bathymetry_port):
        """
        Compute corrected bathymetry profile for a single ping.

        Args:
            ping_timetag: absolute timestamp of the ping in nanoseconds
            angle_bathymetry_port: port beam angles for the ping
            angle_bathymetry_starboard: starboard beam angles for the ping
            time_bathymetry_port: port travel times for the ping
            time_bathymetry_starboard: starboard travel times for the ping
            quality_bathymetry_starboard: starboard quality flags for the ping
            quality_bathymetry_port: port quality flags for the ping

        Returns:
            x_port, z_port, x_stbd, z_stbd
        """
        data = DataManagement(self.dts)

        bathy_port_0_timetag = ping_timetag + np.array(time_bathymetry_port) * 1e9
        bathy_stbd_0_timetag = ping_timetag + np.array(time_bathymetry_starboard) * 1e9

        roll_smp_port = np.array([
            data.roll.get_value_from_timestamp(bathy_port_0_timetag[k])
            for k in range(len(bathy_port_0_timetag))
        ])
        roll_smp_stbd = np.array([
            data.roll.get_value_from_timestamp(bathy_stbd_0_timetag[k])
            for k in range(len(bathy_stbd_0_timetag))
        ])

        bathy_port_0_angle = np.array(angle_bathymetry_port)
        bathy_port_0_angle_roll = bathy_port_0_angle - roll_smp_port - 0.18

        bathy_stbd_0_angle = np.array(angle_bathymetry_starboard)
        bathy_stbd_0_angle_roll = bathy_stbd_0_angle - roll_smp_stbd + 0.28

        time_port = np.array(time_bathymetry_port)
        time_stbd = np.array(time_bathymetry_starboard)

        x_port_roll = 1475.0 * time_port / 2.0 * np.sin(bathy_port_0_angle_roll * np.pi / 180.0)
        z_port_roll = -1475.0 * time_port / 2.0 * np.cos(bathy_port_0_angle_roll * np.pi / 180.0)

        x_stbd_roll = 1475.0 * time_stbd / 2.0 * np.sin(bathy_stbd_0_angle_roll * np.pi / 180.0)
        z_stbd_roll = -1475.0 * time_stbd / 2.0 * np.cos(bathy_stbd_0_angle_roll * np.pi / 180.0)

        heave_smp_port = np.array([
            data.heave.get_value_from_timestamp(bathy_port_0_timetag[k])
            for k in range(len(bathy_port_0_timetag))
        ])
        heave_smp_stbd = np.array([
            data.heave.get_value_from_timestamp(bathy_stbd_0_timetag[k])
            for k in range(len(bathy_stbd_0_timetag))
        ])

        bad_samples_port = np.array(quality_bathymetry_port) == 0
        x_port = x_port_roll[~bad_samples_port]
        z_port = (z_port_roll - heave_smp_port)[~bad_samples_port]

        bad_samples_stbd = np.array(quality_bathymetry_starboard) == 0
        x_stbd = x_stbd_roll[~bad_samples_stbd]
        z_stbd = (z_stbd_roll - heave_smp_stbd)[~bad_samples_stbd]

        return x_port, z_port, x_stbd, z_stbd


    # limit sample correspond à la limite pour les échantillons de port side et starside
    def extract_lines(self, f="LF", storage : List[SonarPing] = [], limit_sample : int = None):
        data = DataManagement(self.dts)
        
        for dt in self.dts:
            r = dt["/Sonar/SLS " + f + "/Port/Block_0"].attrs["range (m)"];
            samples_port = dt["/Sonar/SLS " + f + "/Port/Block_0"].samples.values
            samples_starboard = dt["/Sonar/SLS " + f + "/Starboard/Block_0"].samples.values

            if(limit_sample == None):
                limit_sample = len(samples_port[0])
                
            print("slice")
            timestamps = timetag_to_timestamp(dt["/Sonar/SLS " + f + "/Port/Block_0"].timetag.values);
            for i in range(len(timestamps)):
                
                timestamp = timestamps[i]
                sample_starboard = samples_starboard[i][0:limit_sample]
                sample_port = samples_port[i][0:limit_sample]

                angle_bathymetry_port = dt['/Sonar/Bathymetry/Port/Block_0/angle'][i, :].values
                angle_bathymetry_starboard = dt['/Sonar/Bathymetry/Starboard/Block_0/angle'][i, :].values
                time_bathymetry_port = dt['/Sonar/Bathymetry/Port/Block_0/time'][i, :].values
                time_bathymetry_starboard = dt['/Sonar/Bathymetry/Starboard/Block_0/time'][i, :].values
                quality_bathymetry_port = dt['/Sonar/Bathymetry/Port/Block_0/quality'][i, :].values
                quality_bathymetry_starboard = dt['/Sonar/Bathymetry/Starboard/Block_0/quality'][i, :].values
                ping_timetag = timetag_to_timestamp(dt['/Sonar/Bathymetry/Port/Block_0/timetag'][i].values)

                x_port, z_port, x_stbd, z_stbd = self.compute_bathymetry(
                    ping_timetag,
                    angle_bathymetry_port,
                    angle_bathymetry_starboard,
                    time_bathymetry_port,
                    time_bathymetry_starboard,
                    quality_bathymetry_starboard,
                    quality_bathymetry_port,
                )
                x_bathymetry = np.concatenate([x_port, x_stbd])
                z_bathymetry = np.concatenate([z_port, z_stbd])

                sample = np.zeros(2*limit_sample)
                sample[0:limit_sample] = np.flip(sample_port)
                sample[limit_sample:2*limit_sample] = sample_starboard

                #latitude = data.latitude.get_value_from_timestamp(timestamp)
                #longitude = data.latitude.get_value_from_timestamp(timestamp)
                    
                eastern = data.easting.get_value_from_timestamp(timestamp)
                nothern = data.northing.get_value_from_timestamp(timestamp)

                if(data.heading.get_value_from_timestamp(timestamp) == None):
                    print(data.heading.get_values())

                # heading is already stored in degrees in the input data
                heading = data.heading.get_value_from_timestamp(timestamp)
                roll = data.roll.get_value_from_timestamp(timestamp)
                pitch = data.pitch.get_value_from_timestamp(timestamp)
                yaw = data.yaw.get_value_from_timestamp(timestamp)
                heave = data.heave.get_value_from_timestamp(timestamp);
                sound_speed = data.sound_speed.get_value_from_timestamp(timestamp);

                depth_starboard = measure_depth(sample_starboard, sound_speed)

                depth_port = measure_depth(sample_port, sound_speed)

                # point_starboard / point_port doivent représenter la portée maximum
                # accessible sur chaque bordée, pas la portée horizontale réduite
                # par la profondeur du sol.
                (starboard_eastern, starboard_nothern) = lateral_points_pos_sonar(eastern, nothern, math.radians(heading), r, "starboard");
                (port_eastern, port_nothern) = lateral_points_pos_sonar(eastern, nothern, math.radians(heading), r, "port");

                ship = Point(eastern, nothern)
                starboard = Point(starboard_eastern, starboard_nothern)
                port = Point(port_eastern, port_nothern)

                storage.append(SonarPing(sample, ship, starboard, port, roll, pitch, yaw, heave, heading, timestamp, x_bathymetry, z_bathymetry))
        
        # sort timestamp to avoid big step in files
        storage.sort(key=lambda ping: ping.timestamp)

    def extract_lines_HF(self, limit_sample : int=None):
        self.extract_lines("HF", self.storage_HF, limit_sample)

    def extract_lines_LF(self, limit_sample : int=None):
        self.extract_lines("LF", self.storage_LF, limit_sample)

    def extract_imagette(self, extract_lines, storage : List[SonarPing], nb_sample : int, nb_ping : int):
        extract_lines(nb_sample)
        imagettes = []

        for j in range(0, int(len(storage)/nb_ping) ):

            val = [ ];
            rolls = []
            yaws = []
            pitchs = []
            heaves = []
            heading = []
            timestamp = []
            ship_positions : List[Point] = []
            port_positions : List[Point] = []
            starboard_positions : List[Point] = []
            x_bathymetry : List[float] = []
            z_bathymetry : List[float] = []



            for i in range(nb_ping*j, nb_ping*(j+1)):
                line : SonarPing = storage[i]
                val.append(line.sample)
                ship_positions.append(line.point_ship)
                starboard_positions.append(line.point_starboard)
                port_positions.append(line.point_port)
                

                rolls.append(line.roll)
                yaws.append(line.yaw)
                pitchs.append(line.pitch)
                heaves.append(line.heave)
                heading.append(line.heading)
                timestamp.append(line.timestamp)

                x_bathymetry.append(line.x_bathymetry)
                z_bathymetry.append(line.z_bathymetry)
            
            all_eastern = [p.eastern for p in ship_positions + starboard_positions + port_positions]
            all_nothern = [p.nothern for p in ship_positions + starboard_positions + port_positions]

            eastern_min = min(all_eastern)
            eastern_max = max(all_eastern)

            nothern_min = min(all_nothern)
            nothern_max = max(all_nothern)
        
            imagettes.append(Imagette(val, rolls, pitchs, yaws, heaves, heading, ship_positions, nothern_min, nothern_max, eastern_min, eastern_max, timestamp, x_bathymetry, z_bathymetry))
        
        return imagettes;

    def extract_imagette_HF(self, nb_sample : int, nb_ping : int):
        return self.extract_imagette(self.extract_lines_HF, self.storage_HF, nb_sample, nb_ping)
    
    def extract_imagette_LF(self, nb_sample : int, nb_ping : int):
        return self.extract_imagette(self.extract_lines_LF, self.storage_LF, nb_sample, nb_ping)