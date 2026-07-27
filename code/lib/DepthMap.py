import numpy as np
from typing import List
from lib.DataManagement import DataManagement
from .timetag_to_timestamp import timetag_to_timestamp

class DepthMap:
    def __init__(self, dts, data : DataManagement):
        self.dts = dts
        self.depth_map = {}
        self.data = data

    def extract_depth_map(self):
        for dt in self.dts:
            filename = dt.attrs.get("Survey file")
            # Skip specific faulty or template files
            if filename == "2024__1001410_0001.001_Binned":
                continue
            print(f"Processing file: {filename}")

            bathy_port_path = "/Sonar/Bathymetry/Port/Block_0"
            bathy_stbd_path = "/Sonar/Bathymetry/Starboard/Block_0"

            # Pre-load bathymetry data to avoid repeating heavy disk I/O in the loop
            angles_b_port = dt[f"{bathy_port_path}/angle"].load().values
            angles_b_stbd = dt[f"{bathy_stbd_path}/angle"].load().values
            times_b_port = dt[f"{bathy_port_path}/time"].load().values
            times_b_stbd = dt[f"{bathy_stbd_path}/time"].load().values
            qualities_b_port = dt[f"{bathy_port_path}/quality"].load().values
            qualities_b_stbd = dt[f"{bathy_stbd_path}/quality"].load().values
            bathy_timetags = dt[f"{bathy_port_path}/timetag"].load().values

            
            for i in range(len(bathy_timetags)):
                ping_timestamp = timetag_to_timestamp(bathy_timetags[i])

                x_port, z_port, x_stbd, z_stbd = self.compute_bathymetry(
                    ping_timestamp,
                    angles_b_port[i],
                    angles_b_stbd[i],
                    times_b_port[i],
                    times_b_stbd[i],
                    qualities_b_stbd[i],
                    qualities_b_port[i],
                )

                depth = abs((z_port[-1] + z_stbd[0]) / 2)

                # Concatenate and sort bathymetry profile
                x_bathymetry = np.concatenate([x_port, x_stbd])
                z_bathymetry = np.concatenate([z_port, z_stbd])

                sorted_indices = np.argsort(x_bathymetry)
                x_bathymetry = x_bathymetry[sorted_indices]
                z_bathymetry = z_bathymetry[sorted_indices]

                northing, easting = self.data.get_position_corr(ping_timestamp)

                flat_seafloor_indicator = self.is_flat(x_bathymetry, z_bathymetry)
                rough_seafloor_indicator = self.calculate_roughness(
                    x_bathymetry, z_bathymetry
                )

                self.depth_map[(northing, easting)] = {
                    "depth": depth,
                    "flat_seafloor_indicator": flat_seafloor_indicator,
                    "rough_seafloor_indicator": rough_seafloor_indicator
                }

    def compute_bathymetry(self, ping_timestamp : int, angle_bathymetry_port : List[float], angle_bathymetry_starboard : List[float], time_bathymetry_port : List[float], time_bathymetry_starboard : List[float], quality_bathymetry_starboard : List[int], quality_bathymetry_port : List[int], sound_speed : float = 1475.0):
        """
        Compute corrected bathymetry profile for a single ping.

        Args:
            ping_timestamp: absolute timestamp of the ping in nanoseconds
            angle_bathymetry_port: port beam angles for the ping
            angle_bathymetry_starboard: starboard beam angles for the ping
            time_bathymetry_port: port travel times for the ping
            time_bathymetry_starboard: starboard travel times for the ping
            quality_bathymetry_starboard: starboard quality flags for the ping
            quality_bathymetry_port: port quality flags for the ping
            data: optional DataManagement instance to avoid recreation

        Returns:
            x_port, z_port, x_stbd, z_stbd
        """

        data = self.data

        time_bathymetry_port = np.array(time_bathymetry_port)
        time_bathymetry_starboard = np.array(time_bathymetry_starboard)
        
        bathy_port_0_timetag = ping_timestamp + time_bathymetry_port * 1e9
        bathy_stbd_0_timetag = ping_timestamp + time_bathymetry_starboard * 1e9


        # Vectorized method calls - much faster than list comprehensions
        roll_smp_port = data.roll.get_values_from_timestamps(bathy_port_0_timetag)
        roll_smp_stbd = data.roll.get_values_from_timestamps(bathy_stbd_0_timetag)

        bathy_port_0_angle = np.array(angle_bathymetry_port)
        bathy_port_0_angle_roll = bathy_port_0_angle - roll_smp_port - 0.18

        bathy_stbd_0_angle = np.array(angle_bathymetry_starboard)
        bathy_stbd_0_angle_roll = bathy_stbd_0_angle - roll_smp_stbd + 0.28

        time_port = time_bathymetry_port
        time_stbd = time_bathymetry_starboard

        x_port_roll = sound_speed * time_port / 2.0 * np.sin(bathy_port_0_angle_roll * np.pi / 180.0)
        z_port_roll = -sound_speed * time_port / 2.0 * np.cos(bathy_port_0_angle_roll * np.pi / 180.0)

        x_stbd_roll = sound_speed * time_stbd / 2.0 * np.sin(bathy_stbd_0_angle_roll * np.pi / 180.0)
        z_stbd_roll = -sound_speed * time_stbd / 2.0 * np.cos(bathy_stbd_0_angle_roll * np.pi / 180.0)

        # Vectorized method calls instead of loops
        heave_smp_port = data.heave.get_values_from_timestamps(bathy_port_0_timetag)
        heave_smp_stbd = data.heave.get_values_from_timestamps(bathy_stbd_0_timetag)

        bad_samples_port = np.array(quality_bathymetry_port) == 0
        x_port = x_port_roll[~bad_samples_port]
        z_port = (z_port_roll - heave_smp_port)[~bad_samples_port]

        bad_samples_stbd = np.array(quality_bathymetry_starboard) == 0
        x_stbd = x_stbd_roll[~bad_samples_stbd]
        z_stbd = (z_stbd_roll - heave_smp_stbd)[~bad_samples_stbd]

        return x_port, z_port, x_stbd, z_stbd