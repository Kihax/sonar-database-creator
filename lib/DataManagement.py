import numpy as np

from .Measurement import Measurement

class DataManagement:
    """
    Extract all available data from given DataTree
    
    Args:
        dts : list of DataTree from netCDF files
    """
    def __init__(self, dts):
        self.data = dts
        self.latitude = Measurement('/Platform/Position/latitude', dts, "timetag", offset_path='/Platform/Position', correct_offset=False, offset_index=0)
        self.longitude = Measurement('/Platform/Position/longitude', dts, "timetag", offset_path='/Platform/Position', correct_offset=False, offset_index=1)
        self.height = Measurement('/Platform/Position/height', dts, "timetag", offset_path='/Platform/Position', correct_offset=False, offset_index=2)

        # UTM Reference
        self.northing = Measurement('/Platform/Position/northing', dts, "timetag", offset_path='/Platform/Position', correct_offset=False, offset_index=1, is_bit64=True)
        self.easting = Measurement('/Platform/Position/easting', dts, "timetag", offset_path='/Platform/Position', correct_offset=False, offset_index=1, is_bit64=True)

        self.sound_speed = Measurement('/Environment/Sound Velocity', dts, "time", extra_key="sound_velocity")
        
        self.roll = Measurement('/Platform/Attitude/roll', dts, "timetag", offset_path="/Platform/Attitude", correct_offset=False, offset_index=0, approximate_method="linear_interpolation_angle")
        self.pitch = Measurement('/Platform/Attitude/pitch', dts, "timetag", offset_path="/Platform/Attitude", correct_offset=False, offset_index=1, approximate_method="linear_interpolation_angle")
        self.yaw = Measurement('/Platform/Attitude/yaw', dts, "timetag", offset_path="/Platform/Attitude", correct_offset=False, offset_index=2, approximate_method="linear_interpolation_angle")

        self.heave = Measurement('/Platform/Heave', dts, "timetag", extra_key="heave", offset_path="/Platform/Heave", correct_offset=False, offset_index=0)
        self.depth = Measurement('/Platform/Depth', dts, "timetag", extra_key="depth", offset_path="/Platform/Depth", correct_offset=False, offset_index=0)

        self.heading = Measurement('/Platform/Heading', dts, "timetag", extra_key="heading", offset_path='/Platform/Heading', correct_offset=False, offset_index=0, approximate_method="linear_interpolation_angle")
        self.altitude = Measurement('/Platform/Altitude', dts, "timetag", extra_key="altitude", offset_path='/Platform/Altitude', correct_offset=False, offset_index=0)

    def get_position_corr(self, timestamp):
        """
        Return corrected UTM position (easting, northing) for a timestamp.

        The correction applies the vessel attitude (roll, pitch, yaw) and the
        lever-arm offsets to the GPS reference point, following the same
        rotation logic used in the sonar processing pipeline.
        """
        eastern = self.easting.get_value_from_timestamp(timestamp)
        northern = self.northing.get_value_from_timestamp(timestamp)
        roll = self.roll.get_value_from_timestamp(timestamp)
        pitch = self.pitch.get_value_from_timestamp(timestamp)
        yaw = self.yaw.get_value_from_timestamp(timestamp)
        heading = self.heading.get_value_from_timestamp(timestamp, with_quality=False)
        heave = self.heave.get_value_from_timestamp(timestamp)

        if None in (eastern, northern, roll, pitch, heading, heave):
            return None, None

        if yaw is None:
            yaw = heading

        r_rad = np.radians(roll)
        p_rad = np.radians(pitch)
        y_rad = np.radians(yaw)

        R_roll = np.array([
            [np.cos(r_rad), 0, np.sin(r_rad)],
            [0, 1, 0],
            [-np.sin(r_rad), 0, np.cos(r_rad)],
        ])

        R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(p_rad), -np.sin(p_rad)],
            [0, np.sin(p_rad), np.cos(p_rad)],
        ])

        R_yaw = np.array([
            [np.sin(y_rad), np.cos(y_rad), 0],
            [np.cos(y_rad), -np.sin(y_rad), 0],
            [0, 0, 1],
        ])

        R = R_yaw @ R_pitch @ R_roll

        lever_arm_nav = np.array([0.0, 0.33, -2.836], dtype=float)
        lever_arm_nav_rotated = R @ lever_arm_nav

        vessel_east = eastern - lever_arm_nav_rotated[0]
        vessel_north = northern - lever_arm_nav_rotated[1]
        vessel_z = -heave  # Référence surface

        lever_arm_transducer = np.array([0.0, 0.297, 0.338], dtype=float)
        lever_arm_trans_rotated = R @ lever_arm_transducer

        sonar_east = vessel_east + lever_arm_trans_rotated[0]
        sonar_north = vessel_north + lever_arm_trans_rotated[1]

        return float(sonar_east), float(sonar_north)

    def get_easting_corr(self, timestamp):
        """
        Get corrected easting value for a given timestamp.
        """
        position = self.get_position_corr(timestamp)
        if position is None:
            return None
        return position[0]

    def get_northing_corr(self, timestamp):
        """
        Get corrected northing value for a given timestamp.
        
        Args:
            timestamp: absolute timestamp in nanoseconds
        """
        position = self.get_position_corr(timestamp)
        if position is None:
            return None
        return position[1]
