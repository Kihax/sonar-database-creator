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