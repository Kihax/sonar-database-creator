from typing import List
from .Point import Point

class SonarPing:
    """
        Represent the response of a sonar ping and all measure that other captor hve made during this time

        Args :
            - sample (List<int>) : All value measured by the sonar
            - point_ship (Point) : Point of the ship (latitude and longitude)
            - point_starboard (Point) : Maximum point that can reach the sonar on the starboard side
            - point_port (Point) : Maximum point that can reach the sonar on the port side
            - roll (float) : Roll measured at the moment sonar made his measure
            - pitch (float) : Pitch  measured at the moment sonar made his measure
            - yaw (flaot) : yaw  measured at the moment sonar made his measure
            - heave (float) : 
            - heading (flaot) : 
            - timestamp (int)
    """
    def __init__(self, sample : List[int], point_ship : Point, point_starboard : Point, point_port : Point, roll : float, pitch : float, yaw : float, heave : float, heading : float, timestamp: int, x_bathymetry : List[float] = None, z_bathymetry : List[float] = None):
        self.sample : List[int] = sample
        self.point_ship : Point = point_ship
        self.point_starboard : Point = point_starboard
        self.point_port : Point = point_port
        self.roll : float = roll
        self.pitch : float = pitch
        self.yaw : float = yaw
        self.heave : float = heave
        self.heading : float = heading
        self.timestamp : int = timestamp
        self.x_bathymetry : List[float] = x_bathymetry
        self.z_bathymetry : List[float] = z_bathymetry
