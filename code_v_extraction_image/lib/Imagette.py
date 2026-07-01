from .Point import Point
from typing import List

class Imagette:

    """
        Class to represent an Imagette that includes 
    """
    def __init__(self, value : List[List[int]], roll : List[float], pitch : List[float], yaw : List[float], heave : List[float], heading : List[float], ship_positions : List[Point], nothern_min : List[float], nothern_max : List[float], eastern_min : List[float], eastern_max : List[float], timestamp : List[int], x_bathymetry : List[List[float]], z_bathymetry : List[List[float]]):
        self.value : List[List[int]] = value;
        self.roll : List[float] = roll;
        self.pitch : List[float] = pitch;
        self.yaw : List[float] = yaw;
        self.heave : List[float] = heave;
        self.heading : List[float] = heading;
        self.ship_positions : List[Point] = ship_positions;

        self.nothern_min : List[float] = nothern_min;
        self.nothern_max : List[float] = nothern_max
        self.eastern_min : List[float] = eastern_min;
        self.eastern_max : List[float] = eastern_max;
        self.timestamp : List[int] = timestamp
        self.x_bathymetry : List[List[float]] = x_bathymetry
        self.z_bathymetry : List[List[float]] = z_bathymetry