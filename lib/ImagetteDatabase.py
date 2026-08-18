import numpy as np
from typing import Tuple, List
from .Point import Point

class ImagetteDatabase:
    def __init__(self, 
                 image: np.ndarray, 
                 survey_files: List[str],
                 detection_range: float,
                 frequency: str,
                 centered: bool,
                 side: str = "starboard",
                 pitch: np.ndarray = None,
                 roll: np.ndarray = None,
                 yaw: np.ndarray = None,
                 timestamp: np.ndarray = None,
                 heading: np.ndarray = None,
                 depth: np.ndarray = None,
                 heave: np.ndarray = None,
                 delta_time: np.ndarray = 1.53e-5,
                 sound_speed: np.ndarray = 1475.0,
                 point_eastern: Tuple[float, float] = None,
                 point_western: Tuple[float, float] = None,
                 point_southern: Tuple[float, float] = None,
                 point_northern: Tuple[float, float] = None,
                 eastern : float = 0,
                 nothern : float = 0,
                 ship_position : List[Tuple[float, float]] = None):
        
        # Données principales et métadonnées
        self.image = image
        self.survey_files = survey_files
        self.detection_range = detection_range
        self.frequency = frequency
        self.centered = centered
        self.side = side
        
        # Vecteurs d'attitude et capteurs (taille local_height)
        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw
        self.timestamp = timestamp
        self.heading = heading
        self.depth = depth
        self.heave = heave
        self.delta_time = delta_time
        self.sound_speed = sound_speed
        self.ship_position = ship_position
        
        # Extrémités géométriques (Easting, Northing)
        self.point_eastern = point_eastern
        self.point_western = point_western
        self.point_southern = point_southern
        self.point_northern = point_northern

        self.eastern = eastern
        self.nothern = nothern