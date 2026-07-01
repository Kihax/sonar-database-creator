from lib.SonarPing import SonarPing
from typing import List
import numpy as np
from .Point import Point

class Imagette:

    def __init__(self, pings : List[SonarPing], full_file_waterfall=None, start_y : int = 0, start_x : int = 0, centered : bool = False, side : str = "port"):
        self.pings = pings
        self.imagette = [ping.sample for ping in pings]
        self.full_file_waterfall = full_file_waterfall
        self.start_y = start_y
        self.start_x = start_x
        self.centered = centered
        self.side = side

        # Initialisation des 4 points extrêmes (objets de coordonnées ou tuples)
        self.point_most_eastern = None
        self.point_most_western = None
        self.point_most_northern = None
        self.point_most_southern = None

        if pings:
            # On extrait TOUS les points géoréférencés valides de la fauchée
            all_points = []
            for ping in pings:
                for pt in [ping.point_ship, ping.point_starboard, ping.point_port]:
                    if pt is not None:
                        all_points.append(pt)

            if all_points:
                # On trouve les points spécifiques qui atteignent les extremums
                self.point_most_western : Point = min(all_points, key=lambda pt: pt.eastern)
                self.point_most_eastern : Point = max(all_points, key=lambda pt: pt.eastern)
                self.point_most_southern : Point = min(all_points, key=lambda pt: pt.northern)
                self.point_most_northern : Point = max(all_points, key=lambda pt: pt.northern)

                # Rétrocompatibilité avec tes variables min/max
                self.eastern_min = self.point_most_western.eastern
                self.eastern_max = self.point_most_eastern.eastern
                self.northern_min = self.point_most_southern.northern
                self.northern_max = self.point_most_northern.northern
        else:
            self.eastern_min = self.eastern_max = 0.0
            self.northern_min = self.northern_max = 0.0

    def getImage(self):
        return np.array(self.imagette)