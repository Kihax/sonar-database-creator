import numpy as np
import math


def lateral_points_pos_sonar(e0, n0, heading, dist=30.0, side = "starboard"):
    """
        e0 (m): eastering UTM reference
        n0 (m): nothering UTM reference
        heading (rad) : heading of the ship
    """
    dn = dist * math.sin(heading)
    de = dist * math.cos(heading)

    e = 0
    n = 0
    if(side == "starboard"):
        e=e0+de
        n=n0-dn
    else:
        e=e0-de
        n=n0+dn

    return (e, n)