import math

class Point:
    """
        Point on the earth

        Args:
            - eastern (float) : eastern reference using UTM system of the point
            - nothern (float) : nothern reference using UTM system of the point
            - heading (degree)
    """

    def __init__(self, eastern : float, northern : float, heading : float = 0):
        self.eastern : float = float(eastern);
        self.northern : float = float(northern);
        self.heading : float = float(heading); 

    def lateral_points_pos_sonar(self, dist=30.0, side = "starboard"):
        """
            e0 (m): eastering UTM reference
            n0 (m): nothering UTM reference
            heading (rad) : heading of the ship
        """
        dn = dist * math.sin(math.radians(self.heading))
        de = dist * math.cos(math.radians(self.heading))

        if(side == "starboard"):
            e=self.eastern+de
            n=self.northern-dn
        else:
            e=self.eastern-de
            n=self.northern+dn

        return Point(e, n, self.heading)
    
    def is_between(self, p1: Point, p2: Point, tolerance_meters=1.0) -> bool:
        """
            Check if the point (self) is between p1 (port) et p2 (starboard) with a given distance limit.
            We calculate distance between self and the line made by p1 and p2, if it is superior to tolerence, we return true

            p1 (Point) : port point to check if self is inside
            p2 (Point) : starboard point
            tolerance_meters (float) : distance tolerence to keep or reject the point
        """
        v_e = p2.eastern - p1.eastern
        v_n = p2.northern - p1.northern
        segment_len_sq = v_e**2 + v_n**2
        
        if segment_len_sq == 0:
            dist = math.sqrt((self.eastern - p1.eastern)**2 + (self.northern - p1.northern)**2)
            return dist <= tolerance_meters

        c_e = self.eastern - p1.eastern
        c_n = self.northern - p1.northern

        dot_product = c_e * v_e + c_n * v_n
        t = dot_product / segment_len_sq

        segment_len = math.sqrt(segment_len_sq)
        tolerance_ratio = tolerance_meters / segment_len
        
        if t < -tolerance_ratio or t > (1.0 + tolerance_ratio):
            return False 

        proj_e = p1.eastern + t * v_e
        proj_n = p1.northern + t * v_n
        
        distance_to_segment = math.sqrt((self.eastern - proj_e)**2 + (self.northern - proj_n)**2)

        return distance_to_segment <= tolerance_meters
