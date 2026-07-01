from typing import List
from .Point import Point
import numpy as np
import math

class SonarPing:
    
    def __init__(self, sample : List[int], point_ship : Point, point_starboard : Point, point_port : Point, roll : float, pitch : float, yaw : float, heave : float, heading : float, timestamp: int, depth : float, flat_seafloor_indicator : float, rough_seafloor_indicator: float, x_bathy: List[float] = None, z_bathy : List[float] = None, freq : str = "LF", sound_speed : float = 1475, filename = "", delta_time : float = 1.5359e-5, detection_range: float = 0):
        """
            Represent the response of a sonar ping and all measure that other captor hve made during this time
            Args:
                - sample (List[int]) : Reponse of the ping
                - point_ship (Point) : Location from the ship
                - point_starboard (Point) : Point from maximum measurement in starboard side
                - point_port (Point) : Point from the maximum measurement in port side
                - roll (float) : roll from the ship at the moment of the measure
                - pitch (float) : pitch from the ship at the moment of the measure
                - yaw (float) : yaw from the ship at the moment of the measure
                - heave (float) : heave of the ship at the moment of the measure
                - heading (float) : heading of the ship at the moment of the measure
                - timestamp (float) : timestamp of the measure, time when the ping is sended
                - depth (float) : distance between seafloor and the ship
                - flat_seafloor_indicator (float) : The slope of the seafloor given bathymetry
                - rough_seafloor_indicator (float) : 
                - x_bathy (List[float]) : perpendicular (to the ship's heading) distance wher e0 is the ship position and related to the depth in z_bathy
                - z_bathy (List[float]) : depth of the seafloor given for x_bathy
                - freq ("LF"|"HF") : frequency of the ping "LF" for Low Frequency or "HF" for High Frequency
                - sound_speed (float) : speed of the sound in the sea at this moment
                - filename (String) : filename of the netcdf file where the ping was extracted
                - delta_time (float) : sampling time
        """
        
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
        self.depth : float = depth
        self.flat_seafloor_indicator : float = flat_seafloor_indicator
        self.rough_seafloor_indicator : float = rough_seafloor_indicator
        self.x_bathy : float = x_bathy
        self.z_bathy : float = z_bathy
        self.freq : str = freq
        self.sound_speed : float = sound_speed
        self.filename : str = filename
        self.delta_time : float = delta_time
        self.detection_range = detection_range
        
        self.middle = len(sample) // 2
        
        sample_np = np.array(sample)
        
        sample_stbd = sample_np[self.middle:]
        sample_port = np.flip(sample_np[0:self.middle])

        indices = np.arange(len(sample_stbd))
        
        gains = (indices * delta_time * self.sound_speed / 2) ** 2

        sample_stbd = sample_stbd * gains
        sample_port = sample_port * gains

        self.sample_tvg = np.concatenate((np.flip(sample_stbd), sample_port))

    def get_starboard(self):
        """
            Get measure from the starboard side
        """
        middle = int(len(self.sample)/2)
        return self.sample[middle:len(self.sample)]

    def get_port(self):
        """
            Get measure from port side
        """
        middle = int(len(self.sample)/2)
        return np.flip(self.sample[0:middle])

    def get_index_depth_relative(self):
        """
            Get the index distance where the depth is reached in this sonar ping relative to middle
        """
        return int(-2*self.depth/(self.sound_speed*self.delta_time))
    
    def get_index_depth_absolute(self):
        """
            Get index position wher ethe depth is reached
        """
        i = self.get_index_depth_relative()
        middle = int(len(self.sample)/2)
        return middle-i, middle+i

    def get_starboard_without_nadir(self, security : int =50):
        """
            Get measure from starboard side without nadir
             - security : a step to be sure we aren't in the nadir
        """
        s = self.get_starboard()
        depth_i = self.get_index_depth_relative()
        return s[depth_i+security:len(s)]
    
    def get_port_without_nadir(self, security : int =50):
        """
            Get measure from port side without nadir
            - security : a step to be sure we aren't in the nadir
        """
        s = self.get_port()
        depth_i = self.get_index_depth_relative()
        return s[depth_i+security:len(s)]
    
    def detect_object_starboard(self, window_size : int =100, threshold_factor : float =3.5, security : int =150):
        """
            Detect variability in this sonar ping in starboard side
            - window_size (int) : size of the window to make the moving average
            - threshold_factor : the threshold for a sample to be considered intersting 
            - secuirty : a step to be sure we aren't in the nadir applied on each sonar ping measure
        """
        ping_data = self.get_starboard_without_nadir(security)
        if len(ping_data) == 0:
            return []

        kernel = np.ones(window_size) / window_size
        bruit_local = np.convolve(ping_data, kernel, mode='same')
        seuil_dynamique = bruit_local * threshold_factor

        masque_detection = (ping_data > seuil_dynamique)
        
        marge_bord = window_size // 2
        masque_detection[:marge_bord] = False
        
        masque_detection[-marge_bord:] = False

        detected_indices = np.where(masque_detection)[0]
        
        index_depth = self.get_index_depth_relative()
        middle = int(len(self.sample)/2)
        offset_absolu = middle + index_depth + security
        
        return [offset_absolu + x for x in detected_indices.tolist()]

    def detect_objet_port(self, window_size : int =100, threshold_factor : float = 3.5, security : int = 150):
        """
            Detect variability in this sonar ping in port side
            - window_size (int) : size of the window to make the moving average
            - threshold_factor (float) : the threshold for a sample to be considered intersting 
            - secuirty (int) : a step to be sure we aren't in the nadir applied on each sonar ping measure
        """
        ping_data = self.get_port_without_nadir(security)
        if len(ping_data) == 0:
            return []

        kernel = np.ones(window_size) / window_size
        bruit_local = np.convolve(ping_data, kernel, mode='same')
        seuil_dynamique = bruit_local * threshold_factor

        masque_detection = (ping_data > seuil_dynamique)
        
        marge_bord = window_size // 2
        masque_detection[:marge_bord] = False
        masque_detection[-marge_bord:] = False

        detected_indices = np.where(masque_detection)[0]
        if len(detected_indices) == 0:
            return []
        
        index_depth = self.get_index_depth_relative()
        middle = int(len(self.sample)/2)
        
        offset_absolu = middle - index_depth - security
        
        return [offset_absolu - x for x in detected_indices.tolist()]
    
    def get_position_from_index(self, index : int, sound_speed : float = 1475.0):

        """
            Returns position of an object from a given index of the ping
        """

        middle = int(len(self.sample)/2)

        side = "port"
        relative_index = abs(index-middle)
        if(index > middle):
            side = "starboard"
        
        distance = math.sqrt((sound_speed*self.delta_time*relative_index/2)**2-(self.depth)**2)


        target = self.point_ship.lateral_points_pos_sonar(distance, side)
        return (target.eastern, target.northern)
    