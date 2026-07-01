import xarray as xr
import math
import numpy as np

from .SonarPing import SonarPing
from .Point import Point
from .Imagette import Imagette
from .ImagetteDatabase import ImagetteDatabase

from typing import List, Tuple

class ReadDatabaseImagette:

    def __init__(self, dts):
        self.dts = dts
        self.pos_imagette = {}

    def extract(self):
        for dt in self.dts:
            for group_name in dt.groups:
                # On nettoie le nom pour l'analyse
                clean_name = group_name.lstrip('/')
                
                # S'il reste un '/' c'est un sous-groupe profond, on passe
                if '/' in clean_name:
                    continue
                
                group = dt[group_name]

                # SÉCURITÉ : On vérifie si 'easting' et 'northing' sont présents dans les attributs
                if 'easting' not in group.attrs or 'northing' not in group.attrs:
                    continue

                x_val = group.attrs['easting']
                y_val = group.attrs['northing']
                coord = (x_val, y_val)

                # Parcours des sous-groupes (imagette_0, imagette_1, etc.)
                for sub_group_name in group.groups:
                    sub_group = group[sub_group_name]
                    
                    if "image" in sub_group.data_vars:
                        # 1. Extraction des attributs de l'imagette
                        # Gestion de la liste de fichiers (peut être un string unique ou une liste)
                        s_files = sub_group.attrs.get("survey_files", [])
                        if isinstance(s_files, str):
                            s_files = [s_files]
                        elif isinstance(s_files, np.ndarray):
                            s_files = s_files.tolist()

                        detection_range = float(sub_group.attrs.get("detection_range", 0.0))
                        frequency = sub_group.attrs.get("frequency", "LF")
                        centered = sub_group.attrs.get("centered", "false") == "true"
                        side = sub_group.attrs.get("side", "")

                        # 2. Extraction des variables (converties en numpy arrays)
                        image_lazy = sub_group["image"]
                        pitch = sub_group["pitch"].values if "pitch" in sub_group.data_vars else np.array([])
                        roll = sub_group["roll"].values if "roll" in sub_group.data_vars else np.array([])
                        yaw = sub_group["yaw"].values if "yaw" in sub_group.data_vars else np.array([])
                        timestamp = sub_group["timestamp"].values if "timestamp" in sub_group.data_vars else np.array([])
                        heading = sub_group["heading"].values if "heading" in sub_group.data_vars else np.array([])
                        depth = sub_group["depth"].values if "depth" in sub_group.data_vars else np.array([])
                        heave = sub_group["heave"].values if "heave" in sub_group.data_vars else np.array([])
                        delta_time = sub_group["delta_time"].values if "delta_time" in sub_group.data_vars else np.array([])
                        sound_speed = sub_group["sound_speed"].values if "sound_speed" in sub_group.data_vars else np.array([])

                        # 3. Extraction des points cardinaux (tableaux de taille 2 : [easting, northing])
                        p_eastern = tuple(sub_group["point_eastern"].values) if "point_eastern" in sub_group.data_vars else (0.0, 0.0)
                        p_western = tuple(sub_group["point_western"].values) if "point_western" in sub_group.data_vars else (0.0, 0.0)
                        p_southern = tuple(sub_group["point_southern"].values) if "point_southern" in sub_group.data_vars else (0.0, 0.0)
                        p_northern = tuple(sub_group["point_nothern"].values) if "point_nothern" in sub_group.data_vars else (0.0, 0.0)
                        ship_positions = [tuple(p) for p in sub_group["ship_position"].values]


                        # Instanciation de l'objet ImagetteDatabase complet
                        imagette_obj = ImagetteDatabase(
                            image=image_lazy,
                            survey_files=s_files,
                            detection_range=detection_range,
                            frequency=frequency,
                            centered=centered,
                            side=side,
                            pitch=pitch,
                            roll=roll,
                            yaw=yaw,
                            timestamp=timestamp,
                            heading=heading,
                            depth=depth,
                            heave=heave,
                            delta_time=delta_time,
                            sound_speed=sound_speed,
                            point_eastern=p_eastern,
                            point_western=p_western,
                            point_southern=p_southern,
                            point_northern=p_northern,
                            eastern=x_val,
                            nothern=y_val,
                            ship_position=ship_positions
                        )

                        # Ajout dans le dictionnaire de coordonnées
                        if coord not in self.pos_imagette:
                            self.pos_imagette[coord] = []
                        self.pos_imagette[coord].append(imagette_obj)
            
    def get_groups_with_n_imagettes_containing_point(self, global_imagettes, target_point: Point, n: int):
        """
        Parcourt le dictionnaire et renvoie les positions (groupes complets) 
        qui possèdent au moins 'n' imagettes contenant le point cible.
        
        Args:
            global_imagettes (dict): Le dictionnaire pos_imagette extrait de la DB
            target_point (Point): L'objet Point recherché
            n (int): Le nombre minimal d'imagettes qui doivent contenir le point
        """
        matching_groups = {}

        for coord, list_imagettes in global_imagettes.items():
            match_count = 0
            
            for img_obj in list_imagettes:
                # 1. Extraction des coordonnées UTM de chaque coin (Est et Nord)
                eastings = [
                    img_obj.point_eastern[0], 
                    img_obj.point_western[0], 
                    img_obj.point_southern[0], 
                    img_obj.point_northern[0]
                ]
                northings = [
                    img_obj.point_eastern[1], 
                    img_obj.point_western[1], 
                    img_obj.point_southern[1], 
                    img_obj.point_northern[1]
                ]
                
                # 2. Définition de la Bounding Box de l'imagette
                min_e, max_e = min(eastings), max(eastings)
                min_n, max_n = min(northings), max(northings)
                
                # 3. Si le point est dedans, on incrémente notre compteur
                if (min_e <= target_point.eastern <= max_e) and (min_n <= target_point.northern <= max_n):
                    match_count += 1
                    
            # On garde TOUT le groupe si le nombre d'imagettes valides est suffisant
            if match_count >= n:
                matching_groups[coord] = list_imagettes
                print(f"Position {coord} retenue : {match_count}/{len(list_imagettes)} imagettes contiennent le point.")

        return matching_groups
