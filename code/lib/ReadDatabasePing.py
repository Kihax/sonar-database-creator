import xarray as xr
import math
import numpy as np

from .SonarPing import SonarPing
from .Point import Point
from .Imagette import Imagette

from typing import List, Tuple

class ReadDatabasePing:
    def __init__(self, dts):
        """
            Read database and can extract ping for given DataTree (only work from DataTree from refactored_data)

            Args : 
             - dts (List[DataTree]) : List of DataTree to extract Ping
        """
        self.dts = dts
    
    def get_sonarPings(self) -> List[SonarPing]:
        """
            Extract ping in DataTree given and returns them
        """
        sonarPings : List[SonarPing] = []
        for i, dt in enumerate(self.dts):
            try:
                samples = dt["samples"].values
                pitchs = dt["pitch"].values
                rolls = dt["roll"].values
                yaws = dt["yaw"].values
                heaves = dt["heave"].values
                depths = dt["depth"].values
                headings = dt["heading"].values
                notherings = dt["nothering"].values.astype(np.float64)
                easterings = dt["eastering"].values.astype(np.float64)
                eastering_starboards = dt["eastering_starboard"].values.astype(np.float64)
                nothering_starboards = dt["nothering_starboard"].values.astype(np.float64)
                nothering_ports = dt["nothering_port"].values.astype(np.float64)
                eastering_ports = dt["eastering_port"].values.astype(np.float64)
                flat_seafloor_indicators = dt["flat_seafloor_indicator"].values
                rough_seafloor_indicators = dt["rough_seafloor_indicator"].values
                timestamps = dt["timestamp"].values
                filename = dt["filename"].values;
                

                for i in range(len(samples)):
                    sample = samples[i]
                    pitch = pitchs[i]
                    roll = rolls[i]
                    yaw = yaws[i]
                    heave = heaves[i]
                    heading = headings[i]
                    depth = depths[i]

                    nothering = notherings[i]
                    eastering = easterings[i]

                    nothering_starboard = nothering_starboards[i]
                    eastering_starboard = eastering_starboards[i]

                    nothering_port = nothering_ports[i]
                    eastering_port = eastering_ports[i]

                    flat_seafloor_indicator = flat_seafloor_indicators[i]
                    rough_seafloor_indicator = rough_seafloor_indicators[i]

                    timestamp = timestamps[i]

                    sonarPings.append(SonarPing(sample, Point(eastering, nothering, heading), Point(eastering_starboard, nothering_starboard, heading), Point(eastering_port, nothering_port, heading), roll, pitch, yaw, heave, heading, timestamp, depth, flat_seafloor_indicator, rough_seafloor_indicator, filename=filename))
            except:
                print("nothing")

        return sonarPings
    
    def extract_imagette(self, sonarPings : List[SonarPing], target : Point = None, height : int = 100, width : int = 100, max_dist : float = 1.0, get_all : bool = True, tgv : bool = False, strict_single_file : bool = False, only_centered : bool = False) -> List[Imagette]:
        """
        Extract a 2D image snippet (imagette) of a given size centered on a specific geographic target.
        """
        imagettes = []
        if target is None:
            return imagettes

        detected_idxs = [i for i, ping in enumerate(sonarPings) if target.is_between(ping.point_port, ping.point_starboard, max_dist)]

        if not detected_idxs:
            return imagettes

        n = len(sonarPings)
        half_height = height // 2
        half_width = width // 2
        last_extracted_end = -1

        for idx in detected_idxs:
            if idx <= last_extracted_end:
                continue

            reference_ping = sonarPings[idx]
            target_size = len(reference_ping.sample)
            ref_filename = reference_ping.filename # Fichier cible de référence

            # 1. Détermination de la fenêtre initiale
            start_y = max(0, idx - half_height)
            end_y = start_y + height
            if end_y > n:
                end_y = n
                start_y = max(0, end_y - height)

            if start_y <= last_extracted_end:
                continue

            # --- OPTION 1 : strict_single_file ---
            # On vérifie si TOUS les pings de la fenêtre appartiennent au même fichier
            if strict_single_file:
                has_different_file = any(p.filename != ref_filename for p in sonarPings[start_y:end_y])
                if has_different_file:
                    print(f"Extraction annulée pour le ping {idx} : la fenêtre déborde sur un autre fichier (strict_single_file=True)")
                    continue

            # 2. Vérification de l'homogénéité des tailles dans la fenêtre
            has_size_mismatch = any(len(p.sample) != target_size for p in sonarPings[start_y:end_y])

            if has_size_mismatch:
                if not get_all:
                    print(f"Extraction annulée pour le ping {idx} : conflit de taille détecté (get_all=False)")
                    continue
                
                print(f"Conflit de taille au ping {idx}. Tentative de décalage vers le haut...")
                found_valid_window = False
                min_possible_y = max(0, last_extracted_end + 1)
                
                for test_start in range(start_y, min_possible_y - 1, -1):
                    test_end = test_start + height
                    # Si strict_single_file est True, on valide le décalage uniquement s'il reste dans le même fichier
                    if strict_single_file and any(p.filename != ref_filename for p in sonarPings[test_start:test_end]):
                        continue
                    if test_end <= n and all(len(p.sample) == target_size for p in sonarPings[test_start:test_end]):
                        start_y = test_start
                        end_y = test_end
                        found_valid_window = True
                        print(f"Nouvelle fenêtre homogène trouvée plus haut : [{start_y}:{end_y}]")
                        break
                
                if not found_valid_window:
                    print("Impossible de récupérer au-dessus. Tentative de décalage vers le bas...")
                    for test_start in range(start_y + 1, n - height + 1):
                        test_end = test_start + height
                        # Si strict_single_file est True, on valide le décalage uniquement s'il reste dans le même fichier
                        if strict_single_file and any(p.filename != ref_filename for p in sonarPings[test_start:test_end]):
                            continue
                        if test_start <= idx < test_end and all(len(p.sample) == target_size for p in sonarPings[test_start:test_end]):
                            start_y = test_start
                            end_y = test_end
                            found_valid_window = True
                            print(f"Nouvelle fenêtre homogène trouvée plus bas : [{start_y}:{end_y}]")
                            break
                
                if not found_valid_window:
                    print(f"Échec de la récupération pour le ping {idx} : pas de bloc homogène de taille {height}")
                    continue
            
            target_x, start_x, end_x = self._calculate_x_bounds(reference_ping, target, width, target_size)

            # Calcul du centrage
            centered_y = (idx - start_y) == half_height
            centered_x = (target_x - start_x) == half_width
            imagette_centered = centered_y and centered_x

            # --- OPTION 2 : only_centered ---
            # Si l'utilisateur ne veut que du parfaitement centré et que ça ne l'est pas, on jette
            if only_centered and not imagette_centered:
                print(f"Extraction annulée pour le ping {idx} : l'imagette n'est pas parfaitement centrée (only_centered=True)")
                continue

            (eastering, northering) = sonarPings[idx].get_position_from_index(target_x)
            detection_range = abs(target_x-sonarPings[idx].middle)/len(reference_ping.get_starboard())
            dist = math.sqrt((target.eastern-eastering)**2+(target.northern-northering)**2)

            pings_same_file = [p for p in sonarPings if p.filename == ref_filename]
            full_file_waterfall = np.array([p.sample for p in pings_same_file])

            if tragedy := tgv:  # Correction typo de la variable (tgv au lieu de tgv/tgv)
                waterfall_for_stats = np.array([p.sample_tvg for p in pings_same_file])
            else:
                waterfall_for_stats = full_file_waterfall

            min_val = np.min(waterfall_for_stats)
            max_val = np.percentile(waterfall_for_stats, 99)
            val_range = max_val - min_val if max_val > min_val else 1.0

            global_file_start_idx = next(i for i, p in enumerate(sonarPings) if p.filename == ref_filename)
            relative_start_y = start_y - global_file_start_idx

            side = "port"
            if(target_x > sonarPings[idx].middle):
                side = "starboard"

            sliced_pings = []
            for ping in sonarPings[start_y:end_y]:

                if(tgv):
                    raw_sample = np.array(ping.sample_tvg[start_x:end_x])
                else:
                    raw_sample = np.array(ping.sample[start_x:end_x])
                
                clipped = np.clip(raw_sample, min_val, max_val)
                new_sample = (clipped - min_val) / val_range
                
                sliced_pings.append(SonarPing(
                    sample=new_sample, point_ship=ping.point_ship, point_starboard=ping.point_starboard,
                    point_port=ping.point_port, roll=ping.roll, pitch=ping.pitch, yaw=ping.yaw, heave=ping.heave,
                    heading=ping.heading, timestamp=ping.timestamp, depth=ping.depth,
                    flat_seafloor_indicator=ping.flat_seafloor_indicator, rough_seafloor_indicator=ping.rough_seafloor_indicator,
                    x_bathy=ping.x_bathy, z_bathy=ping.z_bathy, filename=ping.filename, detection_range=detection_range
                ))
            
            imagettes.append(Imagette(sliced_pings, full_file_waterfall=full_file_waterfall, start_y=relative_start_y, start_x=start_x, centered=imagette_centered, side=side))
            last_extracted_end = end_y - 1

        return imagettes

    def _calculate_x_bounds(self, ping: SonarPing, target: Point, width: int, num_samples: int) -> Tuple[int, int]:
        """Méthode outil isolée pour calculer proprement start_x et end_x."""
        half_samples = num_samples // 2
        half_width = width // 2

        dist_ship_to_target_ground = math.sqrt(
            (target.eastern - ping.point_ship.eastern)**2 + 
            (target.northern - ping.point_ship.northern)**2
        )
        
        r_slant_max = ping.sound_speed * ping.delta_time * half_samples / 2
        slant_range_target = math.sqrt(dist_ship_to_target_ground**2 + ping.depth**2)
        
        if slant_range_target > r_slant_max:
            slant_range_target = r_slant_max

        ratio_from_ship = slant_range_target / r_slant_max if r_slant_max > 0 else 0.0
        pixel_offset_from_center = int(round(ratio_from_ship * half_samples))
        
        dist_to_port = math.sqrt((target.eastern - ping.point_port.eastern)**2 + (target.northern - ping.point_port.northern)**2)
        dist_to_stbd = math.sqrt((target.eastern - ping.point_starboard.eastern)**2 + (target.northern - ping.point_starboard.northern)**2)
        
        if dist_to_port < dist_to_stbd:
            target_x_idx = half_samples - pixel_offset_from_center
        else:
            target_x_idx = half_samples + pixel_offset_from_center

        start_x = target_x_idx - half_width

        end_x = start_x + width

        if start_x < 0:
            start_x = 0
            end_x = width
        elif end_x > num_samples:
            end_x = num_samples
            start_x = max(0, end_x - width)
            
        return target_x_idx, start_x, end_x