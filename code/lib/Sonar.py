from .DataManagement import DataManagement
from .timetag_to_timestamp import timetag_to_timestamp
from .Point import Point
from .SonarPing import SonarPing
from .Imagette import Imagette
from typing import List

import numpy as np
import math

class Sonar:
    """
        Class to extract data from the sonar
        
        Args :
            - dts (List<DataTree>) : List of netCDF files
    """
    def __init__(self, dts, data : DataManagement = None):
        self.dts=dts
        self.storage_HF : List[SonarPing] = []
        self.storage_LF : List[SonarPing] = []
        if(data):
            self.data = data
        else:
            self.data = DataManagement(dts)

    def compute_bathymetry(self, ping_timetag : int, angle_bathymetry_port : List[float], angle_bathymetry_starboard : List[float], time_bathymetry_port : List[float], time_bathymetry_starboard : List[float], quality_bathymetry_starboard : List[int], quality_bathymetry_port : List[int], sound_speed : float = 1475.0):
        """
        Compute corrected bathymetry profile for a single ping.

        Args:
            ping_timetag: absolute timestamp of the ping in nanoseconds
            angle_bathymetry_port: port beam angles for the ping
            angle_bathymetry_starboard: starboard beam angles for the ping
            time_bathymetry_port: port travel times for the ping
            time_bathymetry_starboard: starboard travel times for the ping
            quality_bathymetry_starboard: starboard quality flags for the ping
            quality_bathymetry_port: port quality flags for the ping
            data: optional DataManagement instance to avoid recreation

        Returns:
            x_port, z_port, x_stbd, z_stbd
        """

        data = self.data

        time_bathymetry_port = np.array(time_bathymetry_port)
        time_bathymetry_starboard = np.array(time_bathymetry_starboard)
        
        bathy_port_0_timetag = ping_timetag + time_bathymetry_port * 1e9
        bathy_stbd_0_timetag = ping_timetag + time_bathymetry_starboard * 1e9


        # Vectorized method calls - much faster than list comprehensions
        roll_smp_port = data.roll.get_values_from_timestamps(bathy_port_0_timetag)
        roll_smp_stbd = data.roll.get_values_from_timestamps(bathy_stbd_0_timetag)

        bathy_port_0_angle = np.array(angle_bathymetry_port)
        bathy_port_0_angle_roll = bathy_port_0_angle - roll_smp_port - 0.18

        bathy_stbd_0_angle = np.array(angle_bathymetry_starboard)
        bathy_stbd_0_angle_roll = bathy_stbd_0_angle - roll_smp_stbd + 0.28

        time_port = time_bathymetry_port
        time_stbd = time_bathymetry_starboard

        x_port_roll = sound_speed * time_port / 2.0 * np.sin(bathy_port_0_angle_roll * np.pi / 180.0)
        z_port_roll = -sound_speed * time_port / 2.0 * np.cos(bathy_port_0_angle_roll * np.pi / 180.0)

        x_stbd_roll = sound_speed * time_stbd / 2.0 * np.sin(bathy_stbd_0_angle_roll * np.pi / 180.0)
        z_stbd_roll = -sound_speed * time_stbd / 2.0 * np.cos(bathy_stbd_0_angle_roll * np.pi / 180.0)

        # Vectorized method calls instead of loops
        heave_smp_port = data.heave.get_values_from_timestamps(bathy_port_0_timetag)
        heave_smp_stbd = data.heave.get_values_from_timestamps(bathy_stbd_0_timetag)

        bad_samples_port = np.array(quality_bathymetry_port) == 0
        x_port = x_port_roll[~bad_samples_port]
        z_port = (z_port_roll - heave_smp_port)[~bad_samples_port]

        bad_samples_stbd = np.array(quality_bathymetry_starboard) == 0
        x_stbd = x_stbd_roll[~bad_samples_stbd]
        z_stbd = (z_stbd_roll - heave_smp_stbd)[~bad_samples_stbd]

        return x_port, z_port, x_stbd, z_stbd
    
    def is_flat(self, x: np.ndarray, z: np.ndarray) -> float:
        """
        Calculate the absolute slope of a bathymetric profile (x, z).
        
        A slope of 0.0 means perfectly horizontal. The higher the value,
        the steeper the seabed.

        Parameters
        ----------
        x : array_like
            Horizontal distances (e.g., along-track distance).
        z : array_like
            Depths or elevations.

        Returns
        -------
        float
            The absolute slope coefficient of the linear regression.
            Returns 0.0 if there are fewer than 2 points.
        """
        if len(x) < 2:
            return 0.0
            
        # Perform a 1st-degree polynomial fit (linear regression)
        slope, _ = np.polyfit(x, z, 1)
        
        return abs(slope)
    
    def calculate_roughness(self, x: np.ndarray, z: np.ndarray) -> float:
        """
        Calculate the seafloor roughness indicator of a bathymetric profile.
        
        This indicator is computed as the standard deviation of local slopes
        between consecutive points. A value close to 0.0 indicates a perfectly 
        smooth or flat surface.

        Parameters
        ----------
        x : array_like
            Horizontal distances.
        z : array_like
            Depths or elevations.

        Returns
        -------
        float
            The standard deviation of the local slopes.
            Returns 0.0 if there are not enough valid slope segments.
        """
        delta_z = np.diff(z)
        delta_x = np.diff(x)

        # Create a mask to avoid division by zero where x does not change
        valid_mask = delta_x != 0

        # Avoid empty array edge cases
        if not np.any(valid_mask):
            return 0.0

        # Compute local slopes for valid segments
        local_slopes = delta_z[valid_mask] / delta_x[valid_mask]

        # The roughness indicator is the standard deviation of these slopes
        roughness_indicator = np.std(local_slopes)

        return roughness_indicator


    def extract_lines(self, f: str = "LF", storage: List[SonarPing] = None) -> None:
        """Extract sonar lines from the DataTree and compute bathymetric indicators.

        Parameters
        ----------
        f : str, optional
            Frequency type to extract, either "LF" (Low Frequency) or "HF" (High
            Frequency). Default is "LF".
        storage : List[SonarPing], optional
            A mutable list where the extracted and processed SonarPing objects will
            be appended. If None, an empty list is initialized.
        """
        if storage is None:
            storage = []

        data = self.data

        for dt in self.dts:
            filename = dt.attrs.get("Survey file")
            # Skip specific faulty or template files
            if filename == "2024__1001410_0001.001_Binned":
                continue
            print(f"Processing file: {filename}")

            # Base paths for sonar blocks
            port_path = f"/Sonar/SLS {f}/Port/Block_0"
            stbd_path = f"/Sonar/SLS {f}/Starboard/Block_0"
            bathy_port_path = "/Sonar/Bathymetry/Port/Block_0"
            bathy_stbd_path = "/Sonar/Bathymetry/Starboard/Block_0"

            # Extract parameters
            sonar_range = dt[port_path].attrs["range (m)"]
            delta_time = dt[port_path].attrs["sample_duration (s)"]

            # Pre-load data matrices into memory to drastically speed up the loop
            samples_port = dt[port_path].samples.values
            samples_starboard = dt[stbd_path].samples.values
            limit_sample = len(samples_port[0])

            timestamps = timetag_to_timestamp(dt[port_path].timetag.values)

            # Pre-load bathymetry data to avoid repeating heavy disk I/O in the loop
            angles_b_port = dt[f"{bathy_port_path}/angle"].load().values
            angles_b_stbd = dt[f"{bathy_stbd_path}/angle"].load().values
            times_b_port = dt[f"{bathy_port_path}/time"].load().values
            times_b_stbd = dt[f"{bathy_stbd_path}/time"].load().values
            qualities_b_port = dt[f"{bathy_port_path}/quality"].load().values
            qualities_b_stbd = dt[f"{bathy_stbd_path}/quality"].load().values
            bathy_timetags = dt[f"{bathy_port_path}/timetag"].load().values

            # Determine safe bounds for iteration
            ping_limit = min(
                len(timestamps), angles_b_port.shape[0], angles_b_stbd.shape[0]
            )

            for i in range(ping_limit):
                timestamp = timestamps[i]
                sample_port = samples_port[i][0:limit_sample]
                sample_starboard = samples_starboard[i][0:limit_sample]

                if len(sample_port) != limit_sample:
                    continue

                # Compute bathymetry using pre-loaded values
                ping_timetag = timetag_to_timestamp(bathy_timetags[i])

                x_port, z_port, x_stbd, z_stbd = self.compute_bathymetry(
                    ping_timetag,
                    angles_b_port[i],
                    angles_b_stbd[i],
                    times_b_port[i],
                    times_b_stbd[i],
                    qualities_b_stbd[i],
                    qualities_b_port[i],
                )

                # Average depth between last port point and first starboard point
                depth = (z_port[-1] + z_stbd[0]) / 2

                # Concatenate and sort bathymetry profile
                x_bathymetry = np.concatenate([x_port, x_stbd])
                z_bathymetry = np.concatenate([z_port, z_stbd])

                sorted_indices = np.argsort(x_bathymetry)
                x_bathymetry = x_bathymetry[sorted_indices]
                z_bathymetry = z_bathymetry[sorted_indices]

                # Seabed topography indicators (using the newly updated methods)
                flat_seafloor_indicator = self.is_flat(x_bathymetry, z_bathymetry)
                rough_seafloor_indicator = self.calculate_roughness(
                    x_bathymetry, z_bathymetry
                )

                # Reconstruct the complete ping signal (Port is flipped to maintain geometry)
                full_ping_sample = np.zeros(2 * limit_sample)
                full_ping_sample[0:limit_sample] = np.flip(sample_port)
                full_ping_sample[limit_sample : 2 * limit_sample] = sample_starboard

                # Retrieve navigation data
                eastern = data.easting.get_value_from_timestamp(timestamp)
                northern = data.northing.get_value_from_timestamp(timestamp)
                heading = data.heading.get_value_from_timestamp(
                    timestamp, with_quality=False
                )

                if heading is None:
                    continue

                roll = data.roll.get_value_from_timestamp(timestamp)
                pitch = data.pitch.get_value_from_timestamp(timestamp)
                yaw = data.yaw.get_value_from_timestamp(timestamp)
                heave = data.heave.get_value_from_timestamp(timestamp)

                # Security check for missing telemetry data
                if None in (eastern, northern, roll, pitch, heading, heave):
                    continue

                # Convert navigation angles to radians
                r_rad = np.radians(roll)
                p_rad = np.radians(pitch)
                y_rad = np.radians(heading)

                # Construct Euler rotation matrices
                R_roll = np.array([
                    [np.cos(r_rad), 0, np.sin(r_rad)],
                    [0, 1, 0],
                    [-np.sin(r_rad), 0, np.cos(r_rad)],
                ])

                R_pitch = np.array([
                    [1, 0, 0],
                    [0, np.cos(p_rad), -np.sin(p_rad)],
                    [0, np.sin(p_rad), np.cos(p_rad)],
                ])

                # Heading 0° is North (clockwise).
                R_yaw = np.array([
                    [np.sin(y_rad), np.cos(y_rad), 0],
                    [np.cos(y_rad), -np.sin(y_rad), 0],
                    [0, 0, 1],
                ])

                # Combined rotation matrix (Vessel to UTM)
                R = R_yaw @ R_pitch @ R_roll

                # 1. Bras de levier : Antenne GPS -> Point de Référence du navire (0,0,0)
                lever_arm_nav = np.array([0.0, 0.33, 2.836])
                lever_arm_nav_rotated = R @ lever_arm_nav
                
                vessel_east = eastern - lever_arm_nav_rotated[0]
                vessel_north = northern - lever_arm_nav_rotated[1]
                vessel_z = -heave  # Référence surface

                # 2. Bras de levier : Point de Référence -> Centres Acoustiques des Transducteurs
                # D'après le HVF, les têtes 1 (Port) et 2 (Stbd) ont le même offset physique par rapport à l'IMU
                lever_arm_transducer = np.array([0.0, 0.297, 0.338])
                lever_arm_trans_rotated = R @ lever_arm_transducer

                # Position absolue du centre acoustique du sonar dans le repère UTM
                sonar_east = vessel_east + lever_arm_trans_rotated[0]
                sonar_north = vessel_north + lever_arm_trans_rotated[1]
                sonar_z = vessel_z + lever_arm_trans_rotated[2]

                # 3. Instanciation du point du navire mis à jour à la position réelle du sonar
                ship = Point(sonar_east, sonar_north, heading)

                # Prise en compte du décalage vertical (Z) du sonar pour la profondeur réelle sous le capteur
                # (z_port et z_stbd étant déjà négatifs et calculés par rapport au transducteur)
                depth = ((z_port[-1] + z_stbd[0]) / 2) + sonar_z

                # Calculate maximum horizontal swath coverage range
                if sonar_range > abs(depth):
                    x_max = math.sqrt(sonar_range**2 - depth**2)
                else:
                    x_max = 0.0

                starboard_extretmity = ship.lateral_points_pos_sonar(x_max, "starboard")
                port_extremity = ship.lateral_points_pos_sonar(x_max, "port")

                # Append the structured data to the storage list
                storage.append(
                    SonarPing(
                        full_ping_sample,
                        ship,
                        starboard_extretmity,
                        port_extremity,
                        roll,
                        pitch,
                        yaw,
                        heave,
                        heading,
                        timestamp,
                        depth,
                        flat_seafloor_indicator,
                        rough_seafloor_indicator,
                        x_bathy=x_bathymetry,
                        z_bathy=z_bathymetry,
                        freq=f,
                        delta_time=delta_time,
                        filename=filename,
                    )
                )

        # Sort storage by timestamp to prevent asynchronous gaps in the outputs
        storage.sort(key=lambda ping: ping.timestamp)

    def extract_lines_HF(self):
        """
            Extract high frequency lines
        """
        self.extract_lines("HF", self.storage_HF)

    def extract_lines_LF(self):
        """
            Extract low frequency lines
        """
        self.extract_lines("LF", self.storage_LF)
    
    def get_objects(self, storage: List[SonarPing]):
        """
            GEMINI a revoir
        """
        MAX_ROTATION_SPEED = 0.5  
        objs_bruts = []

        # --- ÉTAPE 1 : Collecte et premier filtrage (Dynamique de rotation) ---
        for i, ping in enumerate(storage):
            # --- Tribord (Starboard) ---
            objs_starboard = ping.detect_object_starboard(
                window_size=300,        
                threshold_factor=3.5,   
                security=20
            )
            for obj in objs_starboard:
                eastering, northering = ping.get_position_from_index(obj)
                objs_bruts.append({
                    "column": obj,
                    "line": i,
                    "eastering": eastering,
                    "northering": northering
                })
            
            # --- Bâbord (Port) ---
            objs_port = ping.detect_objet_port(
                window_size=300,        
                threshold_factor=3.5,   
                security=20
            )
            for obj in objs_port:
                eastering, northering = ping.get_position_from_index(obj)
                objs_bruts.append({
                    "column": obj,
                    "line": i,
                    "eastering": eastering,
                    "northering": northering,
                    "case_id": 0,
                })

        # Si aucune détection n'a été faite, on s'arrête là
        if not objs_bruts:
            return []

        # --- ÉTAPE 2 : Filtrage Spatial (Raster/Heatmap) ---
        # Extraction des coordonnées pour la génération de la grille
        geo_x = [o["eastering"] for o in objs_bruts]
        geo_y = [o["northering"] for o in objs_bruts]
        
        cell_size = 1.0  
        x_min, x_max = min(geo_x), max(geo_x)
        y_min, y_max = min(geo_y), max(geo_y)
        
        x_edges = np.arange(x_min, x_max + cell_size, cell_size)
        y_edges = np.arange(y_min, y_max + cell_size, cell_size)
        
        # Génération de la matrice d'occupation
        heatmap, x_edges, y_edges = np.histogram2d(geo_x, geo_y, bins=[x_edges, y_edges])
        
        MIN_CONVERGENCE_THRESHOLD = 8
        filtered_cells = heatmap > MIN_CONVERGENCE_THRESHOLD

        x_indices, y_indices = np.where(filtered_cells)
        cell_to_id = { (x, y): i for i, (x, y) in enumerate(zip(x_indices, y_indices)) }
        
        objs_filtres = []
        for obj in objs_bruts:
            bin_x = np.digitize(obj["eastering"], x_edges) - 1
            bin_y = np.digitize(obj["northering"], y_edges) - 1
            
            bin_x = min(max(bin_x, 0), filtered_cells.shape[0] - 1)
            bin_y = min(max(bin_y, 0), filtered_cells.shape[1] - 1)
            
            # En utilisant .get(), on évite le KeyError si la case n'est pas dans le dictionnaire
            case_id = cell_to_id.get((bin_x, bin_y))
            
            # Si case_id n'est pas None, c'est que la case est valide et dense !
            if case_id is not None:
                obj["case_id"] = case_id  # (ou "object_id" selon le nom choisi)
                objs_filtres.append(obj)

        print(f"Filtrage Raster terminé : {len(objs_bruts)} points bruts réduits à {len(objs_filtres)} objets validés.")
        
        return objs_filtres
    
    def extract_imagette_from_object(self, objects, storage: List[SonarPing], height: int = 100, width: int = 100):
        """
            GEMINI a revoir
        """
        if not objects or not storage:
            return {}

        # 1. Tri par ligne pour appliquer la logique de non-chevauchement des pings
        objects_sorted = sorted(objects, key=lambda o: o["line"])

        n = len(storage)
        half_height = height // 2
        half_width = width // 2
        last_extracted_end = -1

        # Dictionnaire de sortie : { object_id: [Imagette1, Imagette2, ...] }
        groupes_objets = {}

        for obj in objects_sorted:
            idx = obj["line"]
            target_x_idx = obj["column"]
            obj_id = obj["case_id"] # Récupération de l'ID de la case raster

            # --- FILTRE DE CHEVAUCHEMENT STRICT ---
            if idx <= last_extracted_end:
                continue

            # --- AXE Y : Fenêtre Hauteur ---
            start_y = max(0, idx - half_height)
            end_y = start_y + height
            if end_y > n:
                end_y = n
                start_y = max(0, end_y - height)

            if start_y <= last_extracted_end:
                continue

            # --- AXE X : Fenêtre Largeur ---
            reference_ping = storage[idx]
            num_samples = len(reference_ping.sample)

            start_x = target_x_idx - half_width
            end_x = start_x + width

            if start_x < 0:
                start_x = 0
                end_x = width
            elif end_x > num_samples:
                end_x = num_samples
                start_x = max(0, end_x - width)

            # --- EXTRACTION ET SLICE ---
            sliced_pings = []
            for ping in storage[start_y:end_y]:
                sliced_pings.append(SonarPing(
                    sample=ping.sample[start_x:end_x],
                    point_ship=ping.point_ship,
                    point_starboard=ping.point_starboard,
                    point_port=ping.point_port,
                    roll=ping.roll,
                    pitch=ping.pitch,
                    yaw=ping.yaw,
                    heave=ping.heave,
                    heading=ping.heading,
                    timestamp=ping.timestamp,
                    depth=ping.depth,
                    flat_seafloor_indicator=ping.flat_seafloor_indicator,
                    rough_seafloor_indicator=ping.rough_seafloor_indicator,
                    x_bathy=ping.x_bathy,
                    z_bathy=ping.z_bathy
                ))

            imagette = Imagette(sliced_pings)

            # --- RANGEMENT PAR CASE RASTER (OBJECT ID) ---
            if obj_id not in groupes_objets:
                groupes_objets[obj_id] = []
            
            groupes_objets[obj_id].append(imagette)

            # Mise à jour de la borne Y
            last_extracted_end = end_y - 1

        return groupes_objets
            

