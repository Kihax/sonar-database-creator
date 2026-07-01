import numpy as np
from typing import List, Dict, Tuple
from .Imagette import Imagette
import netCDF4 as nc4

class DatabaseCreatorImagette:

    def __init__(self, path: str):
        """
        On n'attend plus le dictionnaire complet ici. 
        On ouvre simplement le fichier pour y écrire au fur et à mesure.
        """
        self.datasetNetCDF = nc4.Dataset(path, "w", format="NETCDF4", encoding='latin-1')

    def init_global_attributes(self, sample_imagette: Imagette, nb_groups: int, option: dict = None):
        """
        Initialise les dimensions et attributs globaux du fichier NetCDF 
        dès qu'on trouve la première imagette valide.
        """
        if option is None:
            option = {}
            
        ds = self.datasetNetCDF
        
        # Définition des dimensions globales basées sur la structure d'une imagette
        ds.setncattr("width", len(sample_imagette.imagette[0]))
        ds.setncattr("height", len(sample_imagette.imagette))
        ds.setncattr("sonar", "EdgeTech 6205")

        print(f"width : {len(sample_imagette.imagette[0])}")
        print(f"height : {len(sample_imagette.imagette)}" )
        
        ds.createDimension("width", len(sample_imagette.imagette[0]))
        ds.createDimension("height", len(sample_imagette.imagette))
        ds.createDimension("size_point", 2)
        
        ds.setncattr("nb_group", nb_groups)
        ds.setncattr("TVG", "true" if option.get("TVG", False) else "false")
        ds.setncattr("filter", option.get("filter", "false"))

    def write_cell_data(self, x: float, y: float, list_imagettes: List[Imagette], id : int = 0):
        """
        Écrit les imagettes d'UNE SEULE coordonnée (x, y) puis libère la mémoire.
        """
        ds = self.datasetNetCDF
        
        # Création du groupe pour la cellule (x, y) courante
        imagettes_x_y = ds.createGroup(f"area_{id}")
        imagettes_x_y.setncattr("easting", x)
        imagettes_x_y.setncattr("northing", y)
        imagettes_x_y.setncattr("nb_imagette", len(list_imagettes))

        # Écriture séquentielle de chaque imagette de la cellule
        for i, imagette in enumerate(list_imagettes):
            
            img_np = np.array(imagette.imagette)
            h, w = img_np.shape

            # Sécurité si une imagette est vide
            if h == 0 or w == 0:
                continue

            sub_img_group = imagettes_x_y.createGroup(f"imagette_{i}")
            
            # --- CORRECTION ICI : On crée des dimensions propres au sous-groupe ---
            sub_img_group.createDimension("local_height", h)
            sub_img_group.createDimension("local_width", w)

            survey_files = set()
            for ping in imagette.pings:
                if ping.filename is not None:
                    filename_str = ping.filename
                    
                    # --- NOUVELLE LOGIQUE DE NETTOYAGE BLINDÉE ---
                    # 1. Gestion des objets NumPy (Tableaux ou Scalaires comme np.str_)
                    if isinstance(filename_str, (np.ndarray, np.generic)):
                        if np.ndim(filename_str) > 0:  # Si c'est un vrai tableau
                            if filename_str.size == 0:
                                continue
                            filename_str = filename_str.flat[0]
                        else:  # Si c'est un scalaire (votre cas ici)
                            filename_str = filename_str.item()
                            
                    # 2. Gestion des listes Python standards
                    elif isinstance(filename_str, list):
                        if len(filename_str) == 0:
                            continue
                        filename_str = filename_str[0]
                    
                    # 3. Conversion finale propre en chaîne Python standard
                    filename_str = str(filename_str).strip()

                    # Ajout sécurisé dans les sets (on ignore les chaînes vides ou les "nan")
                    if filename_str and filename_str.lower() != "nan":
                        survey_files.add(filename_str)

            # Attributs de l'imagette
            sub_img_group.setncattr("survey_files", list(survey_files))
            sub_img_group.setncattr("detection_range", imagette.pings[0].detection_range)
            sub_img_group.setncattr("frequency", imagette.pings[0].freq)
            sub_img_group.setncattr("centered", "true" if imagette.centered else "false")
            sub_img_group.setncattr("side", imagette.side)
            

            # On utilise les dimensions locales ("local_height", "local_width")
            var_image = sub_img_group.createVariable("image", "f4", ("local_height", "local_width"))
            var_pitch = sub_img_group.createVariable("pitch", "f4", ("local_height",))
            var_roll = sub_img_group.createVariable("roll", "f4", ("local_height",))
            var_yaw = sub_img_group.createVariable("yaw", "f4", ("local_height",))
            var_timestamp = sub_img_group.createVariable("timestamp", "i8", ("local_height",))
            var_heading = sub_img_group.createVariable("heading", "f4", ("local_height",))
            var_depth = sub_img_group.createVariable("depth", "f4", ("local_height",))
            var_heave = sub_img_group.createVariable("heave", "f4", ("local_height",))
            
            # "size_point" reste global (il vaut toujours 2)
            var_point_eastern = sub_img_group.createVariable("point_eastern", "f8", ("size_point",))
            var_point_western = sub_img_group.createVariable("point_western", "f8", ("size_point",))
            var_point_southern = sub_img_group.createVariable("point_southern", "f8", ("size_point",))
            var_point_nothern = sub_img_group.createVariable("point_nothern", "f8", ("size_point",))

            var_ship_position = sub_img_group.createVariable("ship_position", "f8", ("local_height", "size_point",))

            var_delta_time = sub_img_group.createVariable("delta_time", "f4", ("local_height",))
            var_sound_speed = sub_img_group.createVariable("sound_speed", "f4", ("local_height",))

            # Cette fois-ci, l'écriture directe passera sans broncher
            var_image[:, :] = img_np
            
            # Utilisation de list comprehensions évaluées à la volée
            var_pitch[:] = [p.pitch for p in imagette.pings]
            var_roll[:] = [p.roll for p in imagette.pings]
            var_yaw[:] = [p.yaw for p in imagette.pings]
            var_timestamp[:] = [p.timestamp for p in imagette.pings]
            var_heading[:] = [p.heading for p in imagette.pings]
            var_depth[:] = [p.depth for p in imagette.pings]
            var_heave[:] = [p.heave for p in imagette.pings]
            var_delta_time[:] = [p.delta_time for p in imagette.pings]
            var_sound_speed[:] = [p.sound_speed for p in imagette.pings]
            var_ship_position[:] = [ [p.point_ship.eastern, p.point_ship.northern] for p in imagette.pings ]

            var_point_eastern[:] = [imagette.point_most_eastern.eastern, imagette.point_most_eastern.northern]
            var_point_western[:] = [imagette.point_most_western.eastern, imagette.point_most_western.northern]
            var_point_southern[:] = [imagette.point_most_southern.eastern, imagette.point_most_southern.northern]
            var_point_nothern[:] = [imagette.point_most_northern.eastern, imagette.point_most_northern.northern]

        ds.sync()

    def close(self):
        """Ferme proprement le fichier en fin de script."""
        self.datasetNetCDF.close()