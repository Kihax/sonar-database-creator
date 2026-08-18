import numpy as np
from typing import List
from .Imagette import Imagette
import netCDF4 as nc4

class DatabaseCreatorImagette:

    def __init__(self, path: str):
        """
        Ouvre le fichier NetCDF en écriture.
        """
        self.datasetNetCDF = nc4.Dataset(path, "w", format="NETCDF4", encoding='latin-1')
        # 💡 SÉCURITÉ 1 : Désactive le parsing automatique des échelles de dimensions
        # qui fait planter HDF5 lors des synchro sur de grands jeux de données.
        self.datasetNetCDF.set_auto_chartostring(False)

    # ... (garder _get_or_create_variable, _get_or_create_dimension, init_global_attributes identiques) ...

    def write_cell_data(self, x: float, y: float, list_imagettes: List[Imagette], id: int = 0):
        ds = self.datasetNetCDF
        
        #target_w = ds.dimensions["width"].size
        #target_h = ds.dimensions["height"].size

        group_area_name = f"area_{id}"
        imagettes_x_y = ds.groups[group_area_name] if group_area_name in ds.groups else ds.createGroup(group_area_name)

        valid_count = 0
        for idx_img, imagette in enumerate(list_imagettes):
            img_np = np.array(imagette.imagette)
            h, w = img_np.shape

            #if h != target_h or w != target_w:
            #    continue

            pings = imagette.pings
            #if len(pings) != target_h:
            #    continue

            try:

                sub_group_name = f"imagette_{valid_count}"
                sub_img_group = imagettes_x_y.groups[sub_group_name] if sub_group_name in imagettes_x_y.groups else imagettes_x_y.createGroup(sub_group_name)

                self._get_or_create_dimension(sub_img_group, "height", h)
                self._get_or_create_dimension(sub_img_group, "width", w)
                self._get_or_create_dimension(sub_img_group, "size_point", 2)


                survey_files = set()
                for ping in pings:
                    if ping.filename is not None:
                        filename_str = str(ping.filename).strip()
                        if filename_str and filename_str.lower() != "nan":
                            survey_files.add(filename_str)

                self._get_or_create_dimension(sub_img_group, "height", h)
                self._get_or_create_dimension(sub_img_group, "width", w)
                self._get_or_create_dimension(sub_img_group, "size_point", 2)

                sub_img_group.setncattr("survey_files", list(survey_files))
                sub_img_group.setncattr("detection_range", pings[0].detection_range)
                sub_img_group.setncattr("frequency", pings[0].freq)
                sub_img_group.setncattr("centered", "true" if imagette.centered else "false")
                sub_img_group.setncattr("side", imagette.side)

                var_image = self._get_or_create_variable(sub_img_group, "image", "f4", ("height", "width"))
                var_pitch = self._get_or_create_variable(sub_img_group, "pitch", "f4", ("height",))
                var_roll = self._get_or_create_variable(sub_img_group, "roll", "f4", ("height",))
                var_yaw = self._get_or_create_variable(sub_img_group, "yaw", "f4", ("height",))
                var_timestamp = self._get_or_create_variable(sub_img_group, "timestamp", "i8", ("height",))
                var_heading = self._get_or_create_variable(sub_img_group, "heading", "f4", ("height",))
                var_depth = self._get_or_create_variable(sub_img_group, "depth", "f4", ("height",))
                var_heave = self._get_or_create_variable(sub_img_group, "heave", "f4", ("height",))
                
                var_point_eastern = self._get_or_create_variable(sub_img_group, "point_eastern", "f8", ("size_point",))
                var_point_western = self._get_or_create_variable(sub_img_group, "point_western", "f8", ("size_point",))
                var_point_southern = self._get_or_create_variable(sub_img_group, "point_southern", "f8", ("size_point",))
                var_point_nothern = self._get_or_create_variable(sub_img_group, "point_nothern", "f8", ("size_point",))

                var_ship_position = self._get_or_create_variable(sub_img_group, "ship_position", "f8", ("height", "size_point"))
                var_delta_time = self._get_or_create_variable(sub_img_group, "delta_time", "f4", ("height",))
                var_sound_speed = self._get_or_create_variable(sub_img_group, "sound_speed", "f4", ("height",))

                var_image[:, :] = img_np
                var_pitch[:] = [p.pitch for p in pings]
                var_roll[:] = [p.roll for p in pings]
                var_yaw[:] = [p.yaw for p in pings]
                var_timestamp[:] = [p.timestamp for p in pings]
                var_heading[:] = [p.heading for p in pings]
                var_depth[:] = [p.depth for p in pings]
                var_heave[:] = [p.heave for p in pings]
                var_delta_time[:] = [p.delta_time for p in pings]
                var_sound_speed[:] = [p.sound_speed for p in pings]
                
                ship_pos = [[p.point_ship.eastern, p.point_ship.northern] for p in pings]
                var_ship_position[:, :] = np.array(ship_pos, dtype=np.float64)

                var_point_eastern[:] = [imagette.point_most_eastern.eastern, imagette.point_most_eastern.northern]
                var_point_western[:] = [imagette.point_most_western.eastern, imagette.point_most_western.northern]
                var_point_southern[:] = [imagette.point_most_southern.eastern, imagette.point_most_southern.northern]
                var_point_nothern[:] = [imagette.point_most_northern.eastern, imagette.point_most_northern.northern]
                
                valid_count += 1

            except Exception as e:
                print(f"Erreur lors de l'écriture de l'imagette {idx_img} dans le groupe {group_area_name}: {e}")
                continue
        

        imagettes_x_y.setncattr("easting", x)
        imagettes_x_y.setncattr("northing", y)
        imagettes_x_y.setncattr("nb_imagette", valid_count)

    def _get_or_create_variable(self, group: nc4.Group, name: str, datatype: str, dimensions: tuple) -> nc4.Variable:
        """
        Récupère la variable si elle existe déjà, sinon la crée.
        """
        if name in group.variables:
            return group.variables[name]
        return group.createVariable(name, datatype, dimensions)

    def _get_or_create_dimension(self, group: nc4.Group, name: str, size: int):
        """
        S'assure qu'une dimension n'est créée qu'une seule fois.
        """
        if name not in group.dimensions:
            group.createDimension(name, size)

    def init_global_attributes(self, width : float, height: float, option: dict = None):
        """
        Initialise les dimensions et attributs globaux à la RACINE du fichier NetCDF.
        """
        if option is None:
            option = {}
            
        ds = self.datasetNetCDF

        ds.setncattr("width", width)
        ds.setncattr("height", height)
        ds.setncattr("sonar", "EdgeTech 6205")
        ds.setncattr("TVG", "true" if option.get("TVG", False) else "false")
        ds.setncattr("filter", option.get("filter", "false"))

    def init_global_attributes_from_database(self, sample_imagette, nb_groups: int, option: dict = None):
        """
        Alternative pour initialiser le fichier à partir d'un objet ImagetteDatabase.
        """
        if option is None:
            option = {}
            
        ds = self.datasetNetCDF
        
        img_np = sample_imagette.image.values if hasattr(sample_imagette.image, "values") else np.array(sample_imagette.image)
        h, w = img_np.shape
        
        ds.setncattr("width", w)
        ds.setncattr("height", h)
        ds.setncattr("sonar", "EdgeTech 6205")

        print(f"Dimensions globales initialisées : width={w}, height={h}")
        
        ds.setncattr("nb_group", nb_groups)
        ds.setncattr("TVG", "true" if option.get("TVG", False) else "false")
        ds.setncattr("filter", option.get("filter", "false"))

    def close(self):
        """
        Ferme proprement le fichier en fin de traitement.
        """
        self.datasetNetCDF.close()