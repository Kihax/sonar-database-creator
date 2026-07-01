def search_coord(dt, easting_searched, northing_searched):
    """
    Recherche toutes les imagettes dont les bornes géographiques contiennent 
    les coordonnées (easting_searched, northing_searched).
    Retourne une liste de dictionnaires prête à être visualisée.
    """
    imagette_zone = []
    
    for i in range(len(dt["image"])):
        easting_min = dt["easting_min"].values[i]
        easting_max = dt["easting_max"].values[i]
        northing_min = dt["northing_min"].values[i]
        northing_max = dt["northing_max"].values[i]

        #print(f"easting min : {easting_min} and easting max : {easting_max} northing min : {northing_min} and max : {northing_max}")

        # Vérification si la coordonnée cherchée est dans les bornes de l'imagette
        if (easting_min <= easting_searched <= easting_max and 
            northing_min <= northing_searched <= northing_max):
            print(i)
            # On construit le dictionnaire complet pour cette imagette d'intérêt
            imagette_data = {
                "image": dt["image"].values[i],
                "roll": dt["roll"].values[i],
                "pitch": dt["pitch"].values[i],
                "yaw": dt["yaw"].values[i],
                "heave": dt["heave"].values[i],
                "heading": dt["heading"].values[i],
                "easting": dt["easting"].values[i],
                "northing": dt["northing"].values[i],
                "easting_max": easting_max,
                "northing_max": northing_max,
                "easting_min": easting_min,
                "northing_min": northing_min,
                "timestamp": dt["timestamp"].values[i],
            }
            imagette_zone.append(imagette_data)
            
    return imagette_zone