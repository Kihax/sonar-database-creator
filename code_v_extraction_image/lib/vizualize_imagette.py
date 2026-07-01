from matplotlib import pyplot as plt
import numpy as np
import math
from datetime import datetime
from lib.measure_depth import mesure_seafloor_backtracking
from lib.lateral_points_pos_sonar import lateral_points_pos_sonar

def vizualize_imagette(imagettes_list, initial_index=0, target_coord=None):
    """
    Visualise une liste d'imagettes sonar avec navigation par flèches (<- et ->).
    Calcule la position géométrique exacte (pixel X, Y) d'une coordonnée cible
    en reconstruisant la géométrie par ping (Pythagore + cap du sonar).
    """
    num_images = len(imagettes_list)
    if num_images == 0:
        print("La liste d'imagettes est vide.")
        return

    all_easting = np.hstack([np.asarray(data["easting"]) for data in imagettes_list])
    all_northing = np.hstack([np.asarray(data["northing"]) for data in imagettes_list])

    # 1. Création de la figure et des subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1], hspace=0.35, wspace=0.3)
    ax_img = fig.add_subplot(gs[0, :])
    ax_traj = fig.add_subplot(gs[1, 0])
    ax_plot = fig.add_subplot(gs[1, 1])

    # 2. Initialisation des objets graphiques
    im = ax_img.imshow(np.zeros((10, 10)), cmap='gray', vmin=0, vmax=10, aspect='auto')
    line_data, = ax_plot.plot([], [], color='cyan', linewidth=1)
    traj_full_line, = ax_traj.plot([], [], color='blue', linewidth=1, label='Trajectoire totale')
    traj_current_line, = ax_traj.plot([], [], color='red', linewidth=2, label='Imagette active')
    current_ping_point, = ax_traj.plot([], [], 'go', markersize=2, label='Ping actuel')
    # Étoile sur la position géographique cible (affichée sur la trajectoire)
    target_marker_traj, = ax_traj.plot([], [], marker='*', color='red', markersize=8, linestyle='None', label='Cible géographique')
    target_marker_traj.set_visible(False)
    
    ax_plot.set_xlabel("Index du pixel (Portée / Colonne)")
    ax_plot.set_ylabel("Intensité acoustique")
    ax_plot.grid(True, alpha=0.3)

    ax_traj.set_title("Trajectoire du navire")
    ax_traj.set_xlabel("Easting")
    ax_traj.set_ylabel("Northing")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_aspect('equal', adjustable='box')
    ax_traj.legend(loc='best', fontsize='small')

    info_text = ax_plot.text(0.05, 0.95, "Cliquez sur un graphique...", 
                             transform=ax_plot.transAxes, 
                             verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    vline_port = ax_plot.axvline(0, color='green', linestyle='--', alpha=0.7)
    vline_starboard = ax_plot.axvline(0, color='green', linestyle='--', alpha=0.7)
    vline_maker = ax_plot.axvline(0, color='blue', linestyle='--', alpha=0.7)
    horizontal_marker = ax_img.axhline(0, color='red', linestyle='--', alpha=0.7)

    # Étoile rouge sur la cible dans l'image, et ligne pointillée sur le profil
    target_marker_img, = ax_img.plot([], [], 'r*', markersize=12, label='Cible recherchée')
    current_sample_marker, = ax_img.plot([], [], marker='o', color='blue', markersize=4, markeredgewidth=0.5, linestyle='None', label='Curseur échantillon')
    target_marker_plot = ax_plot.axvline(0, color='red', linestyle=':', alpha=0.6, visible=False)

    # Constantes physiques du système
    sound_speed = 1470
    time_step = 1.152e-5

    fig.current_image_index = initial_index
    fig.selected_sample = None

    # 3. Fonction de chargement et de calcul d'une imagette
    def load_image_data(index):
        data = imagettes_list[index]
        
        fig.image = data["image"]
        fig.easting = data["easting"]
        fig.northing = data["northing"]
        fig.heading = data["heading"]
        fig.timestamp = data["timestamp"]

        height, width = fig.image.shape
        middle = int(width / 2)
        fig.middle = middle

        im.set_data(fig.image)
        im.set_extent([0, width, height, 0])
        ax_img.set_title(f"[{index + 1}/{num_images}] Sonar Waterfall | easting : {data['easting_min']} - {data['easting_max']}")

        traj_full_line.set_data(all_easting, all_northing)
        traj_current_line.set_data(fig.easting, fig.northing)
        ax_traj.set_title(f"Trajectoire du navire - imagette {index + 1}/{num_images}")
        x_margin = max(1.0, (np.nanmax(all_easting) - np.nanmin(all_easting)) * 0.02)
        y_margin = max(1.0, (np.nanmax(all_northing) - np.nanmin(all_northing)) * 0.02)
        ax_traj.set_xlim(np.nanmin(all_easting) - x_margin, np.nanmax(all_easting) + x_margin)
        ax_traj.set_ylim(np.nanmin(all_northing) - y_margin, np.nanmax(all_northing) + y_margin)

        # --- RECHERCHE GÉOMÉTRIQUE EXACTE VECTORISÉE (SANS BOUCLE FOR) ---
        if target_coord is not None:
            e_target, n_target = target_coord
            # Affiche la position géographique cible sur la trajectoire
            target_marker_traj.set_data([e_target], [n_target])
            target_marker_traj.set_visible(True)
            
            # On prépare des variables pour stocker le meilleur pixel global
            global_best_x = None
            global_best_y = None
            min_geo_distance = float('inf')

            # Vecteur X identique pour chaque ligne
            x_indices = np.arange(width)
            is_starboard = x_indices > middle
            dist_pixels = x_indices - middle
            dist_metres = sound_speed * time_step * np.abs(dist_pixels) / 2
            sides = np.where(is_starboard, "starboard", "port")
            
            # Vectorisation de votre fonction de positionnement
            v_pos_sonar = np.vectorize(lateral_points_pos_sonar)

            # On parcourt chaque ping Y (seulement ~100 itérations, c'est ultra rapide)
            for y_idx in range(height):
                ping_intensities_local = fig.image[y_idx, :]
                
                # Calcul de la bathymétrie locale pour CE ping
                d_port, _ = mesure_seafloor_backtracking(np.flip(ping_intensities_local[0:middle]), sound_speed)
                d_starboard, _ = mesure_seafloor_backtracking(ping_intensities_local[middle:], sound_speed)
                depths = np.where(is_starboard, d_starboard, d_port)
                
                # Pythagore (ground range)
                x_dist_sq = dist_metres**2 - depths**2
                x_dist = np.sqrt(np.maximum(0, x_dist_sq))
                
                # Navigation pour ce ping
                e_sonar = fig.easting[y_idx]
                n_sonar = fig.northing[y_idx]
                h_sonar = np.radians(fig.heading[y_idx])
                
                # Calcul des coordonnées (E, N) pour TOUS les pixels de cette ligne Y
                e_pixels, n_pixels = v_pos_sonar(e_sonar, n_sonar, h_sonar, x_dist, sides)
                
                # Calcul de la distance à la cible pour toute la ligne
                geo_distances = (e_pixels - e_target)**2 + (n_pixels - n_target)**2
                
                # On trouve le meilleur pixel sur cette ligne
                local_best_x = np.argmin(geo_distances)
                local_min_dist = geo_distances[local_best_x]
                
                # Si c'est meilleur que ce qu'on avait sur les lignes précédentes, on stocke
                if local_min_dist < min_geo_distance:
                    min_geo_distance = local_min_dist
                    global_best_x = int(local_best_x)
                    global_best_y = y_idx

            # Sauvegarde des index optimaux trouvés sur la matrice 2D
            if global_best_x is not None and global_best_y is not None:
                target_marker_img.set_data([global_best_x], [global_best_y])
                target_marker_img.set_visible(True)
                fig.target_x = global_best_x
                fig.target_y = global_best_y
            else:
                target_marker_img.set_visible(False)
                fig.target_y = None
        else:
            target_marker_img.set_visible(False)
            target_marker_traj.set_visible(False)
            fig.target_y = None

        # --- MISE À JOUR DU SUBPLOT DU BAS (PROFIL ACUSTIQUE) ---
        default_ping = fig.target_y if fig.target_y is not None else 0
        fig.ping_index = default_ping
        
        ping_intensities = fig.image[default_ping, :]
        fig.depth_port, depth_pixel_port = mesure_seafloor_backtracking(np.flip(ping_intensities[0:middle]), sound_speed)
        fig.depth_starboard, depth_pixel_starboard = mesure_seafloor_backtracking(ping_intensities[middle:], sound_speed)

        line_data.set_data(range(width), ping_intensities)
        current_ping_point.set_data([fig.easting[default_ping]], [fig.northing[default_ping]])
        ax_plot.set_xlim(0, width)
        ax_plot.set_ylim(np.min(fig.image), np.max(fig.image))
        
        if fig.target_y is not None:
            target_marker_plot.set_xdata([fig.target_x, fig.target_x])
            target_marker_plot.set_visible(True)
            ax_plot.set_title(f"Profil du ping cible n° : {default_ping} (Ligne pointillée rouge = cible exacte)")
        else:
            target_marker_plot.set_visible(False)
            ax_plot.set_title(f"Coupe d'intensité du ping n° : {default_ping}")

        if fig.selected_sample is not None:
            current_sample_marker.set_data([fig.selected_sample], [default_ping])
            current_sample_marker.set_visible(True)
        else:
            current_sample_marker.set_visible(False)

        vline_port.set_xdata([middle - depth_pixel_port])
        vline_starboard.set_xdata([middle + depth_pixel_starboard])
        vline_maker.set_xdata([0])
        horizontal_marker.set_ydata([default_ping, default_ping])
        info_text.set_text("Imagette chargée. Utilisez les flèches gauche/droite (<- / ->) pour naviguer.")
        
        fig.canvas.draw_idle()

    def select_ping(ping_index):
        fig.ping_index = ping_index
        ping_intensities_local = fig.image[ping_index, :]

        fig.depth_port, depth_pixel_port = mesure_seafloor_backtracking(np.flip(ping_intensities_local[0:fig.middle]), sound_speed)
        fig.depth_starboard, depth_pixel_starboard = mesure_seafloor_backtracking(ping_intensities_local[fig.middle:], sound_speed)

        line_data.set_ydata(ping_intensities_local)
        vline_port.set_xdata([fig.middle - depth_pixel_port])
        vline_starboard.set_xdata([fig.middle + depth_pixel_starboard])

        if fig.target_y is not None and ping_index == fig.target_y:
            target_marker_plot.set_visible(True)
        else:
            target_marker_plot.set_visible(False)

        current_ping_point.set_data([fig.easting[ping_index]], [fig.northing[ping_index]])
        if fig.selected_sample is not None:
            current_sample_marker.set_data([fig.selected_sample], [ping_index])
            current_sample_marker.set_visible(True)
        else:
            current_sample_marker.set_visible(False)

        ts_ns = int(fig.timestamp[ping_index])
        ts_dt = datetime.utcfromtimestamp(ts_ns / 1e9)
        ts_str = ts_dt.strftime("%H:%M:%S.%f")[:-3]
        e = fig.easting[ping_index]
        n = fig.northing[ping_index]
        ax_plot.set_title(
            f"Ping {ping_index} | E={e:,.3f} | N={n:,.3f}    t={ts_str}"
        )
        horizontal_marker.set_ydata([ping_index, ping_index])
        fig.canvas.draw_idle()

    # 4. Callback Interactivité Souris (Clics)
    def on_click_image(event):
        middle = fig.middle
        if event.inaxes == ax_img and event.ydata is not None:
            ping_index = int(round(event.ydata))
            if 0 <= ping_index < fig.image.shape[0]:
                fig.ping_index = ping_index
                ping_intensities_local = fig.image[ping_index, :]

                fig.depth_port, depth_pixel_port = mesure_seafloor_backtracking(np.flip(ping_intensities_local[0:middle]), sound_speed)
                fig.depth_starboard, depth_pixel_starboard = mesure_seafloor_backtracking(ping_intensities_local[middle:], sound_speed)

                line_data.set_ydata(ping_intensities_local)
                vline_port.set_xdata([middle - depth_pixel_port])
                vline_starboard.set_xdata([middle + depth_pixel_starboard])

                if fig.target_y is not None and ping_index == fig.target_y:
                    target_marker_plot.set_visible(True)
                else:
                    target_marker_plot.set_visible(False)

                current_ping_point.set_data([fig.easting[ping_index]], [fig.northing[ping_index]])
                if fig.selected_sample is not None:
                    current_sample_marker.set_data([fig.selected_sample], [ping_index])
                    current_sample_marker.set_visible(True)
                else:
                    current_sample_marker.set_visible(False)

                ts_ns = int(fig.timestamp[ping_index])
                ts_dt = datetime.utcfromtimestamp(ts_ns / 1e9)
                ts_str = ts_dt.strftime("%H:%M:%S.%f")[:-3]  # heure:mm:ss.mmm
                e = fig.easting[ping_index]
                n = fig.northing[ping_index]
                ax_plot.set_title(
                    f"Ping {ping_index} | E={e:,.3f} | N={n:,.3f}    t={ts_str}"
                )
                horizontal_marker.set_ydata([ping_index, ping_index])
                fig.canvas.draw_idle()

        elif event.inaxes == ax_plot and event.xdata is not None:
            sample_index = int(round(event.xdata))
            is_starboard = sample_index > middle
            ping_index = fig.ping_index
            
            if 0 <= sample_index < fig.image.shape[1]:
                distance_pixels = sample_index - middle
                distance_metres = sound_speed * time_step * abs(distance_pixels) / 2
                depth = fig.depth_starboard if is_starboard else fig.depth_port

                x_dist = math.sqrt(max(0, distance_metres**2 - depth**2))
                side = "starboard" if is_starboard else "port"
                (e, n) = lateral_points_pos_sonar(fig.easting[ping_index], fig.northing[ping_index], np.radians(fig.heading[ping_index]), x_dist, side)

                fig.selected_sample = sample_index
                current_sample_marker.set_data([sample_index], [ping_index])
                current_sample_marker.set_visible(True)

                info_text.set_text(f"sample index : {sample_index}, x_dist : {x_dist:.2f} m, position ({e:.2f}, {n:.2f})")
                vline_maker.set_xdata([sample_index])
                fig.canvas.draw_idle()

    # 5. Callback Clavier (Navigation)
    def on_key_press(event):
        if event.key == 'right' and fig.current_image_index < num_images - 1:
            fig.current_image_index += 1
            load_image_data(fig.current_image_index)
        elif event.key == 'left' and fig.current_image_index > 0:
            fig.current_image_index -= 1
            load_image_data(fig.current_image_index)
        elif event.key == 'up' and fig.ping_index > 0:
            select_ping(fig.ping_index - 1)
        elif event.key == 'down' and fig.ping_index < fig.image.shape[0] - 1:
            select_ping(fig.ping_index + 1)

    # 6. Connexions des événements et affichage initial
    fig.canvas.mpl_connect('button_press_event', on_click_image)
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    load_image_data(fig.current_image_index)
    plt.tight_layout()
    plt.show()