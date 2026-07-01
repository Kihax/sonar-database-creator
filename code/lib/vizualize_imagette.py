from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as patches  # <--- AJOUT : Pour dessiner le rectangle
import numpy as np
from typing import List
from .SonarPing import SonarPing
from .Imagette import Imagette

def vizualize_imagette(imagettes_list : List[Imagette], num_show_imagette : int = 1):
    """
        Show side-by-side: full file waterfall (with a red bounding box showing 
        the imagette position) and the cropped imagette.
        
        Args:
            - imagettes_list (List[Imagette]) : list of Imagette objects
            - num_show_imagette (int) : number of rows to show in the same window
    """
    num_images = len(imagettes_list)
    if num_images == 0:
        print("Empty list")
        return

    num_visibles = min(num_show_imagette, num_images)

    # 2 colonnes : Full Waterfall (avec boîte) | Imagette Ciblée
    fig, axes = plt.subplots(num_visibles, 2, figsize=(14, 5 * num_visibles), squeeze=False)
    
    # Ajustement des espaces pour les titres et le slider à droite
    plt.subplots_adjust(right=0.88, left=0.07, hspace=0.4, wspace=0.25)

    im_full_objects = []
    im_crop_objects = []
    # Liste pour stocker les objets 'rectangle' (les boîtes rouges)
    rect_objects = []

    for i in range(num_visibles):
        # Colonne 0 : Waterfall Complet de tout le fichier
        im_full = axes[i, 0].imshow([[0]], cmap='gray', aspect='auto')
        im_full_objects.append(im_full)
        axes[i, 0].set_title("1. Full File Waterfall", fontsize=9, fontweight='bold')

        # --- AJOUT : Création du rectangle rouge ---
        # On le crée initialement avec des dimensions nulles et invisible
        rect = patches.Rectangle(
            (0, 0), 0, 0,           # (x, y), width, height
            linewidth=2,            # Épaisseur du trait
            edgecolor='r',          # Couleur rouge pour le bord
            facecolor='none',       # Pas de remplissage intérieur
            alpha=0.8,              # Légère transparence
            visible=False           # Masqué par défaut
        )
        rect_objects.append(rect)
        axes[i, 0].add_patch(rect) # Ajout du rectangle au subplot de gauche
        # --------------------------------------------

        # Colonne 1 : Imagette Découpée (Zoom sur la cible)
        im_crop = axes[i, 1].imshow([[0]], cmap='gray', aspect='auto')
        im_crop_objects.append(im_crop)
        axes[i, 1].set_title("2. Cropped Imagette", fontsize=9, fontweight='bold')
        
        for ax in axes[i]:
            ax.grid(False)

    def update_view(start_idx):
        start_idx = int(round(start_idx))
        for i in range(num_visibles):
            current_img_idx = start_idx + i
            
            # Gestion des débordements de la liste d'imagettes
            if current_img_idx >= num_images:
                for ax in axes[i]:
                    ax.set_visible(False)
                rect_objects[i].set_visible(False) # Cacher la boîte
                continue
            
            for ax in axes[i]:
                ax.set_visible(True)
                
            img_data = imagettes_list[current_img_idx]
            
            # --- 1. Matrices selon TES notations ---
            matrix_crop = np.array(img_data.imagette)
            
            # --- 2. Mise à jour de la colonne 1 (Imagette zoomée) ---
            im_crop_objects[i].set_data(matrix_crop)
            im_crop_objects[i].set_clim(vmin=0, vmax=1.0)
            h_c, w_c = matrix_crop.shape
            im_crop_objects[i].set_extent([0, w_c, h_c, 0])

            # --- 3. Mise à jour de la colonne 0 (Waterfall Global ET Boîte Rouge) ---
            if img_data.full_file_waterfall is not None:
                matrix_full = np.array(img_data.full_file_waterfall)
                im_full_objects[i].set_data(matrix_full)
                im_full_objects[i].set_clim(vmin=0, vmax=250)
                h_f, w_f = matrix_full.shape
                # Important pour que les coordonnées X/Y du rectangle correspondent
                im_full_objects[i].set_extent([0, w_f, h_f, 0])
                axes[i, 0].set_visible(True)

                # --- MISE À JOUR DE LA BOÎTE ROUGE ---
                # Récupération des positions relatives stockées dans l'imagette
                box_x = img_data.start_x
                # (Ici, start_y est relative_start_y si tu as fait la correction)
                box_y = img_data.start_y
                
                # Les dimensions de la boîte sont celles de l'imagette découpée
                box_width = w_c  # matrix_crop.shape[1]
                box_height = h_c # matrix_crop.shape[0]

                # Mise à jour des coordonnées et de la taille du rectangle
                rect_objects[i].set_xy((box_x, box_y))
                rect_objects[i].set_width(box_width)
                rect_objects[i].set_height(box_height)
                rect_objects[i].set_visible(True) # Rendre le rectangle visible
                # -------------------------------------

            else:
                # Si pas de waterfall global, on cache le subplot de gauche et la boîte
                axes[i, 0].set_visible(False)
                rect_objects[i].set_visible(False)

            # Titre dynamique de la ligne
            if img_data.pings:
                ref_ping = img_data.pings[len(img_data.pings) // 2]
                fichiers_imagette = set()

                for p in img_data.pings:
                    # Si c'est un tableau NumPy, on extrait le premier élément, sinon on garde la string
                    if isinstance(p.filename, np.ndarray):
                        f_str = str(p.filename.item())  # .item() extrait proprement la string du tableau
                    else:
                        f_str = str(p.filename)
                    
                    # Maintenant f_str est une vraie String, le set va l'accepter sans broncher
                    fichiers_imagette.add(f_str)

                # On transforme le set en chaîne de caractères propre (ex: "fichier1.nc, fichier2.nc")
                filename_str = ", ".join(fichiers_imagette)

                est_centre = "Oui" if getattr(img_data, 'centered', False) else "Non"
                side_str = getattr(img_data, 'side', 'Inconnu')
                det_range = getattr(ref_ping, 'detection_range', None)
                det_range_str = f"{det_range:.3f}" if det_range is not None else "N/A"
                
                title_text = (
                    f"Imagette {current_img_idx + 1} / {num_images} | Fichiers: {filename_str}\n"
                    f"Easting : [{img_data.eastern_min:.1f} ; {img_data.eastern_max:.1f}] | "
                    f"Northing : [{img_data.northern_min:.1f} ; {img_data.northern_max:.1f}]\n"
                    f"Centered : {est_centre} | Side : {side_str} | Detection range : {det_range_str}"
                )
                axes[i, 1].set_title(f"2. Cropped Imagette\n{title_text}", fontsize=9)

        fig.canvas.draw_idle()

    # --- Gestion du Slider vertical et du Scroll de souris ---
    ax_slider = plt.axes([0.93, 0.1, 0.02, 0.8])
    max_scroll = max(0, num_images - num_visibles)
    
    slider = Slider(
        ax=ax_slider, label='', valmin=0, valmax=max_scroll, valinit=0, valstep=1, orientation='vertical'
    )
    ax_slider.invert_yaxis()

    slider.on_changed(lambda val: update_view(max_scroll - val))

    def on_scroll(event):
        if event.button == 'up':
            new_val = max(0, slider.val - 1); slider.set_val(new_val)
        elif event.button == 'down':
            new_val = min(max_scroll, slider.val + 1); slider.set_val(new_val)

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # Premier affichage
    update_view(0)
    slider.set_val(max_scroll)

    plt.show()