from matplotlib import pyplot as plt
import math


class VisualiseurResultatsSiamois:
    """
    Affiche des paires d'images issues du dataset de validation,
    avec défilement (scroll) pour naviguer dans la liste.
    """
    def __init__(self, liste_indices_distances, dataset, seuil):
        """
        Args:
            liste_indices_distances: Liste de tuples (index_dans_dataset, distance_predite)
            dataset: Le dataset de validation (val_dataset)
            seuil: Le seuil de décision optimal appliqué
        """
        self.data_list = liste_indices_distances
        self.dataset = dataset
        self.threshold = seuil
        self.total = len(self.data_list)
        
        if self.total == 0:
            print("⚠️ No data to display. The list of indices and distances is empty.")
            return

        self.current_idx = 0

        # Création de la figure (2 colonnes pour la paire)
        self.fig, self.axes = plt.subplots(1, 2, figsize=(11, 6))
        # Ajustement pour laisser de la place au titre
        plt.subplots_adjust(top=0.85, bottom=0.1, left=0.07, right=0.93, hspace=0.2, wspace=0.2)
        
        # Objets imshow vides pour mise à jour rapide
        self.im_A = self.axes[0].imshow([[0]], cmap='gray', aspect='auto')
        self.im_B = self.axes[1].imshow([[0]], cmap='gray', aspect='auto')
        
        for ax in self.axes:
            ax.axis('off') # Cache les axes par défaut

        # Connexion des événements (souris et clavier)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Premier affichage
        self.update_display()
        plt.show()

    def update_display(self):
        """Met à jour le contenu des plots sans recréer la figure."""
        if not (0 <= self.current_idx < self.total):
            return

        # Récupération des données du tuple
        dataset_idx, dist_predite = self.data_list[self.current_idx]
        
        # Extraction des images et métas depuis le dataset PyTorch
        img_A, meta_A, img_B, meta_B, label_torch = self.dataset[dataset_idx]
        
        # Conversion pour affichage
        img_A_np = img_A[0].numpy()
        img_B_np = img_B[0].numpy()
        meta_A_np = meta_A.numpy()
        meta_B_np = meta_B.numpy()
        actual_label = int(label_torch.item()) # 1: Même, 0: Diff
        
        # Calcul des types de prédiction
        est_positif = (dist_predite < self.threshold)
        
        type_str = ""
        color = 'black'
        
        if actual_label == 1: # Vérité terrain: Même
            if est_positif:
                type_str = "True Positive (Same object, correctly classified)"
                color = 'green'
            else:
                type_str = "False Negative (Same object, misclassified)"
                color = 'red'
        else: # Vérité terrain: Différent
            if est_positif:
                type_str = "False Positive (Different objects, misclassified)"
                color = 'darkorange'
            else:
                type_str = "True Negative (Different objects, correctly classified)"
                color = 'blue'

        # --- ADAPTATION ICI : Protection si les métadonnées sont vides ---
        if len(meta_A_np) >= 3:
            depth_A, sin_A, cos_A = meta_A_np[0], meta_A_np[1], meta_A_np[2]
            cap_A = (math.degrees(math.atan2(sin_A, cos_A)) + 360) % 360
            title_A = f"Image A\nProf: {depth_A:.1f}m | Cap: {cap_A:.1f}°"
        else:
            title_A = "Image A\n(Pas de métadonnées)"

        if len(meta_B_np) >= 3:
            depth_B, sin_B, cos_B = meta_B_np[0], meta_B_np[1], meta_B_np[2]
            cap_B = (math.degrees(math.atan2(sin_B, cos_B)) + 360) % 360
            title_B = f"Image B\nProf: {depth_B:.1f}m | Cap: {cap_B:.1f}°"
        else:
            title_B = "Image B\n(Pas de métadonnées)"

        # --- Mise à jour dynamique ---
        # Titre principal avec infos globales et couleur d'état
        infos_globale = (
            f"Cas n° {self.current_idx + 1} / {self.total}  |  "
            f"Distance: {dist_predite:.4f} (Seuil: {self.threshold:.2f})\n"
            f"Statut : {type_str}"
        )
        self.fig.suptitle(infos_globale, fontsize=12, fontweight='bold', color=color)

        # Mise à jour des images
        self.im_A.set_data(img_A_np)
        self.im_B.set_data(img_B_np)
        
        # Ajustement automatique des contrastes
        self.im_A.set_clim(vmin=img_A_np.min(), vmax=img_A_np.max())
        self.im_B.set_clim(vmin=img_B_np.min(), vmax=img_B_np.max())
        
        # Titres des sous-plots mis à jour
        self.axes[0].set_title(title_A, fontsize=10)
        self.axes[1].set_title(title_B, fontsize=10)
        
        # Rafraîchissement du canvas
        self.fig.canvas.draw_idle()

    # Gestionnaires d'événements
    def on_scroll(self, event):
        """Réagit à la molette de la souris."""
        if event.button == 'up':
            self.change_index(-1)
        elif event.button == 'down':
            self.change_index(1)

    def on_key(self, event):
        """Réagit aux flèches du clavier."""
        if event.key == 'right' or event.key == 'down':
            self.change_index(1)
        elif event.key == 'left' or event.key == 'up':
            self.change_index(-1)

    def change_index(self, delta):
        """Change l'indice de manière circulaire."""
        self.current_idx = (self.current_idx + delta) % self.total
        self.update_display()