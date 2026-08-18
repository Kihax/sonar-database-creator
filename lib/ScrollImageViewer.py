from matplotlib import pyplot as plt

class ScrollImageViewer:
    def __init__(self, coord, imagettes):
        self.coord = coord
        self.imagettes = imagettes  # Liste d'objets ImagetteDatabase
        self.index = 0
        self.num_images = len(imagettes)
        
        # Initialisation de la figure matplotlib (ajustée pour du 3000x200)
        self.fig, self.ax = plt.subplots(figsize=(11, 4))
        self.fig.canvas.manager.set_window_title(f"Position X:{coord[0]} Y:{coord[1]}")
        
        # --- OPTIMISATION RAM : Chargement à la demande (.values) ---
        first_image_pixels = self.imagettes[self.index].image.values
        self.im_plot = self.ax.imshow(first_image_pixels, cmap='gray', origin='lower', aspect='auto')
        
        # Masquer les axes pour maximiser l'espace de l'image
        self.ax.axis('on')
        
        # Premier affichage des métadonnées dans le titre
        self.update_display()
        
        # Connexion de l'événement de la molette de la souris
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    def update_display(self):
        img_obj = self.imagettes[self.index]
        
        # Formatage compact pour le titre (sur 2 lignes pour rester lisible)
        title_text = (
            f"Imagette {self.index + 1}/{self.num_images} | Centered: {img_obj.centered} | "
            f"Range: {img_obj.detection_range} | Files: {', '.join(img_obj.survey_files)}\n"
            f"Extrêmes (E, N) -> E: ({img_obj.point_eastern[0]:.1f}, {img_obj.point_eastern[1]:.1f}) | "
            f"W: ({img_obj.point_western[0]:.1f}, {img_obj.point_western[1]:.1f}) | "
            f"S: ({img_obj.point_southern[0]:.1f}, {img_obj.point_southern[1]:.1f}) | "
            f"N: ({img_obj.point_northern[0]:.1f}, {img_obj.point_northern[1]:.1f})"
        )
        
        # Mise à jour du titre avec une taille de police légèrement réduite pour que tout rentre
        self.ax.set_title(title_text, fontsize=9, loc='center', pad=12)
        
        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        if event.inaxes is None:
            return
            
        if event.step > 0:
            self.index = (self.index + 1) % self.num_images
        elif event.step < 0:
            self.index = (self.index - 1) % self.num_images
            
        # --- OPTIMISATION RAM ---
        next_image_pixels = self.imagettes[self.index].image.values
        self.im_plot.set_data(next_image_pixels)
        
        # Ajustement dynamique du contraste
        self.im_plot.set_clim(vmin=next_image_pixels.min(), vmax=next_image_pixels.max())
        
        self.update_display()

    def show(self):
        plt.show()