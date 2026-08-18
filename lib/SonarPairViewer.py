import matplotlib.pyplot as plt
import numpy as np


def _format_img(img_tensor):
    """
    Convertit un Tensor PyTorch sonar (C, H, W) ou (H, W) en tableau NumPy 2D prêt pour Matplotlib.
    """
    if hasattr(img_tensor, "detach"):
        img_np = img_tensor.detach().cpu().numpy()
    elif hasattr(img_tensor, "cpu"):
        img_np = img_tensor.cpu().numpy()
    else:
        img_np = np.array(img_tensor)

    img_np = np.squeeze(img_np)  # Supprime les dimensions de taille 1

    # Si l'image possède plusieurs canaux (ex: 3), transpose vers (H, W, C)
    if img_np.ndim == 3 and img_np.shape[0] in [1, 3, 4]:
        img_np = np.transpose(img_np, (1, 2, 0))
        img_np = np.squeeze(img_np)

    return img_np


class SonarPairViewer:
    """
    Visionneuse interactive pour naviguer de page en page avec les flèches du clavier.
    """
    def __init__(self, dataset, distances, labels, threshold, category="ALL", page_size=3, rotate=True, split_name="Test"):
        self.dataset = dataset
        self.distances = np.array(distances)
        self.labels = np.array(labels)
        self.threshold = threshold
        self.category = category.upper()
        self.page_size = page_size
        self.rotate = rotate
        self.split_name = split_name
        self.page = 0

        self.label_map = {1: "SIMILAIRE", 0: "DISSIMILAIRE"}
        self.preds = (self.distances < self.threshold).astype(int)

        cat_indices = {
            "VP": np.where((self.labels == 1) & (self.preds == 1))[0],
            "VN": np.where((self.labels == 0) & (self.preds == 0))[0],
            "FP": np.where((self.labels == 0) & (self.preds == 1))[0],
            "FN": np.where((self.labels == 1) & (self.preds == 0))[0],
            "ALL": np.arange(len(self.labels))
        }

        self.indices = cat_indices.get(self.category, cat_indices["ALL"])
        self.total_samples = len(self.indices)
        self.total_pages = max(1, (self.total_samples + self.page_size - 1) // self.page_size)

        if self.total_samples == 0:
            print(f"Aucun échantillon disponible pour la catégorie : {self.category}")
            return

        # Initialisation de la figure Matplotlib
        self.fig = plt.figure(figsize=(16, 8))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.render_page()
        plt.show()

    def render_page(self):
        self.fig.clf()  # Efface le contenu de la page précédente
        
        start_idx = self.page * self.page_size
        end_idx = min(start_idx + self.page_size, self.total_samples)
        page_indices = self.indices[start_idx:end_idx]
        current_count = len(page_indices)

        # En-tête explicatif
        self.fig.suptitle(
            f"[{self.split_name} - {self.category}] Page {self.page + 1}/{self.total_pages} "
            f"(Paires {start_idx + 1} à {end_idx} sur {self.total_samples})\n"
            f"Navigation Clavier : ← / → (ou q / d)", 
            fontsize=12, fontweight="bold"
        )

        subfigs = self.fig.subplots(current_count, 2)
        if current_count == 1:
            subfigs = np.expand_dims(subfigs, axis=0)

        for idx_row, sample_idx in enumerate(page_indices):
            tensor_A, meta_A, tensor_B, meta_B, label_val = self.dataset[sample_idx]

            dist_val = self.distances[sample_idx]
            lbl_val = int(self.labels[sample_idx])
            pred_val = int(self.preds[sample_idx])

            str_pred = self.label_map[pred_val]
            str_real = self.label_map[lbl_val]

            imgA_np = _format_img(tensor_A)
            imgB_np = _format_img(tensor_B)

            if self.rotate:
                imgA_np = np.rot90(imgA_np)
                imgB_np = np.rot90(imgB_np)

            is_correct = (pred_val == lbl_val)
            title_color = "darkgreen" if is_correct else "darkred"

            info_text = (
                f"Paire #{sample_idx} | Prédit: {str_pred} | Réel: {str_real} | "
                f"Dist: {dist_val:.3f} (Seuil: {self.threshold:.2f})"
            )

            subfigs[idx_row, 0].imshow(imgA_np, cmap="gray", aspect="auto")
            subfigs[idx_row, 0].set_title(f"Image A — {info_text}", fontsize=9, fontweight="bold", color=title_color)
            subfigs[idx_row, 0].axis("off")

            subfigs[idx_row, 1].imshow(imgB_np, cmap="gray", aspect="auto")
            subfigs[idx_row, 1].set_title(f"Image B — {info_text}", fontsize=9, fontweight="bold", color=title_color)
            subfigs[idx_row, 1].axis("off")

        self.fig.subplots_adjust(top=0.88, bottom=0.05, hspace=0.35, wspace=0.1)
        self.fig.canvas.draw()

    def on_key(self, event):
        """Gestionnaire de touches pour la navigation"""
        if event.key in ['right', 'd', 'n']:
            if self.page < self.total_pages - 1:
                self.page += 1
                self.render_page()
        elif event.key in ['left', 'q', 'p']:
            if self.page > 0:
                self.page -= 1
                self.render_page()