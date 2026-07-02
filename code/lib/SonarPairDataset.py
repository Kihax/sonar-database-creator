import torch
from torch.utils.data import Dataset
import numpy as np
import random

class SonarPairDataset(Dataset):
    def __init__(self, matching_groups, target_size=(256, 256)):
        self.target_size = target_size
        self.pairs = []
        self.labels = []

        coords = list(matching_groups.keys())
        
        # --- PAIRES POSITIVES ---
        for coord, list_img in matching_groups.items():
            if len(list_img) < 2:
                continue
            for i in range(len(list_img)):
                for j in range(i + 1, len(list_img)):
                    self.pairs.append((list_img[i], list_img[j]))
                    self.labels.append(1)

        num_positives = len(self.pairs)

        # --- PAIRES NÉGATIVES ---
        while len(self.pairs) < 2 * num_positives:
            c1, c2 = random.sample(coords, 2)
            img1 = random.choice(matching_groups[c1])
            img2 = random.choice(matching_groups[c2])
            
            self.pairs.append((img1, img2))
            self.labels.append(0)

    def _prepare_imagette(self, img_obj):
        # 1. Image principale
        img_np = img_obj.image.values if hasattr(img_obj.image, "values") else np.array(img_obj.image)
        img_np = img_np.astype(np.float32)
        height, width = img_np.shape
        
        # 2. Gestion des attitudes ligne par ligne
        pitch_vec = img_obj.pitch.astype(np.float32) if img_obj.pitch is not None else np.zeros(height, dtype=np.float32)
        heading_vec = img_obj.heading.astype(np.float32) if img_obj.heading is not None else np.zeros(height, dtype=np.float32)
        
        # Transformation en matrices 2D
        pitch_2d = np.repeat(pitch_vec[:, np.newaxis], width, axis=1)
        heading_2d = np.repeat(heading_vec[:, np.newaxis], width, axis=1)
        
        # Empilement : [Image Sonar, Pitch 2D, Heading 2D] -> (3, H, W)
        stacked = np.stack([img_np, pitch_2d, heading_2d], axis=0)
        tensor_2d = torch.from_numpy(stacked)
        
        # Redimensionnement spatial fixe
        tensor_2d = torch.nn.functional.interpolate(
            tensor_2d.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False
        ).squeeze(0)
        
        # 3. Métadonnées globales (scalaires)
        mean_depth = np.mean(img_obj.depth) if img_obj.depth is not None else 0.0
        mean_heading = np.mean(heading_vec) if len(heading_vec) > 0 else 0.0
        
        # On regroupe les deux infos globales dans un vecteur de taille 2
        heading_rad = np.radians(mean_heading)
        global_meta = torch.tensor([
            mean_depth, 
            np.sin(heading_rad), 
            np.cos(heading_rad)
        ], dtype=torch.float32)
        
        return tensor_2d, global_meta

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_obj_A, img_obj_B = self.pairs[idx]
        label = self.labels[idx]
        
        tensor_A, meta_A = self._prepare_imagette(img_obj_A)
        tensor_B, meta_B = self._prepare_imagette(img_obj_B)
        
        return tensor_A, meta_A, tensor_B, meta_B, torch.tensor(label, dtype=torch.float32)