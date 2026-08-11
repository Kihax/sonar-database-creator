import itertools
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lib.ReadDatabaseImagette import ReadDatabaseImagette
from lib.file_management import get_tree_from_file

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

    img_np = np.squeeze(img_np)  # Supprime les dimensions de taille 1 (ex: (1, 3000, 100) -> (3000, 100))

    # Si l'image possède plusieurs canaux (ex: 3), transpose vers (H, W, C)
    if img_np.ndim == 3 and img_np.shape[0] in [1, 3, 4]:
        img_np = np.transpose(img_np, (1, 2, 0))
        img_np = np.squeeze(img_np)

    return img_np


class SonarPairDataset(Dataset):
    def __init__(self, subset_matching_groups, target_size=(256, 256), meta_config=None):
        """
        subset_matching_groups: Contient UNIQUEMENT les coordonnées dédiées
                                soit au Train, soit à la Val, soit au Test.
        """
        self.target_size = target_size
        self.meta_config = meta_config or {}
        self.pairs = []
        self.labels = []

        self._build_pairs(subset_matching_groups)

    def _build_pairs(self, subset_groups):
        """Génère les paires positives et négatives équilibrées (1:1)."""
        # 1. GENERATION DES PAIRES POSITIVES
        for images in subset_groups.values():
            if len(images) >= 2:
                for img1, img2 in itertools.combinations(images, 2):
                    self.pairs.append((img1, img2))
                    self.labels.append(1.0)

        num_positives = len(self.pairs)
        coords = list(subset_groups.keys())

        if len(coords) < 2 or num_positives == 0:
            return

        # 2. GENERATION DES PAIRES NEGATIVES
        all_imagettes = [(coord, img) for coord, imgs in subset_groups.items() for img in imgs]
        
        seen_pairs = set()
        max_attempts = num_positives * 20
        attempts = 0

        while len(seen_pairs) < num_positives and attempts < max_attempts:
            attempts += 1
            idx1, idx2 = random.sample(range(len(all_imagettes)), 2)
            (c1, img1), (c2, img2) = all_imagettes[idx1], all_imagettes[idx2]

            if c1 != c2:
                pair_key = (min(id(img1), id(img2)), max(id(img1), id(img2)))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    self.pairs.append((img1, img2))
                    self.labels.append(0.0)

        if len(seen_pairs) < num_positives:
            print(f"⚠️  Attention: Seulement {len(seen_pairs)} paires négatives uniques créées / {num_positives} demandées.")

    @staticmethod
    def _extract_array(obj, attr_name=None, default_shape=None):
        """Helper générique et propre pour extraire des numpy arrays depuis un objet."""
        val = getattr(obj, attr_name) if attr_name else obj
        if val is None:
            return np.zeros(default_shape, dtype=np.float32) if default_shape else None

        if hasattr(val, "values"):
            res = val.values
        elif hasattr(val, "variable"):
            res = val.variable.values
        else:
            res = np.array(val)

        return res.astype(np.float32)

    def _prepare_imagette(self, img_obj):
        # 1. Image de base
        img_np = self._extract_array(img_obj.image)
        height, width = img_np.shape

        # 2. Extraction des sous-canaux / métadonnées vectorielles
        pitch_vec = self._extract_array(img_obj, "pitch", default_shape=(height,))
        roll_vec = self._extract_array(img_obj, "roll", default_shape=(height,))
        heading_vec = self._extract_array(img_obj, "heading", default_shape=(height,))
        depth_vec = self._extract_array(img_obj, "depth", default_shape=(height,))

        # --- BLOC MODULABLE : CANAUX 2D SUPPLEMENTAIRES ---
        # Décommentez les lignes selon les besoins du modèle :
        
        # pitch_2d = np.repeat(pitch_vec[:, np.newaxis], width, axis=1)
        # roll_2d = np.repeat(roll_vec[:, np.newaxis], width, axis=1)
        # heading_2d = np.repeat(heading_vec[:, np.newaxis], width, axis=1)
        # heading_sin = np.sin(heading_2d)
        # heading_cos = np.cos(heading_2d)

        # Empilement dynamique des canaux souhaités (ex: [img_np, pitch_2d, roll_2d])
        channels = [img_np]
        stacked = np.stack(channels, axis=0)

        # Conversion et redimensionnement bilinéaire tensoriel
        tensor_2d = torch.from_numpy(stacked).unsqueeze(0)  # (1, C, H, W)
        tensor_2d = torch.nn.functional.interpolate(
            tensor_2d, size=self.target_size, mode="bilinear", align_corners=False
        ).squeeze(0)  # (C, target_H, target_W)

        # --- BLOC MODULABLE : METADONNEES GLOBALES (1D) ---
        depth_norm = float(self.meta_config.get("depth_normalization", 50.0))
        roll_norm = float(self.meta_config.get("roll_normalization", 30.0))
        detection_range_norm = float(self.meta_config.get("detection_range_normalization", 500.0))

        # Profondeur moyenne
        depth_val = getattr(img_obj, "depth", None)
        if depth_val is not None:
            depth_np = self._extract_array(depth_val)
            mean_depth = float(np.mean(depth_np))
        else:
            mean_depth = 0.0
        mean_depth = np.clip(mean_depth / depth_norm, -1.0, 1.0)

        # Roulis moyen
        mean_roll = float(np.mean(roll_vec)) if len(roll_vec) > 0 else 0.0
        mean_roll = np.clip(mean_roll / roll_norm, -1.0, 1.0)

        # Cap moyen
        mean_heading = float(np.mean(heading_vec)) if len(heading_vec) > 0 else 0.0
        mean_heading_rad = np.radians(mean_heading)

        # Portée de détection
        det_range_val = getattr(img_obj, "detection_range", 0.0)
        det_range = float(self._extract_array(det_range_val) if hasattr(det_range_val, "values") else det_range_val)
        det_range = np.clip(det_range / detection_range_norm, 0.0, 1.0)

        # Centrage & Côté (Bâbord / Tribord)
        is_centered = 1.0 if getattr(img_obj, "centered", False) else 0.0

        side_str = str(getattr(img_obj, "side", "")).lower()
        if "starboard" in side_str or "tribord" in side_str:
            side_val = 1.0
        elif "port" in side_str or "bâbord" in side_str:
            side_val = -1.0
        else:
            side_val = 0.0

        # Liste modulable de métadonnées globales à passer au modèle
        meta_features = [
            # mean_depth,
            # mean_roll,
            # det_range,
            # is_centered,
            # side_val,
            # np.sin(mean_heading_rad),
            # np.cos(mean_heading_rad),
        ]

        global_meta = torch.tensor(meta_features, dtype=torch.float32)

        return tensor_2d, global_meta

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_obj_A, img_obj_B = self.pairs[idx]
        label = self.labels[idx]

        tensor_A, meta_A = self._prepare_imagette(img_obj_A)
        tensor_B, meta_B = self._prepare_imagette(img_obj_B)

        return tensor_A, meta_A, tensor_B, meta_B, torch.tensor(label, dtype=torch.float32)


def build_dataloader(groups: dict, config: dict, batch_size: int | None = None, shuffle: bool = False):
    target_size = tuple(config.get("target_size", (256, 256)))
    effective_batch_size = batch_size if batch_size is not None else int(config.get("batch_size", 16))
    dataset = SonarPairDataset(groups, target_size=target_size, meta_config=config.get("model", {}))
    
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=shuffle,
        num_workers=config.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset


def prepare_datasets(config: dict):
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception as error:
        print(f"⚠️  Impossible de changer la stratégie de partage multiprocessing: {error}")

    dt = get_tree_from_file(config["dataset"] + ".nc", config["dataset_folder"])
    dts = [dt]

    rdi = ReadDatabaseImagette(dts)
    rdi.extract()
    global_imagettes = rdi.pos_imagette

    total_imagettes = sum(len(images) for images in global_imagettes.values())
    print(f"Nombre total d'objets imagettes (bruts) : {total_imagettes}")

    # Récupération dynamique depuis la CONFIG (défaut : 15)
    min_imagettes = int(config.get("min_imagettes", 15))

    # Filtrage des zones
    matching_groups = {
        coord: images for coord, images in global_imagettes.items() if len(images) > min_imagettes
    }

    imagettes_conservees = sum(len(images) for images in matching_groups.values())
    print(f"Zones conservées (> {min_imagettes} imagettes) : {len(matching_groups)} / {len(global_imagettes)}")
    print(f"Total d'imagettes conservées : {imagettes_conservees}")

    all_coords = list(matching_groups.keys())
    random.shuffle(all_coords)

    total_groups = len(all_coords)
    if total_groups == 0:
        raise ValueError(f"❌ Aucune zone ne possède plus de {min_imagettes} imagettes !")

    train_groups_count = int(config.get("train_ratio", 0.80) * total_groups)
    val_groups_count = int(config.get("val_ratio", 0.10) * total_groups)

    train_coords = all_coords[:train_groups_count]
    val_coords = all_coords[train_groups_count : train_groups_count + val_groups_count]
    test_coords = all_coords[train_groups_count + val_groups_count :]

    train_groups = {coord: matching_groups[coord] for coord in train_coords}
    val_groups = {coord: matching_groups[coord] for coord in val_coords}
    test_groups = {coord: matching_groups[coord] for coord in test_coords}

    print("\nRépartition des points d'intérêt (Objets physiques) :")
    print(f"  ├─ Train : {len(train_groups)} objets")
    print(f"  ├─ Val   : {len(val_groups)} objets")
    print(f"  └─ Test  : {len(test_groups)} objets")

    return train_groups, val_groups, test_groups