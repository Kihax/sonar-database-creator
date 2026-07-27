import random

import numpy as np
import torch
from torch.utils.data import Dataset


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

        coords = list(subset_matching_groups.keys())

        # --- PAIRES POSITIVES ---
        for coord, list_img in subset_matching_groups.items():
            if len(list_img) < 2:
                continue
            for i in range(len(list_img)):
                for j in range(i + 1, len(list_img)):
                    self.pairs.append((list_img[i], list_img[j]))
                    self.labels.append(1)

        num_positives = len(self.pairs)

        # --- PAIRES NÉGATIVES ---
        if len(coords) >= 2 and num_positives > 0:
            while len(self.pairs) < 2 * num_positives:
                c1, c2 = random.sample(coords, 2)
                img1 = random.choice(subset_matching_groups[c1])
                img2 = random.choice(subset_matching_groups[c2])

                self.pairs.append((img1, img2))
                self.labels.append(0)

    def _prepare_imagette(self, img_obj):
        if hasattr(img_obj.image, "values"):
            img_np = img_obj.image.values
        elif hasattr(img_obj.image, "variable"):
            img_np = img_obj.image.variable.values
        else:
            img_np = np.array(img_obj.image)

        img_np = img_np.astype(np.float32)
        height, width = img_np.shape

        def get_vec(attr):
            val = getattr(img_obj, attr, None)
            if val is None:
                return np.zeros(height, dtype=np.float32)
            if hasattr(val, "values"):
                return val.values.astype(np.float32)
            if hasattr(val, "variable"):
                return val.variable.values.astype(np.float32)
            return np.array(val, dtype=np.float32)

        pitch_vec = get_vec("pitch")
        roll_vec = get_vec("roll")
        heading_vec = get_vec("heading")

        pitch_2d = np.repeat(pitch_vec[:, np.newaxis], width, axis=1)
        roll_2d = np.repeat(roll_vec[:, np.newaxis], width, axis=1)
        heading_2d = np.repeat(heading_vec[:, np.newaxis], width, axis=1)

        #stacked = np.stack([img_np, pitch_2d, roll_2d], axis=0)
        stacked = np.stack([img_np, heading_2d], axis=0)
        tensor_2d = torch.from_numpy(stacked)

        tensor_2d = torch.nn.functional.interpolate(
            tensor_2d.unsqueeze(0), size=self.target_size, mode="bilinear", align_corners=False
        ).squeeze(0)

        depth_val = getattr(img_obj, "depth", None)
        if depth_val is not None:
            depth_np = depth_val.values if hasattr(depth_val, "values") else np.array(depth_val)
            mean_depth = float(np.mean(depth_np))
        else:
            mean_depth = 0.0

        depth_norm = float(self.meta_config.get("depth_normalization", 500.0))
        roll_norm = float(self.meta_config.get("roll_normalization", 30.0))
        detection_range_norm = float(self.meta_config.get("detection_range_normalization", 500.0))

        mean_depth = np.clip(mean_depth / depth_norm, -1.0, 1.0)
        mean_roll = float(np.mean(roll_vec)) if len(roll_vec) > 0 else 0.0
        mean_roll = np.clip(mean_roll / roll_norm, -1.0, 1.0)

        mean_heading = float(np.mean(heading_vec)) if len(heading_vec) > 0 else 0.0
        _ = np.radians(mean_heading)

        if hasattr(img_obj, "detection_range"):
            det_range_val = img_obj.detection_range
            if hasattr(det_range_val, "values"):
                det_range = float(det_range_val.values)
            else:
                det_range = float(det_range_val)
        else:
            det_range = 0.0

        det_range = np.clip(det_range / detection_range_norm, 0.0, 1.0)

        is_centered = 1.0 if getattr(img_obj, "centered", False) else 0.0

        side_str = getattr(img_obj, "side", "")
        if hasattr(side_str, "values"):
            side_str = str(side_str.values)
        else:
            side_str = str(side_str)
        side_str = side_str.lower()

        if "starboard" in side_str or "tribord" in side_str:
            side_val = 1.0
        elif "port" in side_str or "bâbord" in side_str:
            side_val = -1.0
        else:
            side_val = 0.0

        global_meta = torch.tensor([
            #mean_depth,
            #mean_roll,
            #det_range,
            #is_centered,
            #side_val,
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