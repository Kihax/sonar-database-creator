import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "code"))

from lib.augmentation import apply_translation_augmentation


def test_translation_augmentation_moves_content_and_preserves_shape():
    image = np.arange(25, dtype=np.float32).reshape(5, 5)

    shifted = apply_translation_augmentation(image, shift=(1, 2), fill_value=0.0)

    assert shifted.shape == image.shape
    assert shifted[1, 2] == image[0, 0]
    assert shifted[0, 0] == 0.0
    assert shifted[4, 4] == 0.0
