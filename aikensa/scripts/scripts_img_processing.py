import cv2
import numpy as np
from typing import Sequence, List

def crop_parts(
    img: np.ndarray, *,
    crop_start: int,
    crop_height: int,
    crop_y_positions: Sequence[int],
) -> List[np.ndarray]:
    if img is None:
        raise ValueError("Input image cannot be None.")
    h, w = img.shape[:2]
    crops = []
    x1 = max(0, min(int(crop_start), w))
    x2 = max(0, min(int(w - crop_start), w))
    for y_pos in crop_y_positions:
        y1 = max(0, min(int(y_pos), h))
        y2 = max(0, min(int(y_pos + crop_height), h))
        crops.append(img[y1:y2, x1:x2])
    return crops