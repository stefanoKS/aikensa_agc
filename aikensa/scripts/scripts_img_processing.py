import cv2
import numpy as np
from typing import Sequence, List, Tuple, Union
import os

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

def aruco_detect(
    image_bgr: np.ndarray,
    *,
    dict: str = "DICT_4X4_50",          # match your call signature (shadows built-in 'dict', but OK)
    result: str = "single",              # "single" | "array"
    return_scores: bool = False,         # only used when result="array": return (id, score) tuples
    use_refine: bool = False             # optional: try refineDetectedMarkers if available
) -> Union[int, List[int], List[Tuple[int, float]]]:
    """
    Detect ArUco markers and return either the best single ID or a list of IDs sorted by confidence.

    Confidence proxy: marker perimeter length computed from detected corners.
    Returns:
      - result == "single" -> int (best id) or -1 if none
      - result == "array"  -> list[int] (sorted by score desc) OR list[(id, score)] if return_scores=True
    """
    if image_bgr is None or image_bgr.size == 0:
        return -1 if result == "single" else ([] if not return_scores else [])

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # alpha = 1.5  # contrast control (1.0-3.0)
    alpha = 1.25  # contrast control (1.0-3.0)
    beta = 10     # brightness control (0-100)

    # apply contrast and brightness adjustment
    gray = cv2.convertScaleAbs(gray, alpha=1.0, beta=beta)
    gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=0)


    filename = f"aruco_debug_gray.png"
    base, ext = os.path.splitext(filename)
    n = 1
    while os.path.exists(filename):
        filename = f"{base}_{n}{ext}"
        n += 1

    cv2.imwrite(filename, gray)
    

    # --- get dictionary by name (case-insensitive) ---
    dict_name = dict.upper().strip()
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        try:
            dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
        except Exception:
            # final fallback (AttributeError when dict name invalid)
            if hasattr(cv2.aruco, dict_name):
                dictionary = getattr(cv2.aruco, dict_name)
            else:
                raise ValueError(f"Unknown/unsupported ArUco dictionary: {dict}")
    else:
        if hasattr(cv2.aruco, dict_name):
            dictionary = getattr(cv2.aruco, dict_name)
        else:
            raise ValueError(f"Unknown/unsupported ArUco dictionary: {dict}")

    # detector params
    params = (cv2.aruco.DetectorParameters_create()
              if hasattr(cv2.aruco, "DetectorParameters_create")
              else cv2.aruco.DetectorParameters())

    # --- detect (new API first) ---
    corners = ids = rejected = None
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, rejected = detector.detectMarkers(gray)
        if use_refine and hasattr(cv2.aruco, "refineDetectedMarkers"):
            try:
                # Refine usually needs a board; we pass an empty board to be harmless.
                board = cv2.aruco.Board_create([], [], [])
                corners, ids, rejected, _ = cv2.aruco.refineDetectedMarkers(
                    gray, board, corners, ids, rejected, dictionary, params
                )
            except Exception:
                pass
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
        if use_refine and hasattr(cv2.aruco, "refineDetectedMarkers"):
            try:
                board = cv2.aruco.Board_create([], [], [])
                corners, ids, rejected, _ = cv2.aruco.refineDetectedMarkers(
                    gray, board, corners, ids, rejected, dictionary, params
                )
            except Exception:
                pass

    if ids is None or len(ids) == 0:
        return -1 if result == "single" else ([] if not return_scores else [])

    # --- compute perimeter-based scores ---
    # corners: list of (1,4,2) floats; ids: Nx1 int
    ids = ids.flatten().astype(int)
    scores = []
    for cs in corners:
        pts = cs.reshape(-1, 2)
        perim = float(
            np.linalg.norm(pts[0] - pts[1]) +
            np.linalg.norm(pts[1] - pts[2]) +
            np.linalg.norm(pts[2] - pts[3]) +
            np.linalg.norm(pts[3] - pts[0])
        )
        scores.append(perim)
    scores = np.asarray(scores, dtype=float)

    # sort by score desc
    order = np.argsort(-scores)
    ids_sorted = ids[order]
    scores_sorted = scores[order]

    if result.lower() == "single":
        # best single id
        return int(ids_sorted[0])
    elif result.lower() == "array":
        if return_scores:
            return [(int(ids_sorted[i]), float(scores_sorted[i])) for i in range(len(ids_sorted))]
        else:
            return [int(x) for x in ids_sorted.tolist()]
    else:
        raise ValueError('result must be "single" or "array"')