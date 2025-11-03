from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

# =========================
# 1) Crop config
# =========================
CROP_TOP_HALF = True   # kept for parity with your spec (unused if CROP_FROM_TOP=True)
CROP_FROM_TOP = True   # enable or disable top cropping
CROP_HEIGHT_PIXELS = 64

# =========================
# 2) Side trim in pixels
# =========================
TRIM_LEFT  = 60   # <-- CHANGE ME
TRIM_RIGHT = 120  # <-- CHANGE ME

# =========================
# 3) Split config
# =========================
CENTER_TILE_WIDTH = 128         # not used (center is ignored)
CENTER_OVERLAP_PCT = 20         # not used (center is ignored)
LEFTMOST_CUSTOM_WIDTH  = 128    # width for left tile (no overlap)
RIGHTMOST_CUSTOM_WIDTH = 128    # width for right tile (no overlap)

# =========================
# 4) Signed dx ranges (per side)
#     dx = x(class_1) - x(class_0)
#     OK if low <= dx <= high (signed, NOT absolute)
# =========================
DX_RANGE_LEFT  = (-15.0,  +15.0) 
DX_RANGE_RIGHT = (-15.0, +15.0) 

# =========================
# Runtime parameters
# =========================
YOLO_CONF = 0.20    # confidence threshold
YOLO_IOU  = 0.50    # NMS IoU threshold
DRAW_RADIUS = 2
DRAW_THICK  = 2




@dataclass
class TileDetResult:
    tile_name: str                   # "left" or "right"
    roi_xywh: Tuple[int, int, int, int]  # (x, y, w, h) in the trimmed/cropped image
    kpts: Optional[np.ndarray]       # shape (N, K, 2) in absolute coords (full image)
    kpts_scores: Optional[np.ndarray]# shape (N, K)
    boxes_xyxy: Optional[np.ndarray] # shape (N, 4) in absolute coords (full image)
    scores: Optional[np.ndarray]     # shape (N,)
    classes: Optional[np.ndarray]    # shape (N,)


@dataclass
class J59JResult:
    annotated_bgr: np.ndarray
    left: TileDetResult
    right: TileDetResult
    # You can add derived fields here later if needed (e.g., booleans for OK/NG)


def _safe_crop_top(img: np.ndarray, crop_height: Optional[int]) -> np.ndarray:
    if crop_height is None or crop_height <= 0 or crop_height >= img.shape[0]:
        return img
    return img[:crop_height, :, :]


def _apply_side_trim(img: np.ndarray, trim_left: int, trim_right: int) -> Tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    x0 = max(0, int(trim_left))
    x1 = max(x0, w - int(trim_right))
    return img[:, x0:x1], x0, x1 - x0  # trimmed image, offset_x, new_width


def _make_left_right_tiles(img: np.ndarray,
                           left_w: int,
                           right_w: int) -> Dict[str, Tuple[np.ndarray, Tuple[int,int,int,int]]]:
    """Return dict: name -> (tile_img, (x,y,w,h)) in the given image coordinates."""
    H, W = img.shape[:2]
    lw = int(max(1, min(left_w, W)))
    rw = int(max(1, min(right_w, W)))

    # clamp widths if they overlap
    if lw + rw > W:
        # shrink proportionally
        total = lw + rw
        lw = int(lw * (W / total))
        rw = max(1, W - lw)

    left_roi  = (0, 0, lw, H)
    right_roi = (W - rw, 0, rw, H)

    lx, ly, lw, lh = left_roi
    rx, ry, rw, rh = right_roi

    left_img  = img[ly:ly+lh, lx:lx+lw]
    right_img = img[ry:ry+rh, rx:rx+rw]
    return {
        "left":  (left_img,  left_roi),
        "right": (right_img, right_roi),
    }


def _run_yolo_pose(model, tile_bgr: np.ndarray,
                   conf: float = YOLO_CONF, iou: float = YOLO_IOU):
    """
    Runs Ultralytics YOLO model on a BGR tile. Returns the 'results' object.
    Compatible with ultralytics >=8.0.
    """
    # Ultralytics expects RGB by default; it also accepts BGR but we convert to be safe.
    if tile_bgr is None or tile_bgr.size == 0:
        print("⚠️ Warning: Empty tile passed to YOLO inference.")
        return None
    rgb = tile_bgr
    # .predict() or __call__ returns a Results list; set verbose=False to quiet
    
    return model.predict(source=rgb, conf=conf, iou=iou, verbose=False, save=False, imgsz=256)


def _extract_pose_from_results(results, tile_offset_xy: Tuple[int, int]) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
]:
    """
    Extract (kpts_xy, kpts_scores, boxes_xyxy, scores, classes) from Ultralytics Results.
    All coordinates are converted to ABSOLUTE coordinates in the full (trimmed/cropped) image space
    using tile_offset.
    Shapes:
      kpts_xy: (N, K, 2)
      kpts_scores: (N, K)
      boxes_xyxy: (N, 4)
      scores: (N,)
      classes: (N,)
    """
    ox, oy = tile_offset_xy

    if not results or len(results) == 0:
        return None, None, None, None, None

    r0 = results[0]  # one image per call
    # Keypoints
    kpts_xy = None
    kpts_scores = None
    if getattr(r0, "keypoints", None) is not None and r0.keypoints is not None:
        # r0.keypoints.xy: (N, K, 2) in tile coordinates
        try:
            kpts_xy = r0.keypoints.xy.cpu().numpy()
        except Exception:
            # Fallback for older versions
            kpts_xy = np.asarray(r0.keypoints.xy)
        try:
            kpts_scores = r0.keypoints.conf.cpu().numpy()
        except Exception:
            kpts_scores = None
        # offset to full image coords
        if kpts_xy is not None:
            kpts_xy[..., 0] += ox
            kpts_xy[..., 1] += oy

    # Boxes
    boxes_xyxy = None
    scores = None
    classes = None
    if getattr(r0, "boxes", None) is not None and r0.boxes is not None:
        try:
            boxes_xyxy = r0.boxes.xyxy.cpu().numpy()
            scores     = r0.boxes.conf.cpu().numpy()
            classes    = r0.boxes.cls.cpu().numpy()
        except Exception:
            boxes_xyxy = np.asarray(r0.boxes.xyxy)
            scores     = np.asarray(r0.boxes.conf)
            classes    = np.asarray(r0.boxes.cls)
        if boxes_xyxy is not None:
            boxes_xyxy[:, [0, 2]] += ox
            boxes_xyxy[:, [1, 3]] += oy

    return kpts_xy, kpts_scores, boxes_xyxy, scores, classes


def _draw_tile_annotations(img_bgr: np.ndarray,
                           det: TileDetResult,
                           color=(0, 255, 0)) -> None:
    """Draw keypoints and boxes for a tile on the given image."""
    if det.kpts is not None:
        for n in range(det.kpts.shape[0]):
            for (x, y) in det.kpts[n]:
                cv2.circle(img_bgr, (int(x), int(y)), DRAW_RADIUS, color, -1, lineType=cv2.LINE_4)
    # if det.boxes_xyxy is not None:
    #     for (x1, y1, x2, y2) in det.boxes_xyxy:
    #         cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, DRAW_THICK, lineType=cv2.LINE_AA)

def J59J_Set_Check(img_bgr: np.ndarray,
                   model_left,
                   model_right,
                   return_annotated: bool = True) -> Tuple[J59JResult, Optional[bool]]:
    """
    Main entry you can call like:
        result, ok_flag = J59J_Set_Check(self.SetCorrectInspectionImages[i],
                                         self.AGCJ59JRH_SET_LEFT,
                                         self.AGCJ59JRH_SET_RIGHT)

    - Applies top crop and side trims.
    - Creates left and right tiles (no center).
    - Runs YOLO pose on each tile with the respective model.
    - Returns:
        result: J59JResult with parsed detections for left/right and an annotated image.
        ok_flag: None (placeholder) — you implement your pass/fail logic and set this later.
    """
    assert img_bgr is not None and img_bgr.ndim == 3, "img_bgr must be a BGR image (H,W,3)."

    # 1) Optional top cropping (takes precedence over CROP_TOP_HALF)
    proc = img_bgr.copy()
    if CROP_FROM_TOP:
        proc = _safe_crop_top(proc, CROP_HEIGHT_PIXELS)
    elif CROP_TOP_HALF:
        h = proc.shape[0]
        proc = proc[: max(1, h // 2), :, :]

    # 2) Side trims
    proc, offset_x, _ = _apply_side_trim(proc, TRIM_LEFT, TRIM_RIGHT)
    # Global offset for tiles
    global_offset = (offset_x, 0)

    # 3) Make LEFT and RIGHT tiles
    tiles = _make_left_right_tiles(proc,
                                   left_w=LEFTMOST_CUSTOM_WIDTH,
                                   right_w=RIGHTMOST_CUSTOM_WIDTH)

    # 4) Run YOLO pose on each tile
    # Prepare container
    left_det  = TileDetResult("left",  (0, 0, 0, 0), None, None, None, None, None)
    right_det = TileDetResult("right", (0, 0, 0, 0), None, None, None, None, None)

    # LEFT
    left_img, (lx, ly, lw, lh) = tiles["left"]
    l_results = _run_yolo_pose(model_left, left_img)
    lk, lks, lb, ls, lc = _extract_pose_from_results(l_results, tile_offset_xy=(lx + global_offset[0], ly))
    left_det = TileDetResult("left", (lx + global_offset[0], ly, lw, lh), lk, lks, lb, ls, lc)

    # RIGHT
    right_img, (rx, ry, rw, rh) = tiles["right"]
    r_results = _run_yolo_pose(model_right, right_img)
    rk, rks, rb, rs, rc = _extract_pose_from_results(r_results, tile_offset_xy=(rx + global_offset[0], ry))
    right_det = TileDetResult("right", (rx + global_offset[0], ry, rw, rh), rk, rks, rb, rs, rc)

    # 5) Optional annotation
    annotated = img_bgr.copy()
    if return_annotated:
        _draw_tile_annotations(annotated, left_det,  color=(0, 255, 0))
        _draw_tile_annotations(annotated, right_det, color=(255, 0, 0))
        # Outline tile ROIs (for visual debugging)
        for (x, y, w, h), col in [(left_det.roi_xywh, (0, 255, 0)), (right_det.roi_xywh, (255, 0, 0))]:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), col, 1, lineType=cv2.LINE_AA)

    # 6) Return result + overall OK (based on per-side ranges)
    result = J59JResult(annotated_bgr=annotated, left=left_det, right=right_det)

    # Save preview if you like (fixed extension)
    # cv2.imwrite("./res.png", result.annotated_bgr)

    EVAL = evaluate_sides_and_annotate(
        result,
        dx_range_left=DX_RANGE_LEFT,
        dx_range_right=DX_RANGE_RIGHT,
        min_conf=0.2,
        draw_text=True,
    )
    
    if EVAL["left"]["dx"] is not None:
        print(f"[LEFT]  Δx (class1 - class0): {EVAL['left']['dx']:.2f} px, OK: {EVAL['left']['ok']}")
    else:
        print("[LEFT]  Missing class 0 or class 1")

    if EVAL["right"]["dx"] is not None:
        print(f"[RIGHT] Δx (class1 - class0): {EVAL['right']['dx']:.2f} px, OK: {EVAL['right']['ok']}")
    else:
        print("[RIGHT] Missing class 0 or class 1")


    overall_ok = EVAL["left"]["ok"] and EVAL["right"]["ok"]
    return result.annotated_bgr, overall_ok


def _best_detection_xy_for_class(det: TileDetResult,
                                 target_cls: int,
                                 min_conf: float = 0.0) -> Optional[Tuple[float, float]]:
    """
    Pick the best detection (highest confidence) for a given class.
    Returns (x, y) in absolute image coordinates, or None if not found.
    For K>1, returns the average of the keypoints' x,y.
    """
    if det.kpts is None or det.classes is None:
        return None

    cls_arr = det.classes.astype(int)
    idxs = np.where(cls_arr == int(target_cls))[0]
    if idxs.size == 0:
        return None

    # pick scores: prefer keypoint conf mean, else box score
    scores = []
    for i in idxs:
        if det.kpts_scores is not None:
            scores.append(float(np.nanmean(det.kpts_scores[i])))
        elif det.scores is not None:
            scores.append(float(det.scores[i]))
        else:
            scores.append(0.0)
    scores = np.array(scores)
    # filter by min_conf if available
    valid = np.where(scores >= min_conf)[0]
    if valid.size == 0:
        return None

    best_local = valid[np.argmax(scores[valid])]
    best_idx = idxs[best_local]

    # K=1 → first point; K>1 → mean across keypoints
    kpts_xy = det.kpts[best_idx]  # shape (K, 2)
    x = float(np.mean(kpts_xy[:, 0]))
    y = float(np.mean(kpts_xy[:, 1]))
    return (x, y)


def signed_dx_between_classes(det: TileDetResult,
                              cls0: int = 0,
                              cls1: int = 1,
                              min_conf: float = 0.0) -> Optional[float]:
    """
    Compute signed delta-x = x(cls1) - x(cls0)
      > 0 → class 1 is to the RIGHT of class 0
      < 0 → class 1 is to the LEFT  of class 0
    Returns None if either class is missing (or filtered by min_conf).
    """
    p0 = _best_detection_xy_for_class(det, cls0, min_conf=min_conf)
    p1 = _best_detection_xy_for_class(det, cls1, min_conf=min_conf)
    if p0 is None or p1 is None:
        return None
    x0, _ = p0
    x1, _ = p1
    return x1 - x0


def _draw_border(img_bgr: np.ndarray, roi_xywh: Tuple[int, int, int, int],
                 color: Tuple[int, int, int], thickness: int = 4) -> None:
    x, y, w, h = roi_xywh
    cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, thickness, lineType=cv2.LINE_4)

def evaluate_sides_and_annotate(result: J59JResult,
                                dx_range_left: Tuple[float, float],
                                dx_range_right: Tuple[float, float],
                                min_conf: float = 0.0,
                                draw_text: bool = True) -> Dict[str, Dict]:
    """
    For each side:
      - Check that both class 0 and class 1 exist
      - Compute signed dx = x(class_1) - x(class_0)
      - OK if dx_range_low <= dx <= dx_range_high (per side)
      - Draw GREEN border if OK, RED if NG
      - Optionally draw dx and "OK"/"NG" labels on the image

    Returns:
      {
        'left':  {'dx': Optional[float], 'ok': bool, 'classes_found': bool, 'range': (low, high)},
        'right': {'dx': Optional[float], 'ok': bool, 'classes_found': bool, 'range': (low, high)}
      }
    """
    out = {}

    def _check_one(det: TileDetResult, dx_range: Tuple[float, float], side_name: str):
        low, high = dx_range
        p0 = _best_detection_xy_for_class(det, 0, min_conf=min_conf)
        p1 = _best_detection_xy_for_class(det, 1, min_conf=min_conf)
        classes_found = (p0 is not None) and (p1 is not None)

        dx = None
        ok = False
        if classes_found:
            dx = float(p1[0] - p0[0])
            ok = (low <= dx <= high)

        # choose color
        color = (0, 255, 0) if ok else (0, 0, 255)  # green OK / red NG
        _draw_border(result.annotated_bgr, det.roi_xywh, color=color, thickness=5)

        # draw label
        if draw_text:
            x, y, w, h = det.roi_xywh
            if dx is not None:
                # label = f"{side_name.upper()}: dx={dx:.1f}  OK" if ok else f"{side_name.upper()}: dx={dx:.1f}  NG"
                label = f"{side_name.upper()} OK" if ok else f"{side_name.upper()} NG"
            else:
                label = f"{side_name.upper()}: MISSING"
            cv2.putText(result.annotated_bgr, label,
                        (x + 6, max(25, y + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color, 2, cv2.LINE_AA)
        return {"dx": dx, "ok": ok, "classes_found": classes_found, "range": (low, high)}

    out["left"]  = _check_one(result.left,  dx_range_left,  "left")
    out["right"] = _check_one(result.right, dx_range_right, "right")
    return out