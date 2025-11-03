from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import os

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
LEFTMOST_CUSTOM_WIDTH  = 128    # width for left tile (no overlap)
RIGHTMOST_CUSTOM_WIDTH = 128    # width for right tile (no overlap)

# =========================
# 4) Signed dx ranges (per side)
#     dx = x(class_1) - x(class_0)
#     OK if low <= dx <= high (signed, NOT absolute)
# =========================
DX_RANGE_LEFT  = (-5.0,  +5.0) 
DX_RANGE_RIGHT = (-5.0, +5.0) 

# =========================
# Runtime parameters
# =========================
YOLO_CONF = 0.20    # confidence threshold
YOLO_IOU  = 0.50    # NMS IoU threshold
DRAW_RADIUS = 3
DRAW_THICK  = 2

# =========================
# Center sliding-window config
# =========================
CENTER_CROP_WIDTH   = 128                # default 128 (user-adjustable)
CENTER_CROP_HEIGHT  = CROP_HEIGHT_PIXELS # default: use your top crop height
CENTER_TILE_WIDTH    = 128          # width of each center window
CENTER_OVERLAP_PCT   = 0           # 0-90 (% overlap)
CENTER_KPT_PAIR      = (0, 1)       # which two keypoints to measure distance
CENTER_DIST_RANGE    = (20.0, 60.0) # OK if min_px <= distance <= max_px
CENTER_MIN_KPT_CONF  = 0.25         # keypoint conf threshold to consider it "exists"
CENTER_DRAW_OK_GREEN = True         # draw green on OK (True) or only draw red on NG (False)


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


def _run_yolo_pose(model, tile_bgr: np.ndarray, conf: float = YOLO_CONF, iou: float = YOLO_IOU):
    """Run YOLO model safely on a BGR tile."""
    if tile_bgr is None or tile_bgr.size == 0:
        print("⚠️ Warning: Empty tile passed to YOLO inference.")
        return None

    rgb = tile_bgr
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
                cv2.circle(img_bgr, (int(x), int(y)), DRAW_RADIUS, color, -1, lineType=cv2.LINE_AA)
    if det.boxes_xyxy is not None:
        for (x1, y1, x2, y2) in det.boxes_xyxy:
            cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, DRAW_THICK, lineType=cv2.LINE_AA)

def _center_strip_after_lr(proc: np.ndarray, left_w: int, right_w: int):
    """
    Returns (center_img, (x,y,w,h)) for the middle strip after removing fixed left/right widths.
    Assumes 'proc' is already top-cropped and side-trimmed.
    Height is full proc height (typically CROP_HEIGHT_PIXELS).
    """
    H, W = proc.shape[:2]
    x0 = int(left_w)
    x1 = int(W - right_w)
    x0 = max(0, min(x0, W))
    x1 = max(x0, min(x1, W))
    strip = proc[0:H, x0:x1]
    return strip, (x0, 0, x1 - x0, H)

def _sliding_windows(width: int, tile_w: int, overlap_pct: float):
    """
    Yields (x_start, w) positions across 'width' using tile_w and overlap_pct.
    Ensures last window touches the right edge.
    """
    tile_w = max(1, int(tile_w))
    step = max(1, int(round(tile_w * (1.0 - float(overlap_pct) / 100.0))))
    if step <= 0:
        step = 1
    x = 0
    while x + tile_w < width:
        yield x, tile_w
        x += step
    # last window flush-right
    if width > 0:
        yield max(0, width - tile_w), tile_w

def _evaluate_left_right(result: J59JResult,
                         dx_range_left: Tuple[float, float],
                         dx_range_right: Tuple[float, float],
                         min_conf: float = 0.0,
                         draw_text: bool = True) -> Dict[str, Dict]:
    out = {}
    def _check(det: TileDetResult, rng: Tuple[float, float], tag: str):
        low, high = float(rng[0]), float(rng[1])
        p0 = _best_detection_xy_for_class(det, 0, min_conf=min_conf)
        p1 = _best_detection_xy_for_class(det, 1, min_conf=min_conf)
        classes_found = (p0 is not None) and (p1 is not None)
        dx = None; ok = False

        #debug print class
        print(f"Evaluating {tag} side: class 0 point = {p0}, class 1 point = {p1}, classes_found = {classes_found}")

        if classes_found:
            dx = float(p1[0] - p0[0])  # signed
            ok = (low <= dx <= high)
        color = (0, 255, 0) if ok else (0, 0, 255)
        _draw_border(result.annotated_bgr, det.roi_xywh, color=color, thickness=5)
        if draw_text:
            x, y, w, h = det.roi_xywh
            label = f"{tag}: dx={dx:.1f} in [{low:.1f},{high:.1f}]" if dx is not None else f"{tag}: MISSING"
            cv2.putText(result.annotated_bgr, label, (x + 6, max(25, y + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return {"dx": dx, "ok": ok, "classes_found": classes_found, "range": (low, high)}
    
    out["left"]  = _check(result.left,  DX_RANGE_LEFT,  "L")
    out["right"] = _check(result.right, DX_RANGE_RIGHT, "R")
    return out

def _evaluate_center_tile(model_center,
                          tile_bgr: np.ndarray,
                          tile_offset_xy: Tuple[int,int],
                          kpt_pair: Tuple[int,int],
                          dist_range: Tuple[float,float],
                          min_kpt_conf: float):
    (ox, oy) = tile_offset_xy

    c_results = _run_yolo_pose(model_center, tile_bgr)
    ck, cks, cb, cs, cc = _extract_pose_from_results(c_results, tile_offset_xy=(ox, oy))

    ok = False
    dist = None
    kxy_best = None
    ksc_best = None

    # Nothing predicted → return NG
    if ck is None or ck.shape[0] == 0:
        return {"ok": ok, "dist": dist, "kxy": kxy_best, "ksc": ksc_best}

    num_dets = ck.shape[0]
    # lengths for safety
    cks_len = 0 if cks is None else cks.shape[0]
    cs_len  = 0 if cs  is None else cs.shape[0]

    scores = []
    for i in range(num_dets):
        s = 0.0
        # prefer mean keypoint-conf if present and non-empty
        if cks is not None and i < cks_len:
            arr = cks[i]
            if hasattr(arr, "size") and arr.size > 0 and not np.isnan(arr).all():
                s = float(np.nanmean(arr))
        # otherwise fall back to box conf if present
        elif cs is not None and i < cs_len:
            s = float(cs[i])
        # else keep 0.0
        scores.append(s)

    idx_best = int(np.argmax(scores))
    kxy_best = ck[idx_best]                             # (K,2) absolute
    ksc_best = cks[idx_best] if (cks is not None and idx_best < cks_len) else None

    i0, i1 = int(kpt_pair[0]), int(kpt_pair[1])

    def _exists(j):
        if j < 0 or j >= kxy_best.shape[0]:
            return False
        if ksc_best is None:
            return True
        return (not np.isnan(ksc_best[j])) and (ksc_best[j] >= float(min_kpt_conf))

    if _exists(i0) and _exists(i1):
        x0, y0 = kxy_best[i0]
        x1, y1 = kxy_best[i1]
        dist = float(np.hypot(x1 - x0, y1 - y0))
        low, high = float(dist_range[0]), float(dist_range[1])
        ok = (low <= dist <= high)

    print(f"Center tile eval: best det idx={idx_best}, kpt_pair=({i0},{i1}), dist={dist}, ok={ok}")
    return {"ok": ok, "dist": dist, "kxy": kxy_best, "ksc": ksc_best}



def _scan_center_strip(annotated_bgr: np.ndarray,
                       proc: np.ndarray,
                       global_offset: Tuple[int,int],
                       model_center,
                       left_w: int,
                       right_w: int,
                       tile_w: int,
                       overlap_pct: float,
                       kpt_pair: Tuple[int,int],
                       dist_range: Tuple[float,float],
                       min_kpt_conf: float,
                       draw_ok_green: bool = True):
    """
    Slides across the middle strip (after removing fixed left/right tiles)
    with window width 'tile_w' and overlap 'overlap_pct'. Draws:
      - RED border for NG windows
      - GREEN border for OK windows if draw_ok_green=True (else only draw red)
      - Keypoints for the BEST detection in each window (absolute coords)
      - A line between kpt_pair (if both exist)

    Returns: list of {'roi_xywh': (x,y,w,h), 'ok': bool, 'dist': Optional[float]}
    """
    strip, (sx, sy, sw, sh) = _center_strip_after_lr(proc, left_w=left_w, right_w=right_w)
    results = []

    for local_x, w in _sliding_windows(sw, tile_w, overlap_pct):
        # tile in strip coords
        tile = strip[0:sh, local_x:local_x + w]
        if tile is None or tile.size == 0:
            continue

        # convert to full-image coords (absolute)
        gx = sx + local_x + global_offset[0]
        gy = sy + global_offset[1]
        roi = (gx, gy, w, sh)

        r = _evaluate_center_tile(
            model_center,
            tile_bgr=tile,
            tile_offset_xy=(gx, gy),
            kpt_pair=kpt_pair,
            dist_range=dist_range,
            min_kpt_conf=min_kpt_conf
        )
        r["roi_xywh"] = roi
        results.append(r)

        # choose color for this window
        color = (0, 255, 0) if r["ok"] else (0, 0, 255)

        # draw keypoints (if any)
        kxy = r.get("kxy", None)
        if kxy is not None:
            # draw each keypoint
            for (x, y) in kxy:
                cv2.circle(annotated_bgr, (int(x), int(y)), DRAW_RADIUS, color, -1, lineType=cv2.LINE_AA)

            # draw line between the pair if both indices are valid
            i0, i1 = int(kpt_pair[0]), int(kpt_pair[1])
            if 0 <= i0 < kxy.shape[0] and 0 <= i1 < kxy.shape[0]:
                x0, y0 = kxy[i0]
                x1, y1 = kxy[i1]
                cv2.line(annotated_bgr, (int(x0), int(y0)), (int(x1), int(y1)), color, 2, lineType=cv2.LINE_AA)

        # draw border (always draw NG, optionally draw OK)
        if r["ok"]:
            if draw_ok_green:
                cv2.rectangle(annotated_bgr, (gx, gy), (gx + w, gy + sh), color, 3, cv2.LINE_AA)
        else:
            cv2.rectangle(annotated_bgr, (gx, gy), (gx + w, gy + sh), color, 3, cv2.LINE_AA)

    return results

def J59J_Tape_Check(img_bgr: np.ndarray,
                    model_left,
                    model_right,
                    model_center,
                    return_annotated: bool = True):
    """
    - Top crop → side trim
    - LEFT & RIGHT fixed tiles: check signed dx ∈ DX_RANGE_LEFT/RIGHT
    - CENTER sliding windows across middle strip:
        each window checks kpt distance ∈ CENTER_DIST_RANGE
        NG windows get red border (OK windows green if CENTER_DRAW_OK_GREEN=True)
    Returns: (result: J59JResult, overall_ok: bool, center_windows: List[dict])
    """
    assert img_bgr is not None and img_bgr.ndim == 3

    # --- preproc
    proc = img_bgr.copy()
    if CROP_FROM_TOP:
        proc = _safe_crop_top(proc, CROP_HEIGHT_PIXELS)
    elif CROP_TOP_HALF:
        proc = proc[: max(1, proc.shape[0] // 2), :, :]

    proc, offset_x, _ = _apply_side_trim(proc, TRIM_LEFT, TRIM_RIGHT)
    global_offset = (offset_x, 0)

    # --- left/right fixed tiles
    tiles_lr = _make_left_right_tiles(proc,
                                      left_w=LEFTMOST_CUSTOM_WIDTH,
                                      right_w=RIGHTMOST_CUSTOM_WIDTH)

    # Run models L/R
    left_img, (lx, ly, lw, lh) = tiles_lr["left"]
    l_results = _run_yolo_pose(model_left, left_img)
    lk, lks, lb, ls, lc = _extract_pose_from_results(l_results, tile_offset_xy=(lx + global_offset[0], ly))
    left_det = TileDetResult("left", (lx + global_offset[0], ly, lw, lh), lk, lks, lb, ls, lc)

    right_img, (rx, ry, rw, rh) = tiles_lr["right"]
    r_results = _run_yolo_pose(model_right, right_img)
    rk, rks, rb, rs, rc = _extract_pose_from_results(r_results, tile_offset_xy=(rx + global_offset[0], ry))
    right_det = TileDetResult("right", (rx + global_offset[0], ry, rw, rh), rk, rks, rb, rs, rc)

    # --- annotate points/boxes (optional)
    annotated = img_bgr.copy()
    if return_annotated:
        _draw_tile_annotations(annotated, left_det,  color=(0, 255, 0))
        _draw_tile_annotations(annotated, right_det, color=(255, 0, 0))

    # --- wrap result base (no center stored here; windows annotated directly)
    result = J59JResult(annotated_bgr=annotated, left=left_det, right=right_det)

    # --- evaluate left/right (signed dx)
    eval_lr = _evaluate_left_right(
        result,
        dx_range_left=DX_RANGE_LEFT,
        dx_range_right=DX_RANGE_RIGHT,
        min_conf=YOLO_CONF,
        draw_text=True
    )

    # --- sliding center across middle strip (height = CROP_HEIGHT_PIXELS)
    center_windows = _scan_center_strip(
        annotated_bgr=result.annotated_bgr,
        proc=proc,
        global_offset=global_offset,
        model_center=model_center,
        left_w=LEFTMOST_CUSTOM_WIDTH,
        right_w=RIGHTMOST_CUSTOM_WIDTH,
        tile_w=CENTER_TILE_WIDTH,
        overlap_pct=CENTER_OVERLAP_PCT,
        kpt_pair=CENTER_KPT_PAIR,
        dist_range=CENTER_DIST_RANGE,
        min_kpt_conf=CENTER_MIN_KPT_CONF,
        draw_ok_green=CENTER_DRAW_OK_GREEN
    )

    # overall ok: left ok AND right ok AND all center windows ok
    centers_ok = all(w["ok"] for w in center_windows) if center_windows else False  # require at least one window
    overall_ok = bool(eval_lr["left"]["ok"] and eval_lr["right"]["ok"] and centers_ok)
    return result.annotated_bgr, overall_ok, center_windows


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
    cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, thickness, lineType=cv2.LINE_AA)

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