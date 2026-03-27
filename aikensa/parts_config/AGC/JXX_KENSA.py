from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Any

# =========================
# 0) Global debug switch
# =========================
# If True, all OK results are forced to True (L/R and center); overall_ok becomes True.
DEBUG_MODE = True

# =========================
# 1) Defaults / Config
# =========================

# Vertical crop
CROP_TOP_HALF      = True      # used only if both FROM_* are False
CROP_FROM_TOP      = False     # crop a band from the TOP
CROP_FROM_BOTTOM   = True      # crop a band from the BOTTOM (forces CROP_FROM_TOP=False)
CROP_HEIGHT_PIXELS = 80

# Side trims
TRIM_LEFT  = 0
TRIM_RIGHT = 0

# Fixed L/R tile widths + center windows
LEFTMOST_CUSTOM_WIDTH  = 128
RIGHTMOST_CUSTOM_WIDTH = 128
CENTER_TILE_WIDTH      = 128
CENTER_OVERLAP_PCT     = 0

# Signed Δx OK ranges for L/R
DX_RANGE_LEFT  = (-15.0, +15.0)
DX_RANGE_RIGHT = (-15.0, +15.0)

# Runtime / draw
YOLO_CONF   = 0.20
YOLO_IOU    = 0.50
DRAW_RADIUS = 2
DRAW_THICK  = 2

# Center (classification) evaluation
CENTER_OK_CLASS            = 0
CENTER_PAD                 = (0, 0, 0, 0)   # (top, bottom, left, right) pad for center tiles

# Per-side padding for keypoint models
LEFT_PAD  = (0, 0, 0, 0)    # (top, bottom, left, right)
RIGHT_PAD = (0, 0, 0, 0)

# =========================
# 2) Data structures
# =========================

@dataclass
class TileDetResult:
    tile_name: str
    roi_xywh: Tuple[int, int, int, int]         # (x, y, w, h) in ORIGINAL (global) coords
    kpts: Optional[np.ndarray]                  # (N,K,2) global coords (after unpad)
    kpts_scores: Optional[np.ndarray]           # (N,K) or None
    boxes_xyxy: Optional[np.ndarray]            # (N,4) global coords (after unpad)
    scores: Optional[np.ndarray]                # (N,) or None
    classes: Optional[np.ndarray]               # (N,) or None

@dataclass
class JXXResult:
    annotated_bgr: np.ndarray
    left: TileDetResult
    right: TileDetResult

# =========================
# 3) Image helpers
# =========================

def _crop_vertical(img: np.ndarray,
                   crop_from_top: bool,
                   crop_from_bottom: bool,
                   crop_height: Optional[int],
                   top_half_fallback: bool) -> Tuple[np.ndarray, int]:
    """
    Returns (cropped_img, offset_y_in_original).
    - If crop_from_top: take top 'crop_height' rows. offset_y = 0.
    - If crop_from_bottom: take bottom 'crop_height' rows. offset_y = H - crop_height.
    - Else if top_half_fallback: take top half. offset_y = 0.
    - Else: return original. offset_y = 0.
    """
    H = img.shape[0]
    if crop_height is None or crop_height <= 0 or crop_height >= H:
        return img, 0

    # enforce exclusivity
    if crop_from_bottom:
        crop_from_top = False

    if crop_from_top:
        return img[:crop_height, :, :], 0

    if crop_from_bottom:
        y0 = max(0, H - crop_height)
        return img[y0:, :, :], y0

    if top_half_fallback:
        return img[: max(1, H // 2), :, :], 0

    return img, 0


def _apply_side_trim(img: np.ndarray, trim_left: int, trim_right: int) -> Tuple[np.ndarray, int, int]:
    """Return (trimmed_img, offset_x, new_width)."""
    h, w = img.shape[:2]
    x0 = max(0, int(trim_left))
    x1 = max(x0, w - int(trim_right))
    return img[:, x0:x1], x0, (x1 - x0)


def _make_left_right_tiles(img: np.ndarray,
                           left_w: int,
                           right_w: int) -> Dict[str, Tuple[np.ndarray, Tuple[int,int,int,int]]]:
    """Return dict: name -> (tile_img, (x,y,w,h)) in the trimmed image coords."""
    H, W = img.shape[:2]
    lw = int(max(1, min(left_w, W)))
    rw = int(max(1, min(right_w, W)))

    if lw + rw > W:
        total = lw + rw
        lw = int(lw * (W / total))
        rw = max(1, W - lw)

    left_roi  = (0, 0, lw, H)
    right_roi = (W - rw, 0, rw, H)

    lx, ly, lw, lh = left_roi
    rx, ry, rw, rh = right_roi

    left_img  = img[ly:ly+lh, lx:lx+lw]
    right_img = img[ry:ry+rh, rx:rx+rw]
    return {"left": (left_img, left_roi), "right": (right_img, right_roi)}


def _center_strip_after_lr(proc: np.ndarray, left_w: int, right_w: int) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    """Return (center_img, (x,y,w,h)) for strip after removing fixed L/R widths."""
    H, W = proc.shape[:2]
    x0 = int(left_w)
    x1 = int(W - right_w)
    x0 = max(0, min(x0, W))
    x1 = max(x0, min(x1, W))
    strip = proc[0:H, x0:x1]
    return strip, (x0, 0, x1 - x0, H)


def _sliding_windows(width: int, tile_w: int, overlap_pct: float):
    """Yield (x_start, w) across width with tile_w & overlap_pct."""
    tile_w = max(1, int(tile_w))
    step = max(1, int(round(tile_w * (1.0 - float(overlap_pct) / 100.0))))
    x = 0
    while x + tile_w < width:
        yield x, tile_w
        x += step
    if width > 0:
        yield max(0, width - tile_w), tile_w


def _draw_border(img_bgr: np.ndarray, roi_xywh: Tuple[int, int, int, int],
                 color: Tuple[int, int, int], thickness: int = 4) -> None:
    x, y, w, h = roi_xywh
    cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, thickness, lineType=cv2.LINE_AA)


def _draw_tile_annotations(img_bgr: np.ndarray,
                           det: TileDetResult,
                           color=(0, 255, 0)) -> None:
    """Draw only keypoints by default (boxes optional)."""
    if det.kpts is not None:
        for n in range(det.kpts.shape[0]):
            for (x, y) in det.kpts[n]:
                cv2.circle(img_bgr, (int(x), int(y)), DRAW_RADIUS, color, -1, lineType=cv2.LINE_AA)
    # If you want boxes too, uncomment:
    # if det.boxes_xyxy is not None:
    #     for (x1, y1, x2, y2) in det.boxes_xyxy:
    #         cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, DRAW_THICK, lineType=cv2.LINE_AA)

# =========================
# 4) Model I/O helpers
# =========================

def _run_yolo_pose(model: Any, tile_bgr: np.ndarray, conf: float, iou: float, imgsz: int = 384, max_det: int = 2) -> Any:
    """Run Ultralytics model on a single np.ndarray tile (BGR is fine)."""
    if tile_bgr is None or tile_bgr.size == 0:
        print("⚠️ Warning: Empty tile passed to YOLO inference.")
        return None
    return model.predict(source=tile_bgr, conf=conf, iou=iou, verbose=False, save=False, imgsz=imgsz, max_det=max_det)


def _run_yolo_classify(model: Any, tile_bgr: np.ndarray, imgsz: int = 256) -> Any:
    """Run Ultralytics classification on a single np.ndarray tile."""
    if tile_bgr is None or tile_bgr.size == 0:
        print("⚠️ Warning: Empty tile passed to YOLO classification.")
        return None
    return model.predict(source=tile_bgr, verbose=False, save=False, imgsz=imgsz)


def _pad_image(img: np.ndarray, pad: Tuple[int,int,int,int],
               value: Tuple[int,int,int]=(0,0,0)) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    """Pad BGR image by (top, bottom, left, right)."""
    pt, pb, pl, pr = [int(x) for x in pad]
    padded = cv2.copyMakeBorder(img, pt, pb, pl, pr, borderType=cv2.BORDER_CONSTANT, value=value)
    return padded, (pt, pb, pl, pr)


def _run_yolo_pose_with_pad(model: Any,
                            tile_bgr: np.ndarray,
                            conf: float,
                            iou: float,
                            pad: Tuple[int,int,int,int],
                            max_det: int = 2,
                            imgsz: int = 384) -> Tuple[Any, Tuple[int,int,int,int]]:
    """
    Pads tile, runs model, returns (results, (pt,pb,pl,pr)).
    """
    padded, pad_tuple = _pad_image(tile_bgr, pad)
    results = _run_yolo_pose(model, padded, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det)
    return results, pad_tuple


def _extract_pose_from_results_with_unpad(
    results: Any,
    tile_offset_xy: Tuple[int, int],     # (ox, oy) of the UNPADDED tile in GLOBAL/original coords
    pad_tuple: Tuple[int, int, int, int],# (pt, pb, pl, pr) that was applied to the tile
    tile_shape: Tuple[int, int, int],    # shape of the UNPADDED tile (H, W, C)
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Convert YOLO *padded-tile* predictions to GLOBAL coordinates as if no pad existed:
      - Subtract left/top padding (pl, pt)
      - Add global tile offset (ox, oy)
      - Clamp inside the unpadded tile bounds in global coords
    Returns (kpts_xy, kpts_scores, boxes_xyxy, scores, classes) in GLOBAL coords.
    """
    if not results or len(results) == 0:
        return None, None, None, None, None

    ox, oy = tile_offset_xy
    pt, pb, pl, pr = pad_tuple
    H, W = tile_shape[:2]

    r0 = results[0]

    # Boxes
    boxes_xyxy = scores = classes = None
    if getattr(r0, "boxes", None) is not None and r0.boxes is not None:
        try:
            boxes_xyxy = r0.boxes.xyxy.cpu().numpy().astype(float)
            scores     = r0.boxes.conf.cpu().numpy()
            classes    = r0.boxes.cls.cpu().numpy()
        except Exception:
            boxes_xyxy = np.asarray(r0.boxes.xyxy, dtype=float)
            scores     = np.asarray(r0.boxes.conf)
            classes    = np.asarray(r0.boxes.cls)

        if boxes_xyxy is not None:
            # unpad to tile coords, then add global offset
            boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]] - pl + ox
            boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]] - pt + oy
            # clamp to unpadded-tile bounds (in global)
            x_min, x_max = ox, ox + W
            y_min, y_max = oy, oy + H
            boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], x_min, x_max)
            boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], y_min, y_max)

    # Keypoints
    kpts_xy = kpts_scores = None
    if getattr(r0, "keypoints", None) is not None and r0.keypoints is not None:
        try:
            kpts_xy = r0.keypoints.xy.cpu().numpy().astype(float)  # (N,K,2)
            kpts_scores = r0.keypoints.conf.cpu().numpy()
        except Exception:
            kpts_xy = np.asarray(r0.keypoints.xy, dtype=float)
            kpts_scores = getattr(r0.keypoints, "conf", None)
            if kpts_scores is not None:
                kpts_scores = np.asarray(kpts_scores)

        if kpts_xy is not None:
            kpts_xy[..., 0] = kpts_xy[..., 0] - pl + ox
            kpts_xy[..., 1] = kpts_xy[..., 1] - pt + oy
            x_min, x_max = ox, ox + W
            y_min, y_max = oy, oy + H
            kpts_xy[..., 0] = np.clip(kpts_xy[..., 0], x_min, x_max)
            kpts_xy[..., 1] = np.clip(kpts_xy[..., 1], y_min, y_max)

    return kpts_xy, kpts_scores, boxes_xyxy, scores, classes

# =========================
# 5) L/R evaluation (Δx between best cls0 & cls1 keypoint means)
# =========================

def _best_detection_xy_for_class(det: TileDetResult,
                                 target_cls: int,
                                 min_conf: float = 0.0) -> Optional[Tuple[float, float]]:
    """Return representative (x,y) for best detection of class, using mean keypoint location."""
    if det.kpts is None or det.classes is None:
        return None

    cls_arr = det.classes.astype(int)
    idxs = np.where(cls_arr == int(target_cls))[0]
    if idxs.size == 0:
        return None

    # Prefer mean keypoint conf; fallback to box conf; else 0
    scores = []
    for i in idxs:
        if det.kpts_scores is not None:
            scores.append(float(np.nanmean(det.kpts_scores[i])))
        elif det.scores is not None:
            scores.append(float(det.scores[i]))
        else:
            scores.append(0.0)
    scores = np.asarray(scores, dtype=float)

    valid = np.where(scores >= min_conf)[0]
    if valid.size == 0:
        return None

    best_local = valid[np.argmax(scores[valid])]
    best_idx = idxs[best_local]
    kxy = det.kpts[best_idx]  # (K,2)
    x = float(np.mean(kxy[:, 0]))
    y = float(np.mean(kxy[:, 1]))
    return (x, y)


def evaluate_sides_and_annotate(result: JXXResult,
                                dx_range_left: Tuple[float, float],
                                dx_range_right: Tuple[float, float],
                                min_conf: float = 0.0,
                                draw_text: bool = True,
                                debug_mode: bool = DEBUG_MODE) -> Dict[str, Dict[str, Any]]:
    """
    For each side:
      - Compute signed Δx = x(cls1) - x(cls0) between best class-1 and class-0.
      - OK if Δx in the provided range. If debug_mode=True, force OK=True.
      - Draw green border if OK, red if NG (debug forces green). Optionally draw labels.
    """
    out: Dict[str, Dict[str, Any]] = {}

    def _check(det: TileDetResult, rng: Tuple[float, float], tag: str):
        low, high = float(rng[0]), float(rng[1])
        p0 = _best_detection_xy_for_class(det, 0, min_conf=min_conf)
        p1 = _best_detection_xy_for_class(det, 1, min_conf=min_conf)
        classes_found = (p0 is not None) and (p1 is not None)

        dx = None
        ok = False
        if classes_found:
            dx = float(p1[0] - p0[0])
            ok = (low <= dx <= high)

        # DEBUG override
        if debug_mode:
            ok = True

        color = (0, 255, 0) if ok else (0, 0, 255)
        _draw_border(result.annotated_bgr, det.roi_xywh, color=color, thickness=5)

        if draw_text:
            x, y, w, h = det.roi_xywh
            if dx is not None:
                label = f"{tag} OK" if ok else f"{tag} NG"
                # label = f"OK" if ok else f"NG"
            else:
                label = f"{tag}: MISSING" if not debug_mode else f"{tag} OK"
                # label = f"MISSING" if not debug_mode else f"OK"
            cv2.putText(result.annotated_bgr, label, (x + 9, max(50, y + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        return {"dx": dx, "ok": ok, "classes_found": classes_found, "range": (low, high)}

    out["left"]  = _check(result.left,  dx_range_left,  "LEFT")
    out["right"] = _check(result.right, dx_range_right, "RIGHT")
    return out

# =========================
# 6) Center scanning with padding (classification)
# =========================

def _center_class_label(predicted_class: Optional[int]) -> str:
    if predicted_class == 0:
        return "OK"
    if predicted_class == 1:
        return "NG"
    if predicted_class == 2:
        return "NO PART"
    return "UNKNOWN"


def _evaluate_center_tile_classification(model_center: Any,
                                         tile_bgr: np.ndarray,
                                         tile_offset_xy: Tuple[int, int],
                                         *,
                                         ok_class: int,
                                         pad: Tuple[int, int, int, int] = (0, 0, 0, 0),
                                         annotated_bgr: Optional[np.ndarray] = None,
                                         draw: bool = True,
                                         debug_mode: bool = DEBUG_MODE,
                                         yolo_imgsz: int = 256) -> Dict[str, Any]:
    """
    Pads each center tile and runs Ultralytics classification.
    A tile passes only when the top-1 predicted class matches ok_class.
    Returns dict: {'ok', 'score', 'cls', 'label'}
    """
    ox, oy = tile_offset_xy
    padded, _ = _pad_image(tile_bgr, pad)
    results = _run_yolo_classify(model_center, padded, imgsz=yolo_imgsz)

    predicted_class = None
    confidence = None
    if results and len(results) > 0 and getattr(results[0], "probs", None) is not None:
        probs = results[0].probs
        try:
            predicted_class = int(probs.top1)
        except Exception:
            try:
                predicted_class = int(np.asarray(probs.data).argmax())
            except Exception:
                predicted_class = None

        try:
            confidence = float(probs.top1conf)
        except Exception:
            try:
                confidence = float(np.asarray(probs.data)[predicted_class]) if predicted_class is not None else None
            except Exception:
                confidence = None

    ok = predicted_class == int(ok_class)
    if debug_mode:
        ok = True

    h, w = tile_bgr.shape[:2]
    gx1, gy1 = ox, oy
    gx2, gy2 = ox + w, oy + h
    label = _center_class_label(predicted_class)

    if draw and annotated_bgr is not None:
        color = (0, 255, 0) if ok else (0, 0, 255)
        cv2.rectangle(annotated_bgr, (int(round(gx1)), int(round(gy1))),
                                      (int(round(gx2)), int(round(gy2))),
                                      color, 1, lineType=cv2.LINE_AA)
        cv2.putText(annotated_bgr, label,
                    (int(round(gx1)) + 6, max(18, int(round(gy1)) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    return {
        "ok": bool(ok),
        "score": confidence,
        "cls": predicted_class,
        "label": label,
    }


def _scan_center_strip(annotated_bgr: np.ndarray,
                       proc: np.ndarray,
                       global_offset: Tuple[int,int],
                       model_center: Any,
                       left_w: int,
                       right_w: int,
                       tile_w: int,
                       overlap_pct: float,
                       *,
                                             ok_class: int,
                       pad: Tuple[int,int,int,int]=(0,0,0,0),
                       draw_ok_green: bool = True,
                       debug_mode: bool = DEBUG_MODE,
                       yolo_imgsz: int = 384
                       ) -> List[Dict[str, Any]]:
    """
        Slides across middle strip; each window is classified independently.
        Returns list per window: {'roi_xywh', 'ok', 'score', 'cls', 'label'}
    """
    strip, (sx, sy, sw, sh) = _center_strip_after_lr(proc, left_w=left_w, right_w=right_w)
    results: List[Dict[str, Any]] = []

    for local_x, w in _sliding_windows(sw, tile_w, overlap_pct):
        tile = strip[0:sh, local_x:local_x + w]
        if tile is None or tile.size == 0:
            continue

        gx = sx + local_x + global_offset[0]
        gy = sy + global_offset[1]
        roi = (gx, gy, w, sh)

        r = _evaluate_center_tile_classification(
            model_center,
            tile_bgr=tile,
            tile_offset_xy=(gx, gy),
            ok_class=ok_class,
            pad=pad,
            annotated_bgr=annotated_bgr,
            draw=True,
            debug_mode=debug_mode,
            yolo_imgsz=yolo_imgsz
        )
        r["roi_xywh"] = roi
        results.append(r)

        color = (0, 255, 0) if r["ok"] else (0, 0, 255)
        if r["ok"]:
            if draw_ok_green:
                cv2.rectangle(annotated_bgr, (gx, gy), (gx + w, gy + sh), color, 1, lineType=cv2.LINE_AA)
        else:
            cv2.rectangle(annotated_bgr, (gx, gy), (gx + w, gy + sh), color, 1, lineType=cv2.LINE_AA)

    return results

# =========================
# 7) Unified entry point
# =========================

def JXX_Check(img_bgr: np.ndarray,
              model_left: Any,
              model_right: Any,
              model_center: Any = None,                 # optional; needed if enable_center=True
              return_annotated: bool = True,
              *,
              # --- geometry / preproc ---
              crop_from_top: bool = CROP_FROM_TOP,
              crop_from_bottom: bool = CROP_FROM_BOTTOM,
              crop_height: int = CROP_HEIGHT_PIXELS,
              crop_top_half_fallback: bool = CROP_TOP_HALF,
              trim_left: int = TRIM_LEFT,
              trim_right: int = TRIM_RIGHT,
              left_width: int = LEFTMOST_CUSTOM_WIDTH,
              right_width: int = RIGHTMOST_CUSTOM_WIDTH,
              # --- thresholds / NMS ---
              yolo_conf: float = YOLO_CONF,
              yolo_iou: float = YOLO_IOU,
              yolo_imgsz_side: int = 384,
              yolo_imgsz_center: int = 256,
              # --- L/R evaluation ---
              dx_range_left: Tuple[float, float] = DX_RANGE_LEFT,
              dx_range_right: Tuple[float, float] = DX_RANGE_RIGHT,
              left_pad: Tuple[int,int,int,int] = LEFT_PAD,      # (top,bottom,left,right)
              right_pad: Tuple[int,int,int,int] = RIGHT_PAD,
              # --- center scanning (classification) ---
              enable_center: bool = False,
              center_tile_width: int = CENTER_TILE_WIDTH,
              center_overlap_pct: float = CENTER_OVERLAP_PCT,
              center_ok_class: int = CENTER_OK_CLASS,
              center_pad: Tuple[int,int,int,int] = CENTER_PAD,
              center_draw_ok_green: bool = True,
              # --- debug override (optional) ---
              debug_mode: Optional[bool] = None
              ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    """
    Unified pipeline:
      - Vertical crop (top/bottom exclusive) + side trim → global offset
      - Left/Right tiles with *keypoint* YOLO + per-side padding (coords unpadded & mapped to global)
      - L/R evaluation via signed Δx between best cls0 & cls1 keypoint-mean positions
    - Optional center scanning using classification windows where every tile must predict the configured OK class

    DEBUG:
      - If DEBUG_MODE or debug_mode=True, all OK results are forced to True and overall_ok=True.

    Returns:
      annotated_bgr (np.ndarray), overall_ok (bool), details (dict)
        details = {
            'eval_lr': <dict>,
            'center_windows': <list of dicts>  # empty if center disabled
        }
    """
    assert img_bgr is not None and img_bgr.ndim == 3, "img_bgr must be HxWx3 BGR np.ndarray"

    # decide debug flag
    dbg = DEBUG_MODE if debug_mode is None else bool(debug_mode)

    # 1) Vertical crop
    proc, offset_y = _crop_vertical(
        img_bgr,
        crop_from_top=crop_from_top,
        crop_from_bottom=crop_from_bottom,
        crop_height=crop_height,
        top_half_fallback=crop_top_half_fallback
    )

    # 2) Side trim
    proc, offset_x, _ = _apply_side_trim(proc, trim_left, trim_right)
    global_offset = (offset_x, offset_y)

    # 3) L/R tiles
    tiles_lr = _make_left_right_tiles(proc, left_w=left_width, right_w=right_width)

    # --- LEFT (pad -> infer -> unpad coords -> global) ---
    left_img, (lx, ly, lw, lh) = tiles_lr["left"]
    l_results, lpad = _run_yolo_pose_with_pad(model_left, left_img, conf=yolo_conf, iou=yolo_iou, pad=left_pad, imgsz = yolo_imgsz_side, max_det=2)
    lk, lks, lb, ls, lc = _extract_pose_from_results_with_unpad(
        l_results,
        tile_offset_xy=(lx + global_offset[0], ly + global_offset[1]),
        pad_tuple=lpad,
        tile_shape=left_img.shape,
    )
    left_det = TileDetResult("left",
                             (lx + global_offset[0], ly + global_offset[1], lw, lh),
                             lk, lks, lb, ls, lc)

    # --- RIGHT (pad -> infer -> unpad coords -> global) ---
    right_img, (rx, ry, rw, rh) = tiles_lr["right"]
    r_results, rpad = _run_yolo_pose_with_pad(model_right, right_img, conf=yolo_conf, iou=yolo_iou, pad=right_pad, imgsz = yolo_imgsz_side, max_det=2)
    rk, rks, rb, rs, rc = _extract_pose_from_results_with_unpad(
        r_results,
        tile_offset_xy=(rx + global_offset[0], ry + global_offset[1]),
        pad_tuple=rpad,
        tile_shape=right_img.shape,
    )
    right_det = TileDetResult("right",
                              (rx + global_offset[0], ry + global_offset[1], rw, rh),
                              rk, rks, rb, rs, rc)

    # 4) Annotation base
    annotated = img_bgr.copy()
    if return_annotated:
        _draw_tile_annotations(annotated, left_det,  color=(0, 255, 0))
        _draw_tile_annotations(annotated, right_det, color=(255, 0, 0))
        # debug borders
        for (x, y, w, h), col in [(left_det.roi_xywh, (0, 255, 0)), (right_det.roi_xywh, (255, 0, 0))]:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), col, 1, lineType=cv2.LINE_AA)

    result = JXXResult(annotated_bgr=annotated, left=left_det, right=right_det)

    # 5) L/R evaluate (respect debug)
    eval_lr = evaluate_sides_and_annotate(
        result,
        dx_range_left=dx_range_left,
        dx_range_right=dx_range_right,
        min_conf=yolo_conf,
        draw_text=True,
        debug_mode=dbg
    )
    print(f"Evaluation L/R: {eval_lr}")
    # 6) Center scanning (optional; respect debug)
    center_windows: List[Dict[str, Any]] = []
    if enable_center:
        assert model_center is not None, "enable_center=True requires model_center."
        center_windows = _scan_center_strip(
            annotated_bgr=result.annotated_bgr,
            proc=proc,
            global_offset=global_offset,
            model_center=model_center,
            left_w=left_width,
            right_w=right_width,
            tile_w=center_tile_width,
            overlap_pct=center_overlap_pct,
            ok_class=center_ok_class,
            pad=center_pad,
            draw_ok_green=center_draw_ok_green,
            debug_mode=dbg,
            yolo_imgsz = yolo_imgsz_center
        )

    # 7) Overall OK (respect debug)
    if dbg:
        overall_ok = True
    else:
        lr_ok = bool(eval_lr["left"]["ok"] and eval_lr["right"]["ok"])
        centers_ok = all(w.get("ok", False) for w in center_windows) if center_windows else (not enable_center)
        overall_ok = lr_ok and centers_ok

    details = {"eval_lr": eval_lr, "center_windows": center_windows, "debug_mode": dbg}
    return result.annotated_bgr, overall_ok, details
