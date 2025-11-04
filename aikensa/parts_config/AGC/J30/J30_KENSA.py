from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

# =========================
# 1) Crop config (defaults)
# =========================
CROP_TOP_HALF      = True    # used only if both FROM_* are False
CROP_FROM_TOP      = True
CROP_FROM_BOTTOM   = False   # mutually exclusive with CROP_FROM_TOP
CROP_HEIGHT_PIXELS = 64

# =========================
# 2) Side trim in pixels (defaults)
# =========================
TRIM_LEFT  = 60
TRIM_RIGHT = 120

# =========================
# 3) Split config (defaults)
# =========================
LEFTMOST_CUSTOM_WIDTH  = 128
RIGHTMOST_CUSTOM_WIDTH = 128

# =========================
# 4) Signed dx ranges (defaults)
# =========================
DX_RANGE_LEFT  = (-5.0, +5.0)
DX_RANGE_RIGHT = (-5.0, +5.0)

# =========================
# Runtime parameters (defaults)
# =========================
YOLO_CONF  = 0.20
YOLO_IOU   = 0.50
DRAW_RADIUS = 3
DRAW_THICK  = 2

# =========================
# Center sliding-window config (defaults)
# =========================
CENTER_CROP_WIDTH    = 128                 # not strictly needed; height is proc height
CENTER_CROP_HEIGHT   = CROP_HEIGHT_PIXELS  # kept for reference
CENTER_TILE_WIDTH    = 128
CENTER_OVERLAP_PCT   = 0
CENTER_KPT_PAIR      = (0, 1)              # kept for signature parity (unused in new center eval)
CENTER_DIST_RANGE    = (20.0, 60.0)        # vertical distance OK range
CENTER_MIN_KPT_CONF  = 0.25
CENTER_DRAW_OK_GREEN = True


@dataclass
class TileDetResult:
    tile_name: str
    roi_xywh: Tuple[int, int, int, int]  # (x, y, w, h) in ORIGINAL full image coords
    kpts: Optional[np.ndarray]
    kpts_scores: Optional[np.ndarray]
    boxes_xyxy: Optional[np.ndarray]
    scores: Optional[np.ndarray]
    classes: Optional[np.ndarray]


@dataclass
class J30Result:
    annotated_bgr: np.ndarray
    left: TileDetResult
    right: TileDetResult


def _crop_vertical(img: np.ndarray,
                   crop_from_top: bool,
                   crop_from_bottom: bool,
                   crop_height: Optional[int],
                   top_half_fallback: bool) -> Tuple[np.ndarray, int]:
    """
    Returns (cropped_img, offset_y_in_original).
    - If crop_from_top:     keep top 'crop_height' rows, offset_y=0
    - If crop_from_bottom:  keep bottom 'crop_height' rows, offset_y=H - crop_height
    - Else if top_half_fallback: keep top half, offset_y=0
    - Else: return original, offset_y=0
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
    h, w = img.shape[:2]
    x0 = max(0, int(trim_left))
    x1 = max(x0, w - int(trim_right))
    return img[:, x0:x1], x0, x1 - x0  # trimmed image, offset_x, new_width


def _make_left_right_tiles(img: np.ndarray,
                           left_w: int,
                           right_w: int) -> Dict[str, Tuple[np.ndarray, Tuple[int,int,int,int]]]:
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


def _run_yolo_pose(model, tile_bgr: np.ndarray, conf: float, iou: float):
    """Run YOLO model safely on a BGR tile."""
    if tile_bgr is None or tile_bgr.size == 0:
        print("⚠️ Warning: Empty tile passed to YOLO inference.")
        return None
    rgb = tile_bgr
    return model.predict(source=rgb, conf=conf, verbose=False, save=False, imgsz=256)


def _extract_pose_from_results(results, tile_offset_xy: Tuple[int, int]) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
]:
    ox, oy = tile_offset_xy
    if not results or len(results) == 0:
        return None, None, None, None, None

    r0 = results[0]
    kpts_xy = None
    kpts_scores = None
    if getattr(r0, "keypoints", None) is not None and r0.keypoints is not None:
        try:
            kpts_xy = r0.keypoints.xy.cpu().numpy()
        except Exception:
            kpts_xy = np.asarray(r0.keypoints.xy)
        try:
            kpts_scores = r0.keypoints.conf.cpu().numpy()
        except Exception:
            kpts_scores = None
        if kpts_xy is not None:
            kpts_xy[..., 0] += ox
            kpts_xy[..., 1] += oy

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


def _draw_tile_annotations(img_bgr: np.ndarray, det: TileDetResult, color=(0, 255, 0)) -> None:
    if det.kpts is not None:
        for n in range(det.kpts.shape[0]):
            for (x, y) in det.kpts[n]:
                cv2.circle(img_bgr, (int(x), int(y)), DRAW_RADIUS, color, -1, lineType=cv2.LINE_AA)
    if det.boxes_xyxy is not None:
        for (x1, y1, x2, y2) in det.boxes_xyxy:
            cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, DRAW_THICK, lineType=cv2.LINE_AA)


def _center_strip_after_lr(proc: np.ndarray, left_w: int, right_w: int):
    """Return (center_img, (x,y,w,h)) for middle strip after fixed L/R widths."""
    H, W = proc.shape[:2]
    x0 = int(left_w)
    x1 = int(W - right_w)
    x0 = max(0, min(x0, W))
    x1 = max(x0, min(x1, W))
    strip = proc[0:H, x0:x1]
    return strip, (x0, 0, x1 - x0, H)


def _sliding_windows(width: int, tile_w: int, overlap_pct: float):
    """Yield (x_start, w) across width with given tile width & overlap."""
    tile_w = max(1, int(tile_w))
    step = max(1, int(round(tile_w * (1.0 - float(overlap_pct) / 100.0))))
    x = 0
    while x + tile_w < width:
        yield x, tile_w
        x += step
    if width > 0:
        yield max(0, width - tile_w), tile_w


def _evaluate_left_right(result: J30Result,
                         dx_range_left: Tuple[float, float],
                         dx_range_right: Tuple[float, float],
                         min_conf: float = 0.0,
                         draw_text: bool = True) -> Dict[str, Dict]:
    out = {}

    def _best_detection_xy_for_class(det: TileDetResult,
                                     target_cls: int,
                                     min_conf: float = 0.0) -> Optional[Tuple[float, float]]:
        if det.kpts is None or det.classes is None:
            return None
        cls_arr = det.classes.astype(int)
        idxs = np.where(cls_arr == int(target_cls))[0]
        if idxs.size == 0:
            return None

        scores = []
        for i in idxs:
            if det.kpts_scores is not None:
                scores.append(float(np.nanmean(det.kpts_scores[i])))
            elif det.scores is not None:
                scores.append(float(det.scores[i]))
            else:
                scores.append(0.0)
        scores = np.array(scores)
        valid = np.where(scores >= min_conf)[0]
        if valid.size == 0:
            return None

        best_local = valid[np.argmax(scores[valid])]
        best_idx = idxs[best_local]
        kpts_xy = det.kpts[best_idx]
        x = float(np.mean(kpts_xy[:, 0]))
        y = float(np.mean(kpts_xy[:, 1]))
        return (x, y)

    def _draw_border(img_bgr: np.ndarray, roi_xywh: Tuple[int, int, int, int],
                     color: Tuple[int, int, int], thickness: int = 4) -> None:
        x, y, w, h = roi_xywh
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, thickness, lineType=cv2.LINE_AA)

    def _check(det: TileDetResult, rng: Tuple[float, float], tag: str):
        low, high = float(rng[0]), float(rng[1])
        p0 = _best_detection_xy_for_class(det, 0, min_conf=min_conf)
        p1 = _best_detection_xy_for_class(det, 1, min_conf=min_conf)
        classes_found = (p0 is not None) and (p1 is not None)
        dx = None; ok = False

        print(f"Evaluating {tag} side: class 0 point = {p0}, class 1 point = {p1}, classes_found = {classes_found}")

        if classes_found:
            dx = float(p1[0] - p0[0])
            ok = (low <= dx <= high)

        color = (0, 255, 0) if ok else (0, 0, 255)
        _draw_border(result.annotated_bgr, det.roi_xywh, color=color, thickness=5)
        if draw_text:
            x, y, w, h = det.roi_xywh
            label = f"{tag}: dx={dx:.1f} in [{low:.1f},{high:.1f}]" if dx is not None else f"{tag}: MISSING"
            cv2.putText(result.annotated_bgr, label, (x + 6, max(25, y + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return {"dx": dx, "ok": ok, "classes_found": classes_found, "range": (low, high)}

    out["left"]  = _check(result.left,  dx_range_left,  "L")
    out["right"] = _check(result.right, dx_range_right, "R")
    return out


def _evaluate_center_tile(model_center,
                          tile_bgr: np.ndarray,
                          tile_offset_xy: Tuple[int,int],
                          kpt_pair: Tuple[int,int],             # kept for API parity (unused here)
                          dist_range: Tuple[float,float],
                          min_kpt_conf: float,
                          conf: float,
                          iou: float,
                          *,
                          annotated_bgr: Optional[np.ndarray] = None,
                          draw: bool = True):
    """
    New logic:
      - Find best class-0 and best class-1 detections (highest score per class).
      - Score = mean keypoint conf if available, else box conf.
      - Use mean of all keypoints from each detection as its representative point.
      - Measure VERTICAL distance:  dy = |y1 - y0|  (signed_dy = y1 - y0).
      - OK iff dist_range[0] <= dy <= dist_range[1].
      - Optionally draw det0 (green), det1 (blue), and a line between them; put dy text.
    Returns: dict with ok, dy, signed_dy, p0, p1, idx0, idx1.
    """
    (ox, oy) = tile_offset_xy
    c_results = _run_yolo_pose(model_center, tile_bgr, conf=conf, iou=iou)
    ck, cks, cb, cs, cc = _extract_pose_from_results(c_results, tile_offset_xy=(ox, oy))

    out = {"ok": False, "dy": None, "signed_dy": None, "p0": None, "p1": None, "idx0": None, "idx1": None}

    # Need detections and classes
    if ck is None or ck.shape[0] == 0 or cc is None:
        return out

    num_dets = ck.shape[0]
    cks_len = 0 if cks is None else cks.shape[0]
    cs_len  = 0 if cs  is None else cs.shape[0]
    cc_len  = 0 if cc  is None else len(cc)

    # Per-detection score (prefer mean keypoint conf, else box conf if available)
    scores = np.zeros(num_dets, dtype=float)
    for i in range(num_dets):
        s = 0.0
        if (cks is not None) and (i < cks_len):
            arr = cks[i]
            if hasattr(arr, "size") and arr.size > 0 and not np.isnan(arr).all():
                s = float(np.nanmean(arr))
        elif (cs is not None) and (i < cs_len):
            s = float(cs[i])
        scores[i] = s

    # Helper: best index for a class
    def _best_idx_for_class(target_cls: int) -> Optional[int]:
        if cc_len == 0:
            return None
        idxs = np.where(cc.astype(int) == int(target_cls))[0]
        if idxs.size == 0:
            return None
        best_local = int(np.argmax(scores[idxs]))
        return int(idxs[best_local])

    idx0 = _best_idx_for_class(0)
    idx1 = _best_idx_for_class(1)
    out["idx0"], out["idx1"] = idx0, idx1

    if idx0 is None or idx1 is None:
        return out

    kxy0 = ck[idx0]  # (K,2)
    kxy1 = ck[idx1]  # (K,2)

    # Optional gate: enforce min_kpt_conf by requiring at least one kpt >= threshold
    def _has_valid_kpt(i: int) -> bool:
        if cks is None or i >= cks_len:
            return True  # no conf info → accept
        arr = cks[i]
        if arr is None or (not hasattr(arr, "size")) or arr.size == 0:
            return True
        return np.nanmax(arr) >= float(min_kpt_conf)

    if (not _has_valid_kpt(idx0)) or (not _has_valid_kpt(idx1)):
        return out

    p0x = float(np.mean(kxy0[:, 0])); p0y = float(np.mean(kxy0[:, 1]))
    p1x = float(np.mean(kxy1[:, 0])); p1y = float(np.mean(kxy1[:, 1]))
    out["p0"] = (p0x, p0y)
    out["p1"] = (p1x, p1y)

    # Vertical distance
    signed_dy = p1y - p0y
    dy = abs(signed_dy)
    low, high = float(dist_range[0]), float(dist_range[1])
    ok = (low <= dy <= high)

    out["signed_dy"] = float(signed_dy)
    out["dy"] = float(dy)
    out["ok"] = bool(ok)

    # Draw
    if draw and annotated_bgr is not None:
        color0 = (0, 255, 0)   # class 0: green
        color1 = (255, 0, 0)   # class 1: blue
        colorL = (0, 255, 0) if ok else (0, 0, 255)

        cv2.circle(annotated_bgr, (int(round(p0x)), int(round(p0y))), 4, color0, -1, lineType=cv2.LINE_AA)
        cv2.circle(annotated_bgr, (int(round(p1x)), int(round(p1y))), 4, color1, -1, lineType=cv2.LINE_AA)
        cv2.line(annotated_bgr, (int(round(p0x)), int(round(p0y))),
                               (int(round(p1x)), int(round(p1y))),
                               colorL, 2, lineType=cv2.LINE_AA)
        label = f"dy={dy:.1f}px ({'OK' if ok else 'NG'})"
        tx = int(round((p0x + p1x) / 2.0))
        ty = int(round((p0y + p1y) / 2.0))
        cv2.putText(annotated_bgr, label, (tx + 6, max(15, ty - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colorL, 2, cv2.LINE_AA)

    print(f"Center tile eval (class0 idx={idx0}, class1 idx={idx1}): dy={dy:.2f}, signed_dy={signed_dy:.2f}, ok={ok}")
    return out


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
                       conf: float,
                       iou: float,
                       draw_ok_green: bool = True):
    """
    Slide windows across middle strip; draw NG (red) and OK (green if enabled).
    Returns list of {'roi_xywh': (x,y,w,h), 'ok': bool, 'dy': Optional[float], 'signed_dy': Optional[float],
                     'p0': Optional[(x,y)], 'p1': Optional[(x,y)]}
    """
    strip, (sx, sy, sw, sh) = _center_strip_after_lr(proc, left_w=left_w, right_w=right_w)
    results = []

    for local_x, w in _sliding_windows(sw, tile_w, overlap_pct):
        tile = strip[0:sh, local_x:local_x + w]
        if tile is None or tile.size == 0:
            continue

        gx = sx + local_x + global_offset[0]
        gy = sy + global_offset[1]
        roi = (gx, gy, w, sh)

        r = _evaluate_center_tile(
            model_center,
            tile_bgr=tile,
            tile_offset_xy=(gx, gy),
            kpt_pair=kpt_pair,                # kept for signature parity
            dist_range=dist_range,
            min_kpt_conf=min_kpt_conf,
            conf=conf,
            iou=iou,
            annotated_bgr=annotated_bgr,      # <-- let the function draw
            draw=True
        )
        r["roi_xywh"] = roi
        results.append(r)

        # draw the window rectangle (OK/NG)
        color = (0, 255, 0) if r["ok"] else (0, 0, 255)
        if r["ok"]:
            if draw_ok_green:
                cv2.rectangle(annotated_bgr, (gx, gy), (gx + w, gy + sh), color, 3, lineType=cv2.LINE_AA)
        else:
            cv2.rectangle(annotated_bgr, (gx, gy), (gx + w, gy + sh), color, 3, lineType=cv2.LINE_AA)

    return results


def J30_Tape_Check(img_bgr: np.ndarray,
                   model_left,
                   model_right,
                   model_center,
                   return_annotated: bool = True,
                   *,
                   # ---- external inputs (override defaults) ----
                   crop_from_top: bool = CROP_FROM_TOP,
                   crop_from_bottom: bool = CROP_FROM_BOTTOM,
                   crop_height: int = CROP_HEIGHT_PIXELS,
                   crop_top_half_fallback: bool = CROP_TOP_HALF,
                   trim_left: int = TRIM_LEFT,
                   trim_right: int = TRIM_RIGHT,
                   left_width: int = LEFTMOST_CUSTOM_WIDTH,
                   right_width: int = RIGHTMOST_CUSTOM_WIDTH,
                   dx_range_left: Tuple[float,float] = DX_RANGE_LEFT,
                   dx_range_right: Tuple[float,float] = DX_RANGE_RIGHT,
                   yolo_conf: float = YOLO_CONF,
                   yolo_iou: float = YOLO_IOU,
                   center_tile_width: int = CENTER_TILE_WIDTH,
                   center_overlap_pct: float = CENTER_OVERLAP_PCT,
                   center_kpt_pair: Tuple[int,int] = CENTER_KPT_PAIR,
                   center_dist_range: Tuple[float,float] = CENTER_DIST_RANGE,
                   center_min_kpt_conf: float = CENTER_MIN_KPT_CONF,
                   center_draw_ok_green: bool = CENTER_DRAW_OK_GREEN):
    """
    External-call-friendly version:
      - All tunables are keyword args with sensible defaults from module-level constants.
      - Supports bottom crop via crop_from_bottom (mutually exclusive with crop_from_top).
    Returns: (annotated_bgr, overall_ok, center_windows)
    """
    assert img_bgr is not None and img_bgr.ndim == 3

    # 1) vertical crop with y-offset
    proc, offset_y = _crop_vertical(
        img_bgr,
        crop_from_top=crop_from_top,
        crop_from_bottom=crop_from_bottom,
        crop_height=crop_height,
        top_half_fallback=crop_top_half_fallback
    )

    # 2) side trim with x-offset
    proc, offset_x, _ = _apply_side_trim(proc, trim_left, trim_right)
    global_offset = (offset_x, offset_y)

    # 3) left/right tiles + inference (pass conf/iou)
    tiles_lr = _make_left_right_tiles(proc, left_w=left_width, right_w=right_width)

    left_img, (lx, ly, lw, lh) = tiles_lr["left"]
    l_results = _run_yolo_pose(model_left, left_img, conf=yolo_conf, iou=yolo_iou)
    lk, lks, lb, ls, lc = _extract_pose_from_results(l_results,
                                                     tile_offset_xy=(lx + global_offset[0], ly + global_offset[1]))
    left_det = TileDetResult("left",
                             (lx + global_offset[0], ly + global_offset[1], lw, lh),
                             lk, lks, lb, ls, lc)

    right_img, (rx, ry, rw, rh) = tiles_lr["right"]
    r_results = _run_yolo_pose(model_right, right_img, conf=yolo_conf, iou=yolo_iou)
    rk, rks, rb, rs, rc = _extract_pose_from_results(r_results,
                                                     tile_offset_xy=(rx + global_offset[0], ry + global_offset[1]))
    right_det = TileDetResult("right",
                              (rx + global_offset[0], ry + global_offset[1], rw, rh),
                              rk, rks, rb, rs, rc)

    # 4) annotate base
    annotated = img_bgr.copy()
    if return_annotated:
        _draw_tile_annotations(annotated, left_det,  color=(0, 255, 0))
        _draw_tile_annotations(annotated, right_det, color=(255, 0, 0))

    result = J30Result(annotated_bgr=annotated, left=left_det, right=right_det)

    # 5) evaluate L/R
    eval_lr = _evaluate_left_right(
        result,
        dx_range_left=dx_range_left,
        dx_range_right=dx_range_right,
        min_conf=yolo_conf,
        draw_text=True
    )

    # 6) sliding center windows across middle strip (height = proc height)
    center_windows = _scan_center_strip(
        annotated_bgr=result.annotated_bgr,
        proc=proc,
        global_offset=global_offset,
        model_center=model_center,
        left_w=left_width,
        right_w=right_width,
        tile_w=center_tile_width,
        overlap_pct=center_overlap_pct,
        kpt_pair=center_kpt_pair,
        dist_range=center_dist_range,
        min_kpt_conf=center_min_kpt_conf,
        conf=yolo_conf,
        iou=yolo_iou,
        draw_ok_green=center_draw_ok_green
    )

    centers_ok = all(w["ok"] for w in center_windows) if center_windows else False
    overall_ok = bool(eval_lr["left"]["ok"] and eval_lr["right"]["ok"] and centers_ok)
    return result.annotated_bgr, overall_ok, center_windows
