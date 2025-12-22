import atexit
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Tuple, Optional, Iterable

import numpy as np
import cv2
import imagingcontrol4 as ic4


# =============================================================================
# Placeholder capture (cv2.VideoCapture-like)
# =============================================================================

@dataclass
class PlaceholderIC4Capture:
    """
    Drop-in replacement for IC4Capture when no camera is available.

    - isOpened() -> True
    - read(timeout_ms=...) -> (True, frame) where frame is ALWAYS (height,width,3) BGR
    - get/set methods exist to match your usage.
    """
    cam_id_or_serial: Union[int, str]
    width: int = 3072
    height: int = 2048
    fps: float = 30.0
    color: bool = True
    placeholder_path: str = "./aikensa/assets/no_camera.png"
    _open: bool = True
    _frame_bgr: Optional[np.ndarray] = None

    # cv2-like API
    def isOpened(self) -> bool:
        return self._open

    def release(self):
        self._open = False

    def read(self, timeout_ms: int = 1000) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._open:
            return False, None
        frame = self._get_or_make_frame()
        return True, frame.copy()

    # Keep compatibility with your `.get()` usage
    _W   = getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3)
    _H   = getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4)
    _FPS = getattr(cv2, "CAP_PROP_FPS", 5)
    _EXP = getattr(cv2, "CAP_PROP_EXPOSURE", 15)
    _GAIN= getattr(cv2, "CAP_PROP_GAIN", 14)
    _WB  = getattr(cv2, "CAP_PROP_WB_TEMPERATURE", 45)

    def get(self, prop_id: int) -> float:
        if prop_id == self._W:
            return float(self.width)
        if prop_id == self._H:
            return float(self.height)
        if prop_id == self._FPS:
            return float(self.fps)
        # Not meaningful for placeholder but keep API
        if prop_id in (self._EXP, self._GAIN, self._WB):
            return 0.0
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        if prop_id == self._FPS:
            self.fps = float(value)
            return True
        # ignore EXP/GAIN/WB sets
        return True

    def _get_or_make_frame(self) -> np.ndarray:
        if self._frame_bgr is not None:
            return self._frame_bgr

        # Load image if exists
        img = None
        p = Path(self.placeholder_path)
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)

        if img is None:
            # Fallback: generated black image with text
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            txt = f"NO CAMERA (PLACEHOLDER)\nID: {self.cam_id_or_serial}"
            y = 120
            for line in txt.splitlines():
                cv2.putText(
                    img,
                    line,
                    (60, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA,
                )
                y += 80
        else:
            # Normalize to BGR 3ch
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Enforce requested resolution exactly (3072x2048 default)
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        self._frame_bgr = img
        return img


# =============================================================================
# ---- one-time SDK context ----
# =============================================================================

_IC4_CTX = None
def _ensure_ic4_context():
    global _IC4_CTX
    if _IC4_CTX is None:
        _IC4_CTX = ic4.Library.init_context(
            api_log_level=ic4.LogLevel.WARNING,
            log_targets=ic4.LogTarget.STDERR
        )
        _IC4_CTX.__enter__()
        atexit.register(_IC4_CTX.__exit__, None, None, None)

def _find_device(index_or_serial: Union[int, str]) -> ic4.DeviceInfo:
    devs = ic4.DeviceEnum.devices()
    if not devs:
        raise RuntimeError("No Imaging Source cameras found.")
    if isinstance(index_or_serial, int):
        if not (0 <= index_or_serial < len(devs)):
            raise RuntimeError(f"Camera index {index_or_serial} out of range (0..{len(devs)-1})")
        return devs[index_or_serial]
    for d in devs:
        if d.serial == index_or_serial:
            return d
    for d in devs:
        if (index_or_serial or "").lower() in (d.model_name or "").lower():
            return d
    raise RuntimeError(f"No camera matched '{index_or_serial}'")

# handy helpers for tolerant property IO
def _try_set(pm: ic4.PropertyMap, prop_ids: Iterable, value) -> bool:
    for pid in prop_ids:
        if pid is None:
            continue
        try:
            pm.set_value(pid, value)
            return True
        except ic4.IC4Exception:
            continue
    return False

def _try_get_int(pm: ic4.PropertyMap, prop_ids: Iterable) -> Optional[int]:
    for pid in prop_ids:
        if pid is None:
            continue
        try:
            return int(pm.get_value_int(pid))
        except ic4.IC4Exception:
            continue
    return None

def _try_get_float(pm: ic4.PropertyMap, prop_ids: Iterable) -> Optional[float]:
    for pid in prop_ids:
        if pid is None:
            continue
        try:
            return float(pm.get_value_float(pid))
        except ic4.IC4Exception:
            continue
    return None

def _try_get_str(pm: ic4.PropertyMap, prop_ids: Iterable) -> Optional[str]:
    for pid in prop_ids:
        if pid is None:
            continue
        try:
            return pm.get_value_str(pid)
        except ic4.IC4Exception:
            continue
    return None


# PropId aliases (be tolerant across models)
PID_PIXEL_FORMAT       = getattr(ic4.PropId, "PIXEL_FORMAT", None)
PID_WIDTH              = getattr(ic4.PropId, "WIDTH", None)
PID_HEIGHT             = getattr(ic4.PropId, "HEIGHT", None)
PID_OFFSET_X           = getattr(ic4.PropId, "OFFSET_X", None)
PID_OFFSET_Y           = getattr(ic4.PropId, "OFFSET_Y", None)
PID_OFFSET_AUTO_CENTER = getattr(ic4.PropId, "OFFSET_AUTO_CENTER", None)

PID_EXPOSURE_AUTO      = getattr(ic4.PropId, "EXPOSURE_AUTO", None)
PID_EXPOSURE_TIME      = getattr(ic4.PropId, "EXPOSURE_TIME", None)
PID_GAIN_AUTO          = getattr(ic4.PropId, "GAIN_AUTO", None)
PID_GAIN               = getattr(ic4.PropId, "GAIN", None)
PID_WB_AUTO            = getattr(ic4.PropId, "BALANCE_WHITE_AUTO", None)
PID_WB_TEMP            = getattr(ic4.PropId, "WHITEBALANCE_TEMPERATURE", None)

# Frame-rate names differ; try both, but it's optional for SnapSink.
PID_FRAME_RATE         = getattr(ic4.PropId, "FRAME_RATE", None)
PID_ACQ_FRAME_RATE     = getattr(ic4.PropId, "ACQUISITION_FRAME_RATE", None)
PID_ACQ_FR_EN          = getattr(ic4.PropId, "ACQUISITION_FRAME_RATE_ENABLE", None)

PF = ic4.PixelFormat


class IC4Capture:
    _W   = getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3)
    _H   = getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4)
    _FPS = getattr(cv2, "CAP_PROP_FPS", 5)
    _EXP = getattr(cv2, "CAP_PROP_EXPOSURE", 15)
    _GAIN= getattr(cv2, "CAP_PROP_GAIN", 14)
    _WB  = getattr(cv2, "CAP_PROP_WB_TEMPERATURE", 45)

    def __init__(self,
                 cam: Union[int, str],
                 width: int = 3072, height: int = 2048, fps: float = 5.0,
                 color: bool = True,
                 exposure_us: float | None = None,
                 gain_db: float | None = None,
                 wb_temperature: int | None = None):
        _ensure_ic4_context()

        # Open device EXACTLY like the working sample
        dev_info = _find_device(cam)
        self._grab = ic4.Grabber(dev_info)
        pm = self._grab.device_property_map

        # Pixel format
        if color:
            if not _try_set(pm, (PID_PIXEL_FORMAT,), getattr(PF, "BayerRG8", None)):
                _try_set(pm, (PID_PIXEL_FORMAT,), getattr(PF, "BGR8", None))
        else:
            _try_set(pm, (PID_PIXEL_FORMAT,), getattr(PF, "Mono8", None))

        # ROI / size
        _try_set(pm, (PID_OFFSET_AUTO_CENTER,), "Off")
        _try_set(pm, (PID_OFFSET_X,), 0)
        _try_set(pm, (PID_OFFSET_Y,), 0)
        _try_set(pm, (PID_WIDTH,),  width)
        _try_set(pm, (PID_HEIGHT,), height)

        # Exposure / Gain / WB
        _try_set(pm, (PID_EXPOSURE_AUTO,), "Off")
        if exposure_us is not None:
            _try_set(pm, (PID_EXPOSURE_TIME,), float(exposure_us))
        _try_set(pm, (PID_GAIN_AUTO,), "Off")
        if gain_db is not None:
            _try_set(pm, (PID_GAIN,), float(gain_db))
        if color and wb_temperature is not None:
            _try_set(pm, (PID_WB_AUTO,), "Off")
            _try_set(pm, (PID_WB_TEMP,), int(wb_temperature))

        # Frame rate (optional; if out-of-range it will just be ignored by the device)
        _try_set(pm, (PID_ACQ_FR_EN,), True)
        _try_set(pm, (PID_ACQ_FRAME_RATE, PID_FRAME_RATE), float(fps))

        # Start stream with SnapSink (like the sample)
        self._sink = ic4.SnapSink()
        self._grab.stream_setup(self._sink)

        # cache requested; getters will try readback
        self._w, self._h, self._fps = width, height, float(fps)
        self._open = True

        # Decide conversion path
        pf_name = _try_get_str(pm, (PID_PIXEL_FORMAT,))
        self._convert = None
        pf_enum = getattr(PF, pf_name) if pf_name and hasattr(PF, pf_name) else None
        if pf_enum in (getattr(PF, "BayerRG8", None),
                       getattr(PF, "BayerBG8", None),
                       getattr(PF, "BayerGR8", None),
                       getattr(PF, "BayerGB8", None)):
            self._convert = {
                getattr(PF, "BayerRG8", None): cv2.COLOR_BayerRG2BGR,
                getattr(PF, "BayerBG8", None): cv2.COLOR_BayerBG2BGR,
                getattr(PF, "BayerGR8", None): cv2.COLOR_BayerGR2BGR,
                getattr(PF, "BayerGB8", None): cv2.COLOR_BayerGB2BGR,
            }.get(pf_enum, None)
        elif pf_enum == getattr(PF, "RGB8", None):
            self._convert = cv2.COLOR_RGB2BGR
        elif pf_enum in (getattr(PF, "YUV422Packed", None), getattr(PF, "YUY2", None)):
            self._convert = cv2.COLOR_YUV2BGR_YUY2  # adjust to UYVY if needed

    # cv2-like API
    def isOpened(self) -> bool:
        return self._open

    def release(self):
        if self._open:
            try:
                self._grab.stream_stop()
            finally:
                self._grab.device_close()
                self._open = False

    def read(self, timeout_ms: int = 1000) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._open:
            return False, None
        try:
            buf = self._sink.snap_single(int(timeout_ms))
        except ic4.IC4Exception:
            return False, None
        if buf is None:
            return False, None

        frame = buf.numpy_copy()              # independent NumPy array
        if self._convert is not None:
            frame = cv2.cvtColor(frame, self._convert)
        return True, frame

    def get(self, prop_id: int) -> float:
        pm = self._grab.device_property_map
        if prop_id == self._W:
            val = _try_get_int(pm, (PID_WIDTH,));  return float(val if val is not None else self._w)
        if prop_id == self._H:
            val = _try_get_int(pm, (PID_HEIGHT,)); return float(val if val is not None else self._h)
        if prop_id == self._FPS:
            val = _try_get_float(pm, (PID_ACQ_FRAME_RATE, PID_FRAME_RATE))
            return float(val if val is not None else self._fps)
        if prop_id == self._EXP:
            val = _try_get_float(pm, (PID_EXPOSURE_TIME,)); return float(val if val is not None else 0.0)
        if prop_id == self._GAIN:
            val = _try_get_float(pm, (PID_GAIN,));          return float(val if val is not None else 0.0)
        if prop_id == self._WB:
            val = _try_get_int(pm, (PID_WB_TEMP,));         return float(val if val is not None else 0.0)
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        pm = self._grab.device_property_map
        try:
            if prop_id == self._FPS:
                _try_set(pm, (PID_ACQ_FR_EN,), True)
                ok = _try_set(pm, (PID_ACQ_FRAME_RATE, PID_FRAME_RATE), float(value))
                if ok:
                    self._fps = float(value)
                return ok
            if prop_id == self._EXP:
                _try_set(pm, (PID_EXPOSURE_AUTO,), "Off")
                _try_set(pm, (PID_EXPOSURE_TIME,), float(value))
                return True
            if prop_id == self._GAIN:
                _try_set(pm, (PID_GAIN_AUTO,), "Off")
                _try_set(pm, (PID_GAIN,), float(value))
                return True
            if prop_id == self._WB:
                _try_set(pm, (PID_WB_AUTO,), "Off")
                _try_set(pm, (PID_WB_TEMP,), int(value))
                return True
        except ic4.IC4Exception:
            return False
        return False


# =============================================================================
# Public initializer with check-gate + placeholder fallback
# =============================================================================

def initialize_camera_ic4(
    cam_id_or_serial: Union[int, str],
    width: int = 3072,
    height: int = 2048,
    fps: float = 30.0,
    color: bool = True,
    exposure_us: float | None = None,
    gain_db: float | None = None,
    wb_temperature: int | None = None,
    auto_exposure: bool = False,
    auto_gain: bool = False,
    auto_wb: bool = False,
    *,
    fallback_to_placeholder: bool = True,
    placeholder_path: str = "./aikensa/assets/no_camera.png",
):
    """
    If camera init/config fails -> return PlaceholderIC4Capture.
    Placeholder returns a static frame at exactly (width,height) = (3072,2048) by default.
    """
    try:
        cam = IC4Capture(
            cam_id_or_serial,
            width, height, fps, color,
            exposure_us, gain_db, wb_temperature
        )

        # Apply auto toggles (tolerant)
        pm = cam._grab.device_property_map
        _try_set(pm, (PID_EXPOSURE_AUTO,), auto_exposure)
        _try_set(pm, (PID_GAIN_AUTO,), auto_gain)
        _try_set(pm, (PID_WB_AUTO,), auto_wb)

        # Gate: try a first frame so you immediately know it's usable
        ok, frame = cam.read(timeout_ms=1000)
        if not ok or frame is None:
            raise RuntimeError("Camera opened but first frame grab failed.")

        return cam

    except Exception as e:
        if not fallback_to_placeholder:
            raise

        print(f"[initialize_camera_ic4] Using placeholder (camera unavailable). Reason: {e}")
        return PlaceholderIC4Capture(
            cam_id_or_serial=cam_id_or_serial,
            width=width,
            height=height,
            fps=fps,
            color=color,
            placeholder_path=placeholder_path,
        )
