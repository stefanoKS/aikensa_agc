import atexit
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import cv2
import imagingcontrol4 as ic4
import numpy as np


@dataclass
class PlaceholderIC4Capture:
    cam_id_or_serial: Union[int, str]
    width: int = 3072
    height: int = 2048
    fps: float = 30.0
    color: bool = True
    placeholder_path: str = "./aikensa/assets/no_camera.png"
    _open: bool = True
    _frame_bgr: Optional[np.ndarray] = None

    _W = getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3)
    _H = getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4)
    _FPS = getattr(cv2, "CAP_PROP_FPS", 5)
    _EXP = getattr(cv2, "CAP_PROP_EXPOSURE", 15)
    _GAIN = getattr(cv2, "CAP_PROP_GAIN", 14)
    _WB = getattr(cv2, "CAP_PROP_WB_TEMPERATURE", 45)

    def isOpened(self) -> bool:
        return self._open

    def release(self):
        self._open = False

    def read(self, timeout_ms: int = 1000) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._open:
            return False, None
        frame = self._get_or_make_frame()
        return True, frame.copy()

    def get(self, prop_id: int) -> float:
        if prop_id == self._W:
            return float(self.width)
        if prop_id == self._H:
            return float(self.height)
        if prop_id == self._FPS:
            return float(self.fps)
        if prop_id in (self._EXP, self._GAIN, self._WB):
            return 0.0
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        if prop_id == self._FPS:
            self.fps = float(value)
            return True
        return True

    def _get_or_make_frame(self) -> np.ndarray:
        if self._frame_bgr is not None:
            return self._frame_bgr

        image = None
        placeholder = Path(self.placeholder_path)
        if placeholder.exists():
            image = cv2.imread(str(placeholder), cv2.IMREAD_UNCHANGED)

        if image is None:
            image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            text = f"NO CAMERA (PLACEHOLDER)\nID: {self.cam_id_or_serial}"
            y = 120
            for line in text.splitlines():
                cv2.putText(
                    image,
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
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        self._frame_bgr = image
        return image


_IC4_CTX = None


def _ensure_ic4_context():
    global _IC4_CTX
    if _IC4_CTX is None:
        _IC4_CTX = ic4.Library.init_context(
            api_log_level=ic4.LogLevel.WARNING,
            log_targets=ic4.LogTarget.STDERR,
        )
        _IC4_CTX.__enter__()
        atexit.register(_IC4_CTX.__exit__, None, None, None)


def _find_device(index_or_serial: Union[int, str]) -> ic4.DeviceInfo:
    devices = ic4.DeviceEnum.devices()
    if not devices:
        raise RuntimeError("No Imaging Source cameras found.")
    if isinstance(index_or_serial, int):
        if not (0 <= index_or_serial < len(devices)):
            raise RuntimeError(f"Camera index {index_or_serial} out of range (0..{len(devices) - 1})")
        return devices[index_or_serial]
    for device in devices:
        if device.serial == index_or_serial:
            return device
    for device in devices:
        if (index_or_serial or "").lower() in (device.model_name or "").lower():
            return device
    raise RuntimeError(f"No camera matched '{index_or_serial}'")


def _try_set(pm: ic4.PropertyMap, prop_ids: Iterable, value) -> bool:
    for prop_id in prop_ids:
        if prop_id is None:
            continue
        try:
            pm.set_value(prop_id, value)
            return True
        except ic4.IC4Exception:
            continue
    return False


def _try_get_int(pm: ic4.PropertyMap, prop_ids: Iterable) -> Optional[int]:
    for prop_id in prop_ids:
        if prop_id is None:
            continue
        try:
            return int(pm.get_value_int(prop_id))
        except ic4.IC4Exception:
            continue
    return None


def _try_get_float(pm: ic4.PropertyMap, prop_ids: Iterable) -> Optional[float]:
    for prop_id in prop_ids:
        if prop_id is None:
            continue
        try:
            return float(pm.get_value_float(prop_id))
        except ic4.IC4Exception:
            continue
    return None


def _try_get_str(pm: ic4.PropertyMap, prop_ids: Iterable) -> Optional[str]:
    for prop_id in prop_ids:
        if prop_id is None:
            continue
        try:
            return pm.get_value_str(prop_id)
        except ic4.IC4Exception:
            continue
    return None


def _try_find_property(pm: ic4.PropertyMap, prop_id):
    if prop_id is None:
        return None
    try:
        return pm.find(prop_id)
    except ic4.IC4Exception:
        return None


def _get_supported_enum_names(pm: ic4.PropertyMap, prop_id) -> Tuple[str, ...]:
    prop = _try_find_property(pm, prop_id)
    if prop is None or not hasattr(prop, "entries"):
        return ()
    return tuple(entry.name for entry in prop.entries)


def _clamp_float_property(pm: ic4.PropertyMap, prop_id, value: float) -> float:
    prop = _try_find_property(pm, prop_id)
    if prop is None:
        return float(value)

    minimum = getattr(prop, "minimum", None)
    maximum = getattr(prop, "maximum", None)
    clamped = float(value)
    if minimum is not None:
        clamped = max(clamped, float(minimum))
    if maximum is not None:
        clamped = min(clamped, float(maximum))
    return clamped


def _recommended_timeout_ms(fps: Optional[float], minimum_ms: int = 1000) -> int:
    if fps is None or fps <= 0:
        return minimum_ms
    frame_interval_ms = int(np.ceil(1000.0 / fps))
    return max(minimum_ms, frame_interval_ms * 3)


def _normalize_frame_to_bgr(frame: np.ndarray, convert_code: Optional[int]) -> Optional[np.ndarray]:
    if frame is None:
        return None

    output = frame
    if convert_code is not None:
        output = cv2.cvtColor(output, convert_code)
    elif output.ndim == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    elif output.ndim == 3 and output.shape[2] == 4:
        output = cv2.cvtColor(output, cv2.COLOR_BGRA2BGR)

    if output.dtype != np.uint8:
        output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
        output = output.astype(np.uint8)

    if output.ndim == 3 and output.shape[2] == 1:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    return output


def _frame_is_effectively_black(frame: Optional[np.ndarray]) -> bool:
    if frame is None or frame.size == 0:
        return True
    return int(frame.max()) <= 1


PID_PIXEL_FORMAT = getattr(ic4.PropId, "PIXEL_FORMAT", None)
PID_WIDTH = getattr(ic4.PropId, "WIDTH", None)
PID_HEIGHT = getattr(ic4.PropId, "HEIGHT", None)
PID_OFFSET_X = getattr(ic4.PropId, "OFFSET_X", None)
PID_OFFSET_Y = getattr(ic4.PropId, "OFFSET_Y", None)
PID_OFFSET_AUTO_CENTER = getattr(ic4.PropId, "OFFSET_AUTO_CENTER", None)

PID_EXPOSURE_AUTO = getattr(ic4.PropId, "EXPOSURE_AUTO", None)
PID_EXPOSURE_TIME = getattr(ic4.PropId, "EXPOSURE_TIME", None)
PID_GAIN_AUTO = getattr(ic4.PropId, "GAIN_AUTO", None)
PID_GAIN = getattr(ic4.PropId, "GAIN", None)
PID_WB_AUTO = getattr(ic4.PropId, "BALANCE_WHITE_AUTO", None)
PID_WB_TEMP = getattr(ic4.PropId, "WHITEBALANCE_TEMPERATURE", None)

PID_FRAME_RATE = getattr(ic4.PropId, "FRAME_RATE", None)
PID_ACQ_FRAME_RATE = getattr(ic4.PropId, "ACQUISITION_FRAME_RATE", None)
PID_ACQ_FR_EN = getattr(ic4.PropId, "ACQUISITION_FRAME_RATE_ENABLE", None)

PF = ic4.PixelFormat


class IC4Capture:
    _W = getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3)
    _H = getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4)
    _FPS = getattr(cv2, "CAP_PROP_FPS", 5)
    _EXP = getattr(cv2, "CAP_PROP_EXPOSURE", 15)
    _GAIN = getattr(cv2, "CAP_PROP_GAIN", 14)
    _WB = getattr(cv2, "CAP_PROP_WB_TEMPERATURE", 45)
    _COLOR_PIXEL_FORMATS = (
        "BGR8",
        "RGB8",
        "BayerRG8",
        "BayerBG8",
        "BayerGR8",
        "BayerGB8",
        "Mono8",
    )

    def __init__(
        self,
        cam: Union[int, str],
        width: int = 3072,
        height: int = 2048,
        fps: float = 5.0,
        color: bool = True,
        exposure_us: Optional[float] = None,
        gain_db: Optional[float] = None,
        wb_temperature: Optional[int] = None,
    ):
        _ensure_ic4_context()

        dev_info = _find_device(cam)
        self._grab = ic4.Grabber(dev_info)
        pm = self._grab.device_property_map
        self._selected_pixel_format = None
        self._convert = None
        self._recommended_timeout_ms = 1000

        if color:
            self._select_pixel_format(pm, self._COLOR_PIXEL_FORMATS)
        else:
            self._select_pixel_format(pm, ("Mono8",))

        _try_set(pm, (PID_OFFSET_AUTO_CENTER,), "Off")
        _try_set(pm, (PID_OFFSET_X,), 0)
        _try_set(pm, (PID_OFFSET_Y,), 0)
        _try_set(pm, (PID_WIDTH,), width)
        _try_set(pm, (PID_HEIGHT,), height)

        _try_set(pm, (PID_EXPOSURE_AUTO,), "Off")
        if exposure_us is not None:
            _try_set(pm, (PID_EXPOSURE_TIME,), float(exposure_us))
        _try_set(pm, (PID_GAIN_AUTO,), "Off")
        if gain_db is not None:
            _try_set(pm, (PID_GAIN,), float(gain_db))
        if color and wb_temperature is not None:
            _try_set(pm, (PID_WB_AUTO,), "Off")
            _try_set(pm, (PID_WB_TEMP,), int(wb_temperature))

        _try_set(pm, (PID_ACQ_FR_EN,), True)
        self._apply_frame_rate(pm, fps)

        self._sink = ic4.SnapSink()
        self._grab.stream_setup(self._sink)

        self._w, self._h, self._fps = width, height, float(fps)
        self._open = True

        self._configure_conversion(pm)
        actual_fps = _try_get_float(pm, (PID_ACQ_FRAME_RATE, PID_FRAME_RATE))
        if actual_fps is not None:
            self._fps = float(actual_fps)
        self._recommended_timeout_ms = _recommended_timeout_ms(self._fps)

    def _select_pixel_format(self, pm: ic4.PropertyMap, format_names: Iterable[str]) -> Optional[str]:
        supported_names = set(_get_supported_enum_names(pm, PID_PIXEL_FORMAT))
        for format_name in format_names:
            if supported_names and format_name not in supported_names:
                continue
            pixel_format = getattr(PF, format_name, None)
            if pixel_format is None:
                continue
            if _try_set(pm, (PID_PIXEL_FORMAT,), pixel_format):
                self._selected_pixel_format = format_name
                return format_name
        self._selected_pixel_format = _try_get_str(pm, (PID_PIXEL_FORMAT,))
        return self._selected_pixel_format

    def _apply_frame_rate(self, pm: ic4.PropertyMap, fps: float):
        target_fps = _clamp_float_property(pm, PID_ACQ_FRAME_RATE, float(fps))
        if not _try_set(pm, (PID_ACQ_FRAME_RATE, PID_FRAME_RATE), target_fps):
            self._fps = float(_try_get_float(pm, (PID_ACQ_FRAME_RATE, PID_FRAME_RATE)) or target_fps)
            return
        self._fps = float(_try_get_float(pm, (PID_ACQ_FRAME_RATE, PID_FRAME_RATE)) or target_fps)

    def _configure_conversion(self, pm: ic4.PropertyMap):
        pf_name = _try_get_str(pm, (PID_PIXEL_FORMAT,)) or self._selected_pixel_format
        self._selected_pixel_format = pf_name
        self._convert = None

        pf_enum = getattr(PF, pf_name) if pf_name and hasattr(PF, pf_name) else None
        if pf_enum in (
            getattr(PF, "BayerRG8", None),
            getattr(PF, "BayerBG8", None),
            getattr(PF, "BayerGR8", None),
            getattr(PF, "BayerGB8", None),
        ):
            self._convert = {
                getattr(PF, "BayerRG8", None): cv2.COLOR_BayerRG2BGR,
                getattr(PF, "BayerBG8", None): cv2.COLOR_BayerBG2BGR,
                getattr(PF, "BayerGR8", None): cv2.COLOR_BayerGR2BGR,
                getattr(PF, "BayerGB8", None): cv2.COLOR_BayerGB2BGR,
            }.get(pf_enum, None)
        elif pf_enum == getattr(PF, "RGB8", None):
            self._convert = cv2.COLOR_RGB2BGR
        elif pf_enum in (getattr(PF, "YUV422Packed", None), getattr(PF, "YUY2", None)):
            self._convert = cv2.COLOR_YUV2BGR_YUY2

    def _read_raw(self, timeout_ms: int = 1000) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._open:
            return False, None
        try:
            buffer = self._sink.snap_single(int(timeout_ms))
        except ic4.IC4Exception:
            return False, None
        if buffer is None:
            return False, None
        return True, buffer.numpy_copy()

    def recover_from_black_frame(self, timeout_ms: int = 1000) -> Tuple[bool, Optional[np.ndarray]]:
        pm = self._grab.device_property_map
        current = self._selected_pixel_format or _try_get_str(pm, (PID_PIXEL_FORMAT,))
        format_order = []
        for format_name in (current, *self._COLOR_PIXEL_FORMATS):
            if format_name and format_name not in format_order:
                format_order.append(format_name)

        for format_name in format_order:
            self._select_pixel_format(pm, (format_name,))
            self._configure_conversion(pm)
            ok, raw_frame = self._read_raw(timeout_ms=timeout_ms)
            if not ok:
                continue
            frame = _normalize_frame_to_bgr(raw_frame, self._convert)
            if _frame_is_effectively_black(frame):
                continue
            print(f"[initialize_camera_ic4] Recovered stream using pixel format {self._selected_pixel_format}")
            return True, frame

        return False, None

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
        snap_timeout_ms = max(int(timeout_ms), self._recommended_timeout_ms)
        ok, raw_frame = self._read_raw(timeout_ms=snap_timeout_ms)
        if not ok:
            return False, None

        frame = _normalize_frame_to_bgr(raw_frame, self._convert)
        return frame is not None, frame

    def recommended_timeout_ms(self) -> int:
        return self._recommended_timeout_ms

    def get(self, prop_id: int) -> float:
        pm = self._grab.device_property_map
        if prop_id == self._W:
            value = _try_get_int(pm, (PID_WIDTH,))
            return float(value if value is not None else self._w)
        if prop_id == self._H:
            value = _try_get_int(pm, (PID_HEIGHT,))
            return float(value if value is not None else self._h)
        if prop_id == self._FPS:
            value = _try_get_float(pm, (PID_ACQ_FRAME_RATE, PID_FRAME_RATE))
            return float(value if value is not None else self._fps)
        if prop_id == self._EXP:
            value = _try_get_float(pm, (PID_EXPOSURE_TIME,))
            return float(value if value is not None else 0.0)
        if prop_id == self._GAIN:
            value = _try_get_float(pm, (PID_GAIN,))
            return float(value if value is not None else 0.0)
        if prop_id == self._WB:
            value = _try_get_int(pm, (PID_WB_TEMP,))
            return float(value if value is not None else 0.0)
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


def initialize_camera_ic4(
    cam_id_or_serial: Union[int, str],
    width: int = 3072,
    height: int = 2048,
    fps: float = 30.0,
    color: bool = True,
    exposure_us: Optional[float] = None,
    gain_db: Optional[float] = None,
    wb_temperature: Optional[int] = None,
    auto_exposure: bool = False,
    auto_gain: bool = False,
    auto_wb: bool = False,
    *,
    fallback_to_placeholder: bool = True,
    placeholder_path: str = "./aikensa/assets/no_camera.png",
    first_frame_timeout_ms: int = 1000,
):
    try:
        cam = IC4Capture(
            cam_id_or_serial,
            width,
            height,
            fps,
            color,
            exposure_us,
            gain_db,
            wb_temperature,
        )

        pm = cam._grab.device_property_map
        _try_set(pm, (PID_EXPOSURE_AUTO,), auto_exposure)
        _try_set(pm, (PID_GAIN_AUTO,), auto_gain)
        _try_set(pm, (PID_WB_AUTO,), auto_wb)

        startup_timeout_ms = max(int(first_frame_timeout_ms), cam.recommended_timeout_ms())
        ok, frame = cam.read(timeout_ms=startup_timeout_ms)
        if (not ok or frame is None) or _frame_is_effectively_black(frame):
            ok, frame = cam.recover_from_black_frame(timeout_ms=startup_timeout_ms)
        if not ok or frame is None:
            raise RuntimeError("Camera opened but first frame grab failed.")

        print(
            f"[initialize_camera_ic4] Opened {cam_id_or_serial} with pixel format {cam._selected_pixel_format} "
            f"at {cam.get(cv2.CAP_PROP_FPS):.3f} fps (timeout {cam.recommended_timeout_ms()} ms)"
        )

        return cam

    except Exception as error:
        if not fallback_to_placeholder:
            raise

        print(f"[initialize_camera_ic4] Using placeholder (camera unavailable). Reason: {error}")
        return PlaceholderIC4Capture(
            cam_id_or_serial=cam_id_or_serial,
            width=width,
            height=height,
            fps=fps,
            color=color,
            placeholder_path=placeholder_path,
        )