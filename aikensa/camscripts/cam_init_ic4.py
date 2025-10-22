# ic4_video_capture.py
import atexit
import imagingcontrol4 as ic4
import numpy as np
import cv2

# ---------- one-time library context (like cv2 init) ----------
_ic4_ctx = None
def _ensure_ic4_context():
    global _ic4_ctx
    if _ic4_ctx is None:
        _ic4_ctx = ic4.Library.init_context(
            api_log_level=ic4.LogLevel.WARN,
            log_targets=ic4.LogTarget.STDERR
        )
        _ic4_ctx.__enter__()
        atexit.register(_ic4_ctx.__exit__, None, None, None)

class IC4Capture:
    """
    Minimal wrapper to mimic cv2.VideoCapture where you use:
        cap = initialize_camera(...)
        ok, frame = cap.read()
        cap.release()
    """
    def __init__(
        self,
        device_index_or_serial=0,
        width=3072,
        height=2048,
        fps=30.0,
        pixel_format_hint=("Mono8", "BayerRG8", "YUY2", "RGB8"),  # try in this order
        exposure_us=None,     # e.g., 2000
        gain_db=None,         # e.g., 6.0
        wb_temperature=None,  # e.g., 4500 (for color cams)
        disable_auto=True     # True -> turn off auto exposure/gain/WB if applicable
    ):
        _ensure_ic4_context()

        # --- pick device by index or serial ---
        devs = ic4.DeviceEnum.devices()
        if isinstance(device_index_or_serial, int):
            if not devs:
                raise RuntimeError("No Imaging Source cameras found.")
            dev_info = devs[device_index_or_serial]
        else:
            # match by serial or model name
            matches = [d for d in devs if (d.serial == device_index_or_serial or
                                           device_index_or_serial in (d.model_name or ""))]
            if not matches:
                raise RuntimeError(f"No camera matched '{device_index_or_serial}'.")
            dev_info = matches[0]

        self._grabber = ic4.Grabber(dev_info)
        self._device  = self._grabber.device()

        # --- select exact video format (width/height/pixel format) ---
        self._pixel_format = None
        fmt_candidates = self._device.video_formats()

        chosen = None
        for pf in pixel_format_hint if isinstance(pixel_format_hint, (list, tuple)) else (pixel_format_hint,):
            for f in fmt_candidates:
                if f.width == width and f.height == height and pf in f.pixel_format:
                    chosen = f
                    self._pixel_format = pf
                    break
            if chosen:
                break

        # If not found, fall back to first matching resolution (any pixel format)
        if chosen is None:
            for f in fmt_candidates:
                if f.width == width and f.height == height:
                    chosen = f
                    self._pixel_format = f.pixel_format
                    break

        if chosen is None:
            raise RuntimeError(
                f"No matching format found for {width}x{height}. "
                f"Available examples: {[ (f.width,f.height,f.pixel_format) for f in fmt_candidates[:5] ]}"
            )

        self._device.set_video_format(chosen)
        self._device.set_frame_rate(float(fps))

        # --- set camera properties (true hardware controls) ---
        props = self._device.properties()
        def safe_set(name, value):
            if name in props:
                props[name] = value

        if disable_auto:
            safe_set("Exposure.Auto", False)
            safe_set("Gain.Auto", False)
            safe_set("BalanceWhite.Auto", False)

        if exposure_us is not None:
            safe_set("Exposure.Time", float(exposure_us))
        if gain_db is not None:
            safe_set("Gain.Value", float(gain_db))
        if wb_temperature is not None:
            safe_set("BalanceWhite.Temperature", int(wb_temperature))

        # --- streaming sink (continuous) ---
        self._sink = ic4.QueueSink(queue_size=8)
        self._grabber.stream_setup(self._sink)
        self._grabber.stream_start()

        # color conversion if Bayer or YUY2
        self._color_convert = None
        if "BayerRG8" in self._pixel_format:
            self._color_convert = cv2.COLOR_BayerRG2BGR
        elif "BayerBG8" in self._pixel_format:
            self._color_convert = cv2.COLOR_BayerBG2BGR
        elif "BayerGR8" in self._pixel_format:
            self._color_convert = cv2.COLOR_BayerGR2BGR
        elif "BayerGB8" in self._pixel_format:
            self._color_convert = cv2.COLOR_BayerGB2BGR
        elif "YUY2" in self._pixel_format:
            self._color_convert = "YUY2->BGR"  # handled specially below

        self._closed = False

    def read(self, timeout_ms=100):
        """
        Return (ok, frame) like cv2.VideoCapture.read().
        frame is a NumPy array (Mono8: HxW, Color: HxWx3 BGR if conversion is set).
        """
        if self._closed:
            return False, None

        buf = self._sink.pop_next(timeout_ms=timeout_ms)
        if buf is None:
            return False, None

        arr = buf.as_array()                # zero-copy view
        frame = np.array(arr, copy=False)   # NumPy wrapper

        # Pixel format handling
        if self._color_convert is None:
            # Mono8 or already RGB8/BGR8 from device
            # If it's RGB8, convert to BGR for OpenCV-compat
            if "RGB8" in self._pixel_format:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame

        if self._color_convert == "YUY2->BGR":
            # OpenCV expects HxWx2 in YUY2 and converts to BGR
            return True, cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
        else:
            # Bayer to BGR
            return True, cv2.cvtColor(frame, self._color_convert)

    def release(self):
        if not self._closed:
            try:
                self._grabber.stream_stop()
            finally:
                self._grabber.device_close()
                self._closed = True

    # Optional: minimal .set(...) to change FPS at runtime (extend if you need more)
    def set(self, prop_id, value):
        # Support only FPS for now to keep things simple
        # cv2.CAP_PROP_FPS == 5 (commonly)
        CAP_PROP_FPS = 5
        if prop_id == CAP_PROP_FPS:
            try:
                self._device.set_frame_rate(float(value))
                return True
            except Exception:
                return False
        return False

# --------- your existing function name, now backed by ic4 ----------
def initialize_camera(camNum):
    """
    Drop-in replacement. Returns an object with .read() and .release() like cv2.VideoCapture.
    camNum: index (0,1,...) or serial string (e.g., '12345678').
    """
    # Mirror your original defaults:
    width, height = 3072, 2048
    fps = 24.0

    # For color cams, prefer Bayer to keep bandwidth low and let you control debayer quality.
    # For mono cams, keep Mono8.
    return IC4Capture(
        device_index_or_serial=camNum,
        width=width,
        height=height,
        fps=fps,
        pixel_format_hint=("BayerRG8", "Mono8", "YUY2", "RGB8"),
        # Uncomment to pin exposure/gain/WB as you did with CAP_PROP_*:
        # exposure_us=2000,
        # gain_db=6.0,
        # wb_temperature=4500,
        disable_auto=True
    )
