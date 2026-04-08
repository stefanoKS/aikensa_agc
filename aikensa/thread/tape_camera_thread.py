import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage


class TapeCameraThread(QThread):
    """
    A simple camera thread for streaming from a second camera (tape camera)
    using OpenCV. Captures at 1280x720 resolution.
    """
    frameSignal = pyqtSignal(QImage)

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720, parent=None):
        super(TapeCameraThread, self).__init__(parent)
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.running = False
        self.cap = None
        self.camera_available = False

    @staticmethod
    def find_available_camera(start_index: int = 0, max_index: int = 5) -> int:
        """Find the first available camera index."""
        for i in range(start_index, max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                return i
            cap.release()
        return -1

    def run(self):
        self.running = True
        
        # Try the specified index first, then search for available camera
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            print(f"[TapeCameraThread] Camera index {self.camera_index} not available, searching...")
            available_idx = self.find_available_camera()
            if available_idx >= 0:
                self.camera_index = available_idx
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            print(f"[TapeCameraThread] No camera available. Tape camera stream disabled.")
            self.running = False
            self.camera_available = False
            return
        
        self.camera_available = True

        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Optionally set FPS
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"[TapeCameraThread] Camera {self.camera_index} opened at {self.width}x{self.height}")
        
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Get image dimensions
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                
                # Create QImage
                qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                # Emit the signal with the QImage
                self.frameSignal.emit(qimg.copy())
            else:
                # If frame read failed, wait a bit before retrying
                self.msleep(10)

        # Cleanup
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print(f"[TapeCameraThread] Camera {self.camera_index} released")

    def stop(self):
        """Stop the camera thread."""
        self.running = False
        self.wait()

    def is_running(self):
        """Check if the thread is running."""
        return self.running
