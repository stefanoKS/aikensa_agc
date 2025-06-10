from PyQt5.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
import time
from aikensa.scripts.TIS import TIS, SinkFormats
import cv2
import yaml

class CameraThread(QThread):
    new_frame = pyqtSignal(np.ndarray)

    def __init__(self, config_path="./TIS_config.yaml"):
        super().__init__()
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        self.serial = config.get("name")
        self.width = config.get("width")
        self.height = config.get("height")
        self.fps = config.get("fps")
        self.running = False

        if None in (self.serial, self.width, self.height, self.fps):
            raise ValueError("Invalid config file.")

    def run(self):
        self.Tis = TIS()
        self.Tis.open_device(self.serial, self.width, self.height, f"{self.fps}/1", SinkFormats.BGRA, False)
        self.Tis.start_pipeline()
        self.running = True
        while self.running:
            if self.Tis.snap_image(0.1):
                image = self.Tis.get_image()
                if image is not None:
                    self.new_frame.emit(image.copy())
            time.sleep(0.01)

    def stop(self):
        self.running = False
        self.wait()
        self.Tis.stop_pipeline()