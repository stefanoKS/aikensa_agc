from PyQt5.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
import time
from aikensa.scripts.TIS import TIS, SinkFormats
import cv2
import yaml
import yaml

class CameraThread(QThread):
    new_frame = pyqtSignal(np.ndarray)

    def __init__(self, config_path = "./TIS_config.yaml", camera_config_path = "../config_yaml/cam_config.yaml"):
        super().__init__()
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        self.serial = config.get("name")
        self.width = config.get("width")
        self.height = config.get("height")
        self.fps = config.get("fps")
        self.running = False

        with open(camera_config_path, "r") as file:
            camera_config = yaml.safe_load(file)
        self.black_level = camera_config.get("BlackLevel")
        self.exposure_auto = camera_config.get("ExposureAuto")
        self.exposure_time = camera_config.get("ExposureTime")
        self.gain = camera_config.get("Gain")
        self.gain_auto = camera_config.get("GainAuto")
        self.white_balance_auto = camera_config.get("WhiteBalanceAuto")

        if None in (self.serial, self.width, self.height, self.fps):
            raise ValueError("Invalid config file.")

    def run(self):
        self.Tis = TIS()
        self.Tis.open_device(self.serial, self.width, self.height, f"{self.fps}/1", SinkFormats.BGRA, False)
        self.Tis.start_pipeline()
        # self.Tis.list_properties()


        self.Tis.set_property("BlackLevel", self.black_level)
        self.Tis.set_property("ExposureAuto", "Off")
        self.Tis.set_property("ExposureTime", self.exposure_time)
        self.Tis.set_property("Gain", self.gain)
        self.Tis.set_property("GainAuto", "Off")
        self.Tis.set_property("BalanceWhiteAuto", "Off")
    
        BlackLevel = self.Tis.get_property("BlackLevel")
        ExposureAuto = self.Tis.get_property("ExposureAuto")
        ExposureTime = self.Tis.get_property("ExposureTime")
        Gain = self.Tis.get_property("Gain")
        GainAuto = self.Tis.get_property("GainAuto")
        WhiteBalanceAuto = self.Tis.get_property("BalanceWhiteAuto")

        print(f"BlackLevel: {BlackLevel}, ExposureAuto: {ExposureAuto}, ExposureTime: {ExposureTime}, Gain: {Gain}, GainAuto: {GainAuto}")
        print(f"WhiteBalanceAuto: {WhiteBalanceAuto}")

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