import cv2
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
from datetime import datetime
from networkx import jaccard_coefficient
import numpy as np
from sympy import fu
import yaml
import time
import logging

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

from aikensa.camscripts.cam_init import initialize_camera
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard, detectCharucoBoard_scaledImage, calculatecameramatrix, calculatecameramatrix_scaledImage, warpTwoImages, calculateHomography_template, warpTwoImages_template
from aikensa.opencv_imgprocessing.arucoplanarize import planarize, planarize_image, planarize_image_temp
from dataclasses import dataclass, field
from typing import List, Tuple

from aikensa.parts_config.sound import play_do_sound, play_picking_sound, play_re_sound, play_mi_sound, play_alarm_sound, play_konpou_sound, play_keisoku_sound


@dataclass
class CalibrationConfig:
    widget: int = 0
    cameraID: int = -1 # -1 indicates no camera selected
    
    calculateSingeFrameMatrix: bool = False
    calculateCamMatrix: bool = False
    delCamMatrix: bool = False
    savecalculatedCamImage: bool = False

    calibrationMatrix: np.ndarray = field(default=None)
    mapCalculated: list = field(default_factory=lambda: [False]*10) #for 10 cameras
    map1: list = field(default_factory=lambda: [None]*10) #for 10 cameras
    map2: list = field(default_factory=lambda: [None]*10) #for 10 cameras

    calibrationMatrix_scaled: np.ndarray = field(default=None)
    map1_downscaled: list = field(default_factory=lambda: [None]*10) #for 10 cameras
    map2_downscaled: list = field(default_factory=lambda: [None]*10) #for 10 cameras

    calculateHomo_cam1: bool = False
    calculateHomo_cam2: bool = False
    calculateHomo_cam3: bool = False
    calculateHomo_cam4: bool = False
    calculateHomo_cam5: bool = False

    deleteHomo: bool = False

    mergeCam: bool = False
    saveImage: bool = False

    savePlanarize: bool = False
    delPlanarize: bool = False

    savePlanarize_temp: bool = False

    opacity: float = 0.5
    blur: int = 10
    lower_canny: int = 100
    upper_canny: int = 200
    contrast: int = 200
    brightness: int = 0
    savecannyparams: bool = False

    HDRes: bool = False


class CalibrationThread(QThread):

    CalibCamStream = pyqtSignal(QImage)
    CamMerge1 = pyqtSignal(QImage)
    CamMerge2 = pyqtSignal(QImage)
    CamMerge3 = pyqtSignal(QImage)
    CamMerge4 = pyqtSignal(QImage)
    CamMerge5 = pyqtSignal(QImage)
    CamMergeAll = pyqtSignal(QImage)

    def __init__(self, calib_config: CalibrationConfig = None):
        super(CalibrationThread, self).__init__()
        self.running = True

        if calib_config is None:
            self.calib_config = CalibrationConfig()    
        else:
            self.calib_config = calib_config

        self.widget_dir_map={
            1: "CamCalibration1",
            2: "CamCalibration2",
            3: "CamCalibration3",
            4: "CamCalibration4",
            5: "CamCalibration5",
        }


        self.cameraMatrix = None
        self.kanjiFontPath = "aikensa/font/NotoSansJP-ExtraBold.ttf"
        self.cap_cam = None
        self.frame  = None
        self.frame_downsampled = None
        self.frame_scaled = None

        self.multiCam_stream = False
        self.cap_cam1 = None
        self.cap_cam2 = None
        self.cap_cam3 = None
        self.cap_cam4 = None
        self.cap_cam5 = None
        self.mergeframe1 = None
        self.mergeframe2 = None
        self.mergeframe3 = None
        self.mergeframe4 = None
        self.mergeframe5 = None
        self.mergeframe1_scaled = None
        self.mergeframe2_scaled = None
        self.mergeframe3_scaled = None
        self.mergeframe4_scaled = None
        self.mergeframe5_scaled = None
        self.mergeframe1_downsampled = None
        self.mergeframe2_downsampled = None
        self.mergeframe3_downsampled = None
        self.mergeframe4_downsampled = None
        self.mergeframe5_downsampled = None

        self.homography_template = None
        self.homography_matrix1 = None
        self.homography_matrix2 = None
        self.homography_matrix3 = None
        self.homography_matrix4 = None
        self.homography_matrix5 = None
        self.homography_template_scaled = None
        self.homography_matrix1_scaled = None
        self.homography_matrix2_scaled = None
        self.homography_matrix3_scaled = None
        self.homography_matrix4_scaled = None
        self.homography_matrix5_scaled = None
        self.H1 = None
        self.H2 = None
        self.H3 = None
        self.H4 = None
        self.H5 = None
        self.H1_scaled = None
        self.H2_scaled = None
        self.H3_scaled = None
        self.H4_scaled = None
        self.H5_scaled = None


        self.homography_size = None
        self.homography_blank_canvas = None
        self.homography_blank_canvas_scaled = None

        self.combinedImage = None
        self.combinedImage_scaled = None
        self.combinedImage_scaled_temp = None

        self.scale_factor = 5.0
        self.frame_width = 3072
        self.frame_height = 2048
        self.scaled_width = None
        self.scaled_height = None

        self.planarizeTransform = None
        self.planarizeTransform_scaled = None

        self.planarizeTransform_temp = None
        self.planarizeTransform_temp_scaled = None



    def initialize_single_camera(self, camID):
        if self.cap_cam is not None:
            self.cap_cam.release()  # Release the previous camera if it's already open
            print(f"Camera {self.calib_config.cameraID} released.")

        if camID == -1:
            print("No valid camera selected, displaying placeholder.")
            self.cap_cam = None  # No camera initialized
            # self.frame = self.create_placeholder_image()
        else:
            self.cap_cam = initialize_camera(camID)
            print(f"Initialized Camera on ID {camID}")

    def release_all_camera(self):
        if self.cap_cam1 is not None:
            self.cap_cam1.release()
            print(f"Camera 1 released.")
        if self.cap_cam2 is not None:
            self.cap_cam2.release()
            print(f"Camera 2 released.")
        if self.cap_cam3 is not None:
            self.cap_cam3.release()
            print(f"Camera 3 released.")
        if self.cap_cam4 is not None:
            self.cap_cam4.release()
            print(f"Camera 4 released.")
        if self.cap_cam5 is not None:
            self.cap_cam5.release()
            print(f"Camera 5 released.")

    def initialize_all_camera(self):
        if self.cap_cam1 is not None:
            self.cap_cam1.release()
            print(f"Camera 1 released.")
        if self.cap_cam2 is not None:
            self.cap_cam2.release()
            print(f"Camera 2 released.")
        if self.cap_cam3 is not None:
            self.cap_cam3.release()
            print(f"Camera 3 released.")
        if self.cap_cam4 is not None:
            self.cap_cam4.release()
            print(f"Camera 4 released.")
        if self.cap_cam5 is not None:
            self.cap_cam5.release()
            print(f"Camera 5 released.")
        
        self.cap_cam1 = initialize_camera(1)
        self.cap_cam2 = initialize_camera(2)
        self.cap_cam3 = initialize_camera(3)
        self.cap_cam4 = initialize_camera(4)
        self.cap_cam5 = initialize_camera(5)

        if not self.cap_cam1.isOpened():
            print(f"Failed to open camera with ID 1")
            self.cap_cam1 = None
        else:
            print(f"Initialized Camera on ID 1")
        if not self.cap_cam2.isOpened():
            print(f"Failed to open camera with ID 2")
            self.cap_cam2 = None
        else:
            print(f"Initialized Camera on ID 2")
        if not self.cap_cam3.isOpened():
            print(f"Failed to open camera with ID 3")
            self.cap_cam3 = None
        else:
            print(f"Initialized Camera on ID 3")
        if not self.cap_cam4.isOpened():
            print(f"Failed to open camera with ID 4")
            self.cap_cam4 = None
        else:
            print(f"Initialized Camera on ID 4")
        if not self.cap_cam5.isOpened():
            print(f"Failed to open camera with ID 5")
            self.cap_cam5 = None
        else:
            print(f"Initialized Camera on ID 5")

            
    def run(self):

        #print thread started
        print("Calibration Thread Started")

        self.current_cameraID = self.calib_config.cameraID
        self.initialize_single_camera(self.current_cameraID)
        self._save_dir = f"aikensa/cameracalibration/"

        self.homography_template = cv2.imread("aikensa/homography_template/homography_template_border.png")
        self.homography_size = (self.homography_template.shape[0], self.homography_template.shape[1])

        #make dark blank image with same size as homography_template
        self.homography_blank_canvas = np.zeros(self.homography_size, dtype=np.uint8)
        self.homography_blank_canvas = cv2.cvtColor(self.homography_blank_canvas, cv2.COLOR_GRAY2RGB)
        
        self.homography_template_scaled = cv2.resize(self.homography_template, (self.homography_template.shape[1]//5, self.homography_template.shape[0]//5), interpolation=cv2.INTER_LINEAR)
        self.homography_blank_canvas_scaled = cv2.resize(self.homography_blank_canvas, (self.homography_blank_canvas.shape[1]//5, self.homography_blank_canvas.shape[0]//5), interpolation=cv2.INTER_LINEAR)

        self.scaled_height  = int(self.frame_height / self.scale_factor)
        self.scaled_width = int(self.frame_width / self.scale_factor)


        #INIT all variables
        if os.path.exists("./aikensa/cameracalibration/homography_param_cam1.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam1.yaml") as file:
                self.homography_matrix1 = yaml.load(file, Loader=yaml.FullLoader)
                self.H1 = np.array(self.homography_matrix1)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam2.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam2.yaml") as file:
                self.homography_matrix2 = yaml.load(file, Loader=yaml.FullLoader)
                self.H2 = np.array(self.homography_matrix2)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam3.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam3.yaml") as file:
                self.homography_matrix3 = yaml.load(file, Loader=yaml.FullLoader)
                self.H3 = np.array(self.homography_matrix3)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam4.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam4.yaml") as file:
                self.homography_matrix4 = yaml.load(file, Loader=yaml.FullLoader)
                self.H4 = np.array(self.homography_matrix4)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam5.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam5.yaml") as file:
                self.homography_matrix5 = yaml.load(file, Loader=yaml.FullLoader)
                self.H5 = np.array(self.homography_matrix5)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam1_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam1_scaled.yaml") as file:
                self.homography_matrix1_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H1_scaled = np.array(self.homography_matrix1_scaled)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam2_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam2_scaled.yaml") as file:
                self.homography_matrix2_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H2_scaled = np.array(self.homography_matrix2_scaled)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam3_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam3_scaled.yaml") as file:
                self.homography_matrix3_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H3_scaled = np.array(self.homography_matrix3_scaled)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam4_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam4_scaled.yaml") as file:
                self.homography_matrix4_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H4_scaled = np.array(self.homography_matrix4_scaled)

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam5_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam5_scaled.yaml") as file:
                self.homography_matrix5_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H5_scaled = np.array(self.homography_matrix5_scaled)

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform = np.array(transform_list)

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_scaled.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_scaled.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_scaled = np.array(transform_list)

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_temp.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_temp.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_temp = np.array(transform_list)

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_temp_scaled.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_temp_scaled.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_temp_scaled = np.array(transform_list)     

        

        while self.running:

            if self.calib_config.widget == 0:
                self.calib_config.cameraID = -1

            if self.calib_config.widget in [1, 2, 3, 4, 5]:
                
                if self.calib_config.cameraID != self.current_cameraID:
                    # Camera ID has changed, reinitialize the camera
                    if self.current_cameraID != -1:
                        self.cap_cam.release()
                        print(f"Camera {self.current_cameraID} released.")
                    self.current_cameraID = self.calib_config.cameraID
                    self.initialize_single_camera(self.current_cameraID)

                if self.cap_cam is not None:
                    try:
                        ret, self.frame = self.cap_cam.read()
                        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                        self.frame = cv2.rotate(self.frame, cv2.ROTATE_180)

                        self.frame_scaled = cv2.resize(self.frame, (self.scaled_width, self.scaled_height), interpolation=cv2.INTER_LINEAR)

                        if not ret:
                            print("Failed to capture frame")
                            continue
                    except cv2.error as e:
                        print("An error occurred while reading frames from the cameras:", str(e))


                self.calib_config.cameraID = self.calib_config.widget
            
                if self.calib_config.mapCalculated[self.calib_config.cameraID] is False and self.frame is not None:
                    if os.path.exists(self._save_dir + f"Calibration_camera_{self.calib_config.cameraID}.yaml"):
                        camera_matrix, dist_coeffs = self.load_matrix_from_yaml(self._save_dir + f"Calibration_camera_{self.calib_config.cameraID}.yaml")
                        # Precompute the undistort and rectify map for faster processing
                        h, w = self.frame.shape[:2]
                        self.calib_config.map1[self.calib_config.cameraID], self.calib_config.map2[self.calib_config.cameraID] = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2)
                        print(f"map1 and map2 value is calculated for camera {self.calib_config.cameraID}")
                        self.calib_config.mapCalculated[self.calib_config.cameraID] = True
                
                if self.calib_config.mapCalculated[self.calib_config.cameraID] is True:
                    self.frame = cv2.remap(self.frame, self.calib_config.map1[self.calib_config.cameraID], self.calib_config.map2[self.calib_config.cameraID], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                if self.calib_config.calculateSingeFrameMatrix:
                    self.frame, _, _ = detectCharucoBoard(self.frame)
                    self.frame_scaled, _, _ = detectCharucoBoard_scaledImage(self.frame_scaled)
                    self.calib_config.calculateSingeFrameMatrix = False

                if self.calib_config.calculateCamMatrix:
                    self.calib_config.calibrationMatrix = calculatecameramatrix()
                    self.calib_config.calibrationMatrix_scaled = calculatecameramatrix_scaledImage()

                    print(f"Calibration Matrix Value: {self.calib_config.calibrationMatrix}")
                    print(f"Calibration Matrix Scaled Value: {self.calib_config.calibrationMatrix_scaled}")
                    
                    os.makedirs(self._save_dir, exist_ok=True)
                    self.save_calibration_to_yaml(self.calib_config.calibrationMatrix, self._save_dir + f"Calibration_camera_{self.calib_config.cameraID}.yaml")
                    self.save_calibration_to_yaml(self.calib_config.calibrationMatrix_scaled, self._save_dir + f"Calibration_camera_scaled_{self.calib_config.cameraID}.yaml")
                    self.calib_config.calculateCamMatrix = False

                if self.frame is not None:
                    self.frame_downsampled = self.downSampling(self.frame, 1229, 819)

                if self.frame is not None:
                    self.CalibCamStream.emit(self.convertQImage(self.frame_downsampled))
            
            if self.calib_config.widget == 6:
                if self.multiCam_stream is False:
                    self.multiCam_stream = True
                    self.initialize_all_camera()
                    
                _, self.mergeframe1 = self.cap_cam1.read()
                _, self.mergeframe2 = self.cap_cam2.read()
                _, self.mergeframe3 = self.cap_cam3.read()
                _, self.mergeframe4 = self.cap_cam4.read()
                _, self.mergeframe5 = self.cap_cam5.read()

                self.mergeframe1 = cv2.cvtColor(self.mergeframe1, cv2.COLOR_BGR2RGB)
                self.mergeframe2 = cv2.cvtColor(self.mergeframe2, cv2.COLOR_BGR2RGB)
                self.mergeframe3 = cv2.cvtColor(self.mergeframe3, cv2.COLOR_BGR2RGB)
                self.mergeframe4 = cv2.cvtColor(self.mergeframe4, cv2.COLOR_BGR2RGB)
                self.mergeframe5 = cv2.cvtColor(self.mergeframe5, cv2.COLOR_BGR2RGB)
                self.mergeframe1 = cv2.rotate(self.mergeframe1, cv2.ROTATE_180)
                self.mergeframe2 = cv2.rotate(self.mergeframe2, cv2.ROTATE_180)
                self.mergeframe3 = cv2.rotate(self.mergeframe3, cv2.ROTATE_180)
                self.mergeframe4 = cv2.rotate(self.mergeframe4, cv2.ROTATE_180)
                self.mergeframe5 = cv2.rotate(self.mergeframe5, cv2.ROTATE_180)
                #original res
                self.mergeframe1_scaled = self.downSampling(self.mergeframe1, self.scaled_width, self.scaled_height)
                self.mergeframe2_scaled = self.downSampling(self.mergeframe2, self.scaled_width, self.scaled_height)
                self.mergeframe3_scaled = self.downSampling(self.mergeframe3, self.scaled_width, self.scaled_height)
                self.mergeframe4_scaled = self.downSampling(self.mergeframe4, self.scaled_width, self.scaled_height)
                self.mergeframe5_scaled = self.downSampling(self.mergeframe5, self.scaled_width, self.scaled_height)

                #Calculate all map from calibration matrix for 5 cameras, thus i in range(1, 6)
                for i in range(1, 6):
                    if self.calib_config.mapCalculated[i] is False:
                        if os.path.exists(self._save_dir + f"Calibration_camera_{i}.yaml"):
                            camera_matrix, dist_coeffs = self.load_matrix_from_yaml(self._save_dir + f"Calibration_camera_{i}.yaml")
                            # Precompute the undistort and rectify map for faster processing
                            h, w = self.mergeframe1.shape[:2] #use mergeframe1 as reference
                            self.calib_config.map1[i], self.calib_config.map2[i] = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2)
                            print(f"map1 and map2 value is calculated")
                            self.calib_config.mapCalculated[i] = True
                            print(f"Calibration map is calculated for Camera {i}")
                        #do the same for map1_downscaled and map2_downscaled
                        if os.path.exists(self._save_dir + f"Calibration_camera_scaled_{i}.yaml"):
                            camera_matrix, dist_coeffs = self.load_matrix_from_yaml(self._save_dir + f"Calibration_camera_scaled_{i}.yaml")
                            # Precompute the undistort and rectify map for faster processing
                            h, w = self.mergeframe1_scaled.shape[:2] #use mergeframe1scaled as reference
                            self.calib_config.map1_downscaled[i], self.calib_config.map2_downscaled[i] = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2)
                            print(f"map1_downscaled and map2_downscaled value is calculated")
               

                if all(self.calib_config.mapCalculated[i] for i in range(1, 6)):
                    # print("All calibration maps are calculated.")
                    self.mergeframe1 = cv2.remap(self.mergeframe1, self.calib_config.map1[1], self.calib_config.map2[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    self.mergeframe2 = cv2.remap(self.mergeframe2, self.calib_config.map1[2], self.calib_config.map2[2], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    self.mergeframe3 = cv2.remap(self.mergeframe3, self.calib_config.map1[3], self.calib_config.map2[3], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    self.mergeframe4 = cv2.remap(self.mergeframe4, self.calib_config.map1[4], self.calib_config.map2[4], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    self.mergeframe5 = cv2.remap(self.mergeframe5, self.calib_config.map1[5], self.calib_config.map2[5], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                    self.mergeframe1_scaled = cv2.remap(self.mergeframe1_scaled, self.calib_config.map1_downscaled[1], self.calib_config.map2_downscaled[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    self.mergeframe2_scaled = cv2.remap(self.mergeframe2_scaled, self.calib_config.map1_downscaled[2], self.calib_config.map2_downscaled[2], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    self.mergeframe3_scaled = cv2.remap(self.mergeframe3_scaled, self.calib_config.map1_downscaled[3], self.calib_config.map2_downscaled[3], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    self.mergeframe4_scaled = cv2.remap(self.mergeframe4_scaled, self.calib_config.map1_downscaled[4], self.calib_config.map2_downscaled[4], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    self.mergeframe5_scaled = cv2.remap(self.mergeframe5_scaled, self.calib_config.map1_downscaled[5], self.calib_config.map2_downscaled[5], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                #Calculate Homography matrix
                if self.calib_config.calculateHomo_cam1 is True:
                    self.calib_config.calculateHomo_cam1 = False
                    if self.mergeframe1 is not None:
                        _, self.homography_matrix1 = calculateHomography_template(self.homography_template, self.mergeframe1)
                        self.H1 = np.array(self.homography_matrix1)
                        print(f"Homography matrix is calculated for Camera 1 with value {self.homography_matrix1}")
                        os.makedirs(self._save_dir, exist_ok=True)
                        with open("./aikensa/cameracalibration/homography_param_cam1.yaml", "w") as file:
                            yaml.dump(self.homography_matrix1.tolist(), file)
                    else:
                        ("mergeframe1 is empty")

                    if self.mergeframe1_scaled is not None:
                        _, self.homography_matrix1_scaled = calculateHomography_template(self.homography_template_scaled, self.mergeframe1_scaled)
                        self.H1_scaled = np.array(self.homography_matrix1_scaled)
                        print(f"Homography scaled matrix is calculated for Camera 1 with value {self.homography_matrix1_scaled}")
                        with open("./aikensa/cameracalibration/homography_param_cam1_scaled.yaml", "w") as file:
                            yaml.dump(self.homography_matrix1_scaled.tolist(), file)
                    else:
                        ("mergeframe1 scaled is empty")

                if self.calib_config.calculateHomo_cam2 is True:
                    self.calib_config.calculateHomo_cam2 = False
                    _, self.homography_matrix2 = calculateHomography_template(self.homography_template, self.mergeframe2)
                    self.H2 = np.array(self.homography_matrix2)
                    print(f"Homography matrix is calculated for Camera 2 with value {self.homography_matrix2}")
                    os.makedirs(self._save_dir, exist_ok=True)
                    with open("./aikensa/cameracalibration/homography_param_cam2.yaml", "w") as file:
                        yaml.dump(self.homography_matrix2.tolist(), file)

                    _, self.homography_matrix2_scaled = calculateHomography_template(self.homography_template_scaled, self.mergeframe2_scaled)
                    self.H2_scaled = np.array(self.homography_matrix2_scaled)
                    print(f"Homography scaled matrix is calculated for Camera 2 with value {self.homography_matrix2_scaled}")
                    with open("./aikensa/cameracalibration/homography_param_cam2_scaled.yaml", "w") as file:
                        yaml.dump(self.homography_matrix2_scaled.tolist(), file) 

                if self.calib_config.calculateHomo_cam3 is True:
                    self.calib_config.calculateHomo_cam3 = False
                    _, self.homography_matrix3 = calculateHomography_template(self.homography_template, self.mergeframe3)
                    self.H3 = np.array(self.homography_matrix3)
                    print(f"Homography matrix is calculated for Camera 3 with value {self.homography_matrix3}")
                    os.makedirs(self._save_dir, exist_ok=True)
                    with open("./aikensa/cameracalibration/homography_param_cam3.yaml", "w") as file:
                        yaml.dump(self.homography_matrix3.tolist(), file)

                    _, self.homography_matrix3_scaled = calculateHomography_template(self.homography_template_scaled, self.mergeframe3_scaled)
                    self.H3_scaled = np.array(self.homography_matrix3_scaled)
                    print(f"Homography scaled matrix is calculated for Camera 3 with value {self.homography_matrix3_scaled}")
                    with open("./aikensa/cameracalibration/homography_param_cam3_scaled.yaml", "w") as file:
                        yaml.dump(self.homography_matrix3_scaled.tolist(), file)
    
                if self.calib_config.calculateHomo_cam4 is True:
                    self.calib_config.calculateHomo_cam4 = False
                    _, self.homography_matrix4 = calculateHomography_template(self.homography_template, self.mergeframe4)
                    self.H4 = np.array(self.homography_matrix4)
                    print(f"Homography matrix is calculated for Camera 4 with value {self.homography_matrix4}")
                    os.makedirs(self._save_dir, exist_ok=True)
                    with open("./aikensa/cameracalibration/homography_param_cam4.yaml", "w") as file:
                        yaml.dump(self.homography_matrix4.tolist(), file)

                    _, self.homography_matrix4_scaled = calculateHomography_template(self.homography_template_scaled, self.mergeframe4_scaled)
                    self.H4_scaled = np.array(self.homography_matrix4_scaled)
                    print(f"Homography scaled matrix is calculated for Camera 4 with value {self.homography_matrix4_scaled}")
                    with open("./aikensa/cameracalibration/homography_param_cam4_scaled.yaml", "w") as file:
                        yaml.dump(self.homography_matrix4_scaled.tolist(), file)

                if self.calib_config.calculateHomo_cam5 is True:
                    self.calib_config.calculateHomo_cam5 = False
                    _, self.homography_matrix5 = calculateHomography_template(self.homography_template, self.mergeframe5)
                    self.H5 = np.array(self.homography_matrix5)
                    print(f"Homography matrix is calculated for Camera 5 with value {self.homography_matrix5}")
                    os.makedirs(self._save_dir, exist_ok=True)
                    with open("./aikensa/cameracalibration/homography_param_cam5.yaml", "w") as file:
                        yaml.dump(self.homography_matrix5.tolist(), file)

                    _, self.homography_matrix5_scaled = calculateHomography_template(self.homography_template_scaled, self.mergeframe5_scaled)
                    self.H5_scaled = np.array(self.homography_matrix5_scaled)
                    print(f"Homography scaled matrix is calculated for Camera 5 with value {self.homography_matrix5_scaled}")
                    with open("./aikensa/cameracalibration/homography_param_cam5_scaled.yaml", "w") as file:
                        yaml.dump(self.homography_matrix5_scaled.tolist(), file)


                if self.H1 is None:
                    print("H1 is None")
                    if os.path.exists("./aikensa/cameracalibration/homography_param_cam1.yaml"):
                        with open("./aikensa/cameracalibration/homography_param_cam1.yaml") as file:
                            self.homography_matrix1 = yaml.load(file, Loader=yaml.FullLoader)
                        self.H1 = np.array(self.homography_matrix1)
                if self.H2 is None:
                    print("H2 is None")  
                    if os.path.exists("./aikensa/cameracalibration/homography_param_cam2.yaml"):
                        with open("./aikensa/cameracalibration/homography_param_cam2.yaml") as file:
                            self.homography_matrix2 = yaml.load(file, Loader=yaml.FullLoader)
                            self.H2 = np.array(self.homography_matrix2)
                if self.H3 is None:
                    print("H3 is none")
                    if os.path.exists("./aikensa/cameracalibration/homography_param_cam3.yaml"):
                        with open("./aikensa/cameracalibration/homography_param_cam3.yaml") as file:
                            self.homography_matrix3 = yaml.load(file, Loader=yaml.FullLoader)
                            self.H3 = np.array(self.homography_matrix3)
                if self.H4 is None:
                    print("H4 is none")
                    if os.path.exists("./aikensa/cameracalibration/homography_param_cam4.yaml"):
                        with open("./aikensa/cameracalibration/homography_param_cam4.yaml") as file:
                            self.homography_matrix4 = yaml.load(file, Loader=yaml.FullLoader)
                            self.H4 = np.array(self.homography_matrix4)
                if self.H5 is None:
                    print("H5 is None")
                    if os.path.exists("./aikensa/cameracalibration/homography_param_cam5.yaml"):
                        with open("./aikensa/cameracalibration/homography_param_cam5.yaml") as file:
                            self.homography_matrix5 = yaml.load(file, Loader=yaml.FullLoader)
                            self.H5 = np.array(self.homography_matrix5)


                if self.H1_scaled is None:
                    print("H1_scaled is None")
                    if os.path.exists("./aikensa/cameracalibration/homography_param_cam1_scaled.yaml"):
                        with open("./aikensa/cameracalibration/homography_param_cam1_scaled.yaml") as file:
                            self.homography_matrix1_scaled = yaml.load(file, Loader=yaml.FullLoader)
                        self.H1_scaled = np.array(self.homography_matrix1_scaled)
                if self.H2_scaled is None:
                    print("H2_scaled is None")  
                    if os.path.exists("./aikensa/cameracalibration/homography_param_cam2_scaled.yaml"):
                        with open("./aikensa/cameracalibration/homography_param_cam2_scaled.yaml") as file:
                            self.homography_matrix2_scaled = yaml.load(file, Loader=yaml.FullLoader)
                            self.H2_scaled = np.array(self.homography_matrix2_scaled)
                if self.H3_scaled is None:
                    print("H3_scaled is none")
                    if os.path.exists("./aikensa/cameracalibration/homography_param_cam3_scaled.yaml"):
                        with open("./aikensa/cameracalibration/homography_param_cam3_scaled.yaml") as file:
                            self.homography_matrix3_scaled = yaml.load(file, Loader=yaml.FullLoader)
                            self.H3_scaled = np.array(self.homography_matrix3_scaled)
                if self.H4_scaled is None:
                    print("H4_scaled is none")
                    if os.path.exists("./aikensa/cameracalibration/homography_param_cam4_scaled.yaml"):
                        with open("./aikensa/cameracalibration/homography_param_cam4_scaled.yaml") as file:
                            self.homography_matrix4_scaled = yaml.load(file, Loader=yaml.FullLoader)
                            self.H4_scaled = np.array(self.homography_matrix4_scaled)
                if self.H5_scaled is None:
                    print("H5_scaled is None")
                    if os.path.exists("./aikensa/cameracalibration/homography_param_cam5_scaled.yaml"):
                        with open("./aikensa/cameracalibration/homography_param_cam5_scaled.yaml") as file:
                            self.homography_matrix5_scaled = yaml.load(file, Loader=yaml.FullLoader)
                            self.H5_scaled = np.array(self.homography_matrix5_scaled)

                self.combinedImage = warpTwoImages_template(self.homography_blank_canvas, self.mergeframe1, self.H1)
                self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe2, self.H2)
                self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe3, self.H3)
                self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe4, self.H4)
                self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe5, self.H5)

                self.combinedImage_scaled = warpTwoImages_template(self.homography_blank_canvas_scaled, self.mergeframe1_scaled, self.H1_scaled)
                self.combinedImage_scaled = warpTwoImages_template(self.combinedImage_scaled, self.mergeframe2_scaled, self.H2_scaled)
                self.combinedImage_scaled = warpTwoImages_template(self.combinedImage_scaled, self.mergeframe3_scaled, self.H3_scaled)
                self.combinedImage_scaled = warpTwoImages_template(self.combinedImage_scaled, self.mergeframe4_scaled, self.H4_scaled)
                self.combinedImage_scaled = warpTwoImages_template(self.combinedImage_scaled, self.mergeframe5_scaled, self.H5_scaled)

                # cv2.imwrite("combinedImage_scaled.png", self.combinedImage_scaled)
                combined_image_copy = self.combinedImage.copy()
                #bgr to rgb
                combined_image_copy = cv2.cvtColor(combined_image_copy, cv2.COLOR_BGR2RGB)
                cv2.imwrite("combinedImage.png", combined_image_copy)

                if self.calib_config.savePlanarize is True:
                    self.calib_config.savePlanarize = False
                    self.combinedImage, self.planarizeTransform = planarize_image(self.combinedImage, 
                                                                                  target_width=self.homography_size[1], target_height=self.homography_size[0], 
                                                                                  top_offset=250, bottom_offset=250)
                    self.combinedImage_scaled, self.planarizeTransform_scaled = planarize_image(self.combinedImage_scaled,
                                                                                                  target_width=int(self.homography_size[1]/self.scale_factor), target_height=int(self.homography_size[0]/self.scale_factor),
                                                                                                  top_offset=int(250/self.scale_factor), bottom_offset=int(250/self.scale_factor))
                    os.makedirs(self._save_dir, exist_ok=True)
                    with open("./aikensa/cameracalibration/planarizeTransform.yaml", "w") as file:  
                        yaml.dump(self.planarizeTransform.tolist(), file)
                    with open("./aikensa/cameracalibration/planarizeTransform_scaled.yaml", "w") as file:
                        yaml.dump(self.planarizeTransform_scaled.tolist(), file)

                    print(f"Image size after warping is {self.combinedImage.shape}")
                    # cv2.imwrite("combinedImage.png", self.combinedImage)
                    # cv2.imwrite("combinedImage_scaled.png", self.combinedImage_scaled)

                if self.calib_config.savePlanarize_temp is True:
                    self.calib_config.savePlanarize_temp = False
                    self.combinedImage, self.planarizeTransform_temp = planarize_image_temp(self.combinedImage, 
                                                                                  target_width=self.homography_size[1], target_height=self.homography_size[0], 
                                                                                  top_offset=0, bottom_offset=0)
                    
                    with open("./aikensa/cameracalibration/planarizeTransform_temp.yaml", "w") as file:  
                        yaml.dump(self.planarizeTransform_temp.tolist(), file)
                                  
                    self.combinedImage_scaled, self.planarizeTransform_temp_scaled = planarize_image_temp(self.combinedImage_scaled,
                                                                                                  target_width=int(self.homography_size[1]/self.scale_factor), target_height=int(self.homography_size[0]/self.scale_factor),
                                                                                                  top_offset=0, bottom_offset=0)

                    with open("./aikensa/cameracalibration/planarizeTransform_temp_scaled.yaml", "w") as file:
                        yaml.dump(self.planarizeTransform_temp_scaled.tolist(), file)

                    cv2.imwrite("combinedImage_temp.png", self.combinedImage_scaled)
                    print(f"Planarize Transform Temp: {self.planarizeTransform_temp}")
                    print(f"Planarize Transform Temp Scaled: {self.planarizeTransform_temp_scaled}")

                    os.makedirs(self._save_dir, exist_ok=True)


                    
                    # cv2.imwrite("combinedImage_scaled_temp.png", self.combinedImage_scaled)

                # if self.planarizeTransform is not None:
                #     self.combinedImage = cv2.warpPerspective(self.combinedImage, self.planarizeTransform, (self.homography_size[1],self.homography_size[0]))


                self.combinedImage_scaled_temp = self.combinedImage_scaled.copy()

                if self.planarizeTransform_scaled is not None:
                    self.combinedImage_scaled = cv2.warpPerspective(self.combinedImage_scaled, self.planarizeTransform_scaled, (int(self.homography_size[1]/self.scale_factor), int(self.homography_size[0]/self.scale_factor)))


                if self.planarizeTransform_temp_scaled is not None:
                    self.combinedImage_scaled_temp = cv2.warpPerspective(self.combinedImage_scaled_temp, self.planarizeTransform_temp_scaled, (int(self.homography_size[1]/self.scale_factor), int(self.homography_size[0]/self.scale_factor)))

                # cv2.imwrite("combinedImage_scaled_temp.png", self.combinedImage_scaled_temp)

                # self.combinedImage = cv2.resize(self.combinedImage, (self.homography_size[1], int(self.homography_size[0]/1.26)))
                self.combinedImage_scaled = cv2.resize(self.combinedImage_scaled, (int(self.homography_size[1]/(self.scale_factor*1.48)), int(self.homography_size[0]/(self.scale_factor*1.26*1.48))))#1.48 for the qt, 1.26 for the aspect ratio

                self.mergeframe1_downsampled = self.downSampling(self.mergeframe1, 246, 163)
                self.mergeframe2_downsampled = self.downSampling(self.mergeframe2, 246, 163)
                self.mergeframe3_downsampled = self.downSampling(self.mergeframe3, 246, 163)
                self.mergeframe4_downsampled = self.downSampling(self.mergeframe4, 246, 163)
                self.mergeframe5_downsampled = self.downSampling(self.mergeframe5, 246, 163)

                if self.mergeframe1_downsampled is not None:
                    self.CamMerge1.emit(self.convertQImage(self.mergeframe1_downsampled))
                if self.mergeframe2_downsampled is not None:
                    self.CamMerge2.emit(self.convertQImage(self.mergeframe2_downsampled))
                if self.mergeframe3_downsampled is not None:
                    self.CamMerge3.emit(self.convertQImage(self.mergeframe3_downsampled))
                if self.mergeframe4_downsampled is not None:
                    self.CamMerge4.emit(self.convertQImage(self.mergeframe4_downsampled))
                if self.mergeframe5_downsampled is not None:
                    self.CamMerge5.emit(self.convertQImage(self.mergeframe5_downsampled))
                if self.combinedImage_scaled is not None:
                    self.CamMergeAll.emit(self.convertQImage(self.combinedImage_scaled))

            #wait for 5ms
            self.msleep(2)
            
        print(f"Camera {self.calib_config.cameraID} released.")

    def create_placeholder_image(self):
        # Create a small black image with a white dot in the center
        size = 100
        placeholder = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.circle(placeholder, (size // 2, size // 2), 10, (255, 255, 255), -1)
        return placeholder

    def convertQImage(self, image):
        # Convert resized cv2 image to QImage
        h, w, ch = image.shape
        bytesPerLine = ch * w
        processed_image = QImage(image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return processed_image

    def undistortFrame(self, frame,cameraMatrix, distortionCoeff):
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.undistort(frame, cameraMatrix, distortionCoeff, None, cameraMatrix)
        return frame

    def stop(self):
        self.running = False
        self.release_all_camera()
        print("Calibration thread stopped.")
    
    def downSampling(self, image, width=384, height=256):
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        return resized_image

    def save_calibration_to_yaml(self, calibrationMatrix, filename):
        with open(filename, 'w') as file:
            yaml.dump(calibrationMatrix, file)

    def load_matrix_from_yaml(self, filename):
        with open(filename, 'r') as file:
            calibration_param = yaml.load(file, Loader=yaml.FullLoader)
            camera_matrix = np.array(calibration_param.get('camera_matrix'))
            distortion_coeff = np.array(calibration_param.get('distortion_coefficients'))
        return camera_matrix, distortion_coeff
        
    def initialize_maps(camera_matrix, dist_coeffs, image_size):
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, camera_matrix, image_size, cv2.CV_16SC2)
        return map1, map2