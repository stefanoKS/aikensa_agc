import cv2
import os
from datetime import datetime
from networkx import jaccard_coefficient
import numpy as np
from sympy import fu
import yaml
import time
import logging
import sqlite3
import mysql.connector

from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

from aikensa.camscripts.cam_init import initialize_camera
from aikensa.camscripts.cam_hole_init import initialize_hole_camera
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard , calculatecameramatrix, warpTwoImages, calculateHomography_template, warpTwoImages_template
from aikensa.opencv_imgprocessing.arucoplanarize import planarize, planarize_image
from dataclasses import dataclass, field
from typing import List, Tuple

from aikensa.parts_config.sound import play_do_sound, play_picking_sound, play_re_sound, play_mi_sound, play_alarm_sound, play_konpou_sound, play_keisoku_sound

from ultralytics import YOLO
from aikensa.parts_config.hoodFR_65820W030P import partcheck
from aikensa.parts_config.P658207YA0A_SEALASSYHOODFR import partcheck as P658207YA0A_partcheck

from PIL import ImageFont, ImageDraw, Image

@dataclass
class InspectionConfig:
    widget: int = 0
    cameraID: int = -1 # -1 indicates no camera selected

    mapCalculated: list = field(default_factory=lambda: [False]*10) #for 10 cameras
    map1: list = field(default_factory=lambda: [None]*10) #for 10 cameras
    map2: list = field(default_factory=lambda: [None]*10) #for 10 cameras

    map1_downscaled: list = field(default_factory=lambda: [None]*10) #for 10 cameras
    map2_downscaled: list = field(default_factory=lambda: [None]*10) #for 10 cameras

    doInspection: bool = False

    kouden_sensor: list =  field(default_factory=lambda: [0]*5)
    button_sensor: int = 0

    kensainNumber: str = None
    furyou_plus: bool = False
    furyou_minus: bool = False
    kansei_plus: bool = False
    kansei_minus: bool = False
    furyou_plus_10: bool = False #to add 10
    furyou_minus_10: bool = False
    kansei_plus_10: bool = False
    kansei_minus_10: bool = False

    counterReset: bool = False

    today_numofPart: list = field(default_factory=lambda: [[0, 0] for _ in range(30)])
    current_numofPart: list = field(default_factory=lambda: [[0, 0] for _ in range(30)])



class InspectionThread(QThread):

    part1Cam = pyqtSignal(QImage)
    part2Cam = pyqtSignal(QImage)
    part3Cam = pyqtSignal(QImage)
    part4Cam = pyqtSignal(QImage)
    part5Cam = pyqtSignal(QImage)

    hole1Cam = pyqtSignal(QImage)
    hole2Cam = pyqtSignal(QImage)
    hole3Cam = pyqtSignal(QImage)
    hole4Cam = pyqtSignal(QImage)
    hole5Cam = pyqtSignal(QImage)

    dailytenkenCam = pyqtSignal(QImage)

    today_numofPart_signal = pyqtSignal(list)
    current_numofPart_signal = pyqtSignal(list)
    
    hoodFR_InspectionResult_PitchMeasured = pyqtSignal(list)
    P8462284S00_InspectionResult_PitchMeasured = pyqtSignal(list)

    hoodFR_InspectionResult_PitchResult = pyqtSignal(list)
    P8462284S00_InspectionResult_PitchResult = pyqtSignal(list)
    
    hoodFR_InspectionStatus = pyqtSignal(list)
    P8462284S00_InspectionStatus = pyqtSignal(list)

    hoodFR_HoleStatus = pyqtSignal(list)

    ethernet_status_red_tenmetsu = pyqtSignal(list)
    ethernet_status_green_hold = pyqtSignal(list)
    ethernet_status_red_hold = pyqtSignal(list)


    

    def __init__(self, inspection_config: InspectionConfig = None):
        super(InspectionThread, self).__init__()
        self.running = True

        if inspection_config is None:
            self.inspection_config = InspectionConfig()    
        else:
            self.inspection_config = inspection_config

        #Initialize AI checkpoint


        self.kanjiFontPath = "aikensa/font/NotoSansJP-ExtraBold.ttf"

        self.multiCam_stream = False

        self.cap_cam = None
        self.cap_cam0 = None
        self.cap_cam1 = None
        self.cap_cam2 = None
        self.cap_cam3 = None
        self.cap_cam4 = None
        self.cap_cam5 = None

        self.bottomframe = None
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

        self.holeFrame1 = None
        self.holeFrame2 = None
        self.holeFrame3 = None
        self.holeFrame4 = None
        self.holeFrame5 = None

        self.holeImageMerge = [None]*5

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

        self.part1Crop = None
        self.part2Crop = None
        self.part3Crop = None
        self.part4Crop = None
        self.part5Crop = None
        
        self.part1Crop_scaled = None
        self.part2Crop_scaled = None
        self.part3Crop_scaled = None
        self.part4Crop_scaled = None
        self.part5Crop_scaled = None

        self.homography_size = None
        self.homography_size_scaled = None
        self.homography_blank_canvas = None
        self.homography_blank_canvas_scaled = None

        self.combinedImage = None
        self.combinedImage_scaled = None

        self.scale_factor = 5.0 #Scale Factor, might increase this later
        self.scale_factor_hole = 2.0
        self.frame_width = 3072
        self.frame_height = 2048
        self.scaled_width = None
        self.scaled_height = None

        self.planarizeTransform = None
        self.planarizeTransform_scaled = None

        self.planarizeTransform_temp = None
        self.planarizeTransform_temp_scaled = None
        
        self.scaled_height  = int(self.frame_height / self.scale_factor)
        self.scaled_width = int(self.frame_width / self.scale_factor)

        self.part_height_offset = 110
        self.part_height_offset_scaled = int(self.part_height_offset//self.scale_factor)

        self.part_height_offset_hoodFR = 140
        self.part_height_offset_hoodFR_scaled = int(self.part_height_offset_hoodFR//self.scale_factor)

        self.part_height_offset_nissanhoodFR = 180
        self.part_height_offset_nissanhoodFR_scaled = int(self.part_height_offset_nissanhoodFR//self.scale_factor)

        self.dailyTenken_cropWidth = 950
        self.dailyTenken_cropWidth_scaled = int(self.dailyTenken_cropWidth//self.scale_factor)

        self.part1Crop_YPos = 15
        self.part2Crop_YPos = 290
        self.part3Crop_YPos = 580
        self.part4Crop_YPos = 870
        self.part5Crop_YPos = 1150


        #nissan HOOD FR
        
        self.part1Crop_YPos_hoodFR = 25*5
        self.part2Crop_YPos_hoodFR = 67*5
        self.part3Crop_YPos_hoodFR = 109*5
        self.part4Crop_YPos_hoodFR = 151*5
        self.part5Crop_YPos_hoodFR = 193*5

        self.part1Crop_YPos_scaled = int(self.part1Crop_YPos//self.scale_factor)
        self.part2Crop_YPos_scaled = int(self.part2Crop_YPos//self.scale_factor)
        self.part3Crop_YPos_scaled = int(self.part3Crop_YPos//self.scale_factor)
        self.part4Crop_YPos_scaled = int(self.part4Crop_YPos//self.scale_factor)
        self.part5Crop_YPos_scaled = int(self.part5Crop_YPos//self.scale_factor)

        self.part1Crop_Ypos_hoodFR_scaled = int(self.part1Crop_YPos_hoodFR//self.scale_factor)
        self.part2Crop_Ypos_hoodFR_scaled = int(self.part2Crop_YPos_hoodFR//self.scale_factor)
        self.part3Crop_Ypos_hoodFR_scaled = int(self.part3Crop_YPos_hoodFR//self.scale_factor)
        self.part4Crop_Ypos_hoodFR_scaled = int(self.part4Crop_YPos_hoodFR//self.scale_factor)
        self.part5Crop_Ypos_hoodFR_scaled = int(self.part5Crop_YPos_hoodFR//self.scale_factor)
        
        self.hole1Crop_XYpos_scaled = (int(70//self.scale_factor_hole), int(180//self.scale_factor_hole))
        self.hole2Crop_XYpos_scaled = (int(295//self.scale_factor_hole), int(180//self.scale_factor_hole))
        self.hole3Crop_XYpos_scaled = (int(525//self.scale_factor_hole), int(180//self.scale_factor_hole))
        self.hole4Crop_XYpos_scaled = (int(760//self.scale_factor_hole), int(180//self.scale_factor_hole))
        self.hole5Crop_XYpos_scaled = (int(980//self.scale_factor_hole), int(180//self.scale_factor_hole))

        self.height_hole_offset = int(120//self.scale_factor_hole)
        self.width_hole_offset = int(370//self.scale_factor_hole)

        self.timerStart = None
        self.timerFinish = None
        self.fps = None

        self.timerStart_mini = None
        self.timerFinish_mini = None
        self.fps_mini = None

        self.InspectionImages = [None]*5

        self.InspectionImages_endSegmentation_Left = [None]*5
        self.InspectionImages_endSegmentation_Right = [None]*5
        
        self.InspectionResult_ClipDetection = [None]*5

        self.InspectionResult_EndSegmentation_Left = [None]*5
        self.InspectionResult_EndSegmentation_Right = [None]*5

        self.InspectionResult_HoleDetection = [None]*5

        self.InspectionResult_PitchMeasured = [None]*5
        self.InspectionResult_PitchResult = [None]*5
        self.InspectionResult_DetectionID = [None]*5
        self.InspectionResult_Status = [None]*5
        self.InspectionResult_DeltaPitch = [None]*30
        self.InspectionResult_NGReason = [None]*5
        
        self.InspectionStatus = [None]*5

        self.DetectionResult_HoleDetection = [None]*5

        self.ethernet_status_red_tenmetsu_status = [0]*5
        self.ethernet_status_green_hold_status = [0]*5
        self.ethernet_status_red_hold_status = [0]*5

        self.ethernet_status_red_tenmetsu_status_prev = [0]*5
        self.ethernet_status_green_hold_status_prev = [0]*5
        self.ethernet_status_red_hold_status_prev = [0]*5

        self.InspectionImages_prev = [None]*5
        self.hoodFR_InspectionResult_PitchMeasured_prev = None
        self.hoodFR_InspectionResult_PitchResult_prev =None
        self.hoodFR_InspectionStatus_prev = None

        self._test = [0]*5

        self.hole_detection = [None] * 5
        self.hole_detection_result = [None] * 5

        self.widget_dir_map = {
            8: "65820W030P",
            9: "658207YA0A",
        }

        self.InspectionWaitTime = 15.0
        self.InspectionTimeStart = None

        self.test = 0
        self.firstTimeInspection = True

                # "Read mysql id and password from yaml file"
        with open("aikensa/mysql/id.yaml") as file:
            credentials = yaml.load(file, Loader=yaml.FullLoader)
            self.mysqlID = credentials["id"]
            self.mysqlPassword = credentials["pass"]
            self.mysqlHost = credentials["host"]
            self.mysqlHostPort = credentials["port"]


    def release_all_camera(self):
        if self.cap_cam0 is not None:
            self.cap_cam0.release()
            print(f"Camera 0 released.")
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

    def initialize_single_camera(self, camID):
        if self.cap_cam is not None:
            self.cap_cam.release()  # Release the previous camera if it's already open
            print(f"Camera {self.inspection_config.cameraID} released.")

        if camID == -1:
            print("No valid camera selected, displaying placeholder.")
            self.cap_cam = None  # No camera initialized
            # self.frame = self.create_placeholder_image()
        else:
            self.cap_cam = initialize_camera(camID)
            if not self.cap_cam.isOpened():
                print(f"Failed to open camera with ID {camID}")
                self.cap_cam = None
            else:
                print(f"Initialized Camera on ID {camID}")

    def initialize_all_camera(self):
        if self.cap_cam0 is not None:
            self.cap_cam0.release()
            print(f"Camera 0 released.")
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
        
        self.cap_cam0 = initialize_hole_camera(0)
        self.cap_cam1 = initialize_camera(1)
        self.cap_cam2 = initialize_camera(2)
        self.cap_cam3 = initialize_camera(3)
        self.cap_cam4 = initialize_camera(4)
        self.cap_cam5 = initialize_camera(5)

        if not self.cap_cam0.isOpened():
            print(f"Failed to open camera with ID 0")
            self.cap_cam0 = None
        else:
            print(f"Initialized Camera on ID 0")
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
        #initialize the database
        if not os.path.exists("./aikensa/inspection_results"):
            os.makedirs("./aikensa/inspection_results")

        self.conn = sqlite3.connect('./aikensa/inspection_results/database_results.db')
        self.cursor = self.conn.cursor()
        # Create the table if it doesn't exist
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS inspection_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            partName TEXT,
            numofPart TEXT,
            currentnumofPart TEXT,
            timestampHour TEXT,
            timestampDate TEXT,
            deltaTime REAL,
            kensainName TEXT,
            detected_pitch TEXT,
            delta_pitch TEXT,
            total_length REAL
        )
        ''')

        # List of columns to add
        columns_to_add = [
            ("resultpitch", "TEXT"),
            ("status", "TEXT"),
            ("NGreason", "TEXT")
        ]

        # Using the function to add columns
        self.add_columns(self.cursor, "inspection_results", columns_to_add)

        self.conn.commit()


        #Initialize connection to mysql server if available
        try:
            self.mysql_conn = mysql.connector.connect(
                host=self.mysqlHost,
                user=self.mysqlID,
                password=self.mysqlPassword,
                port=self.mysqlHostPort,
                database="AIKENSAresults"
            )
            print(f"Connected to MySQL database at {self.mysqlHost}")
        except Exception as e:
            print(f"Error connecting to MySQL database: {e}")
            self.mysql_conn = None

        #try adding data to the schema in mysql
        if self.mysql_conn is not None:
            self.mysql_cursor = self.mysql_conn.cursor()
            self.mysql_cursor.execute('''
            CREATE TABLE IF NOT EXISTS inspection_results (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                partName TEXT,
                numofPart TEXT,
                currentnumofPart TEXT,
                timestampHour TEXT,
                timestampDate TEXT,
                deltaTime REAL,
                kensainName TEXT,
                detected_pitch TEXT,
                delta_pitch TEXT,
                total_length REAL,
                resultpitch TEXT,
                status TEXT,
                NGreason TEXT
            )
            ''')
            self.mysql_conn.commit()




        #print thread started
        print("Inspection Thread Started")
        self.initialize_model()
        print("AI Model Initialized")

        self.current_cameraID = self.inspection_config.cameraID
        self.initialize_single_camera(self.current_cameraID)
        self._save_dir = f"aikensa/cameracalibration/"

        self.homography_template = cv2.imread("aikensa/homography_template/homography_template_border.png")
        self.homography_size = (self.homography_template.shape[0], self.homography_template.shape[1])
        self.homography_size_scaled = (self.homography_template.shape[0]//5, self.homography_template.shape[1]//5)

        #make dark blank image with same size as homography_template
        self.homography_blank_canvas = np.zeros(self.homography_size, dtype=np.uint8)
        self.homography_blank_canvas = cv2.cvtColor(self.homography_blank_canvas, cv2.COLOR_GRAY2RGB)
        
        self.homography_template_scaled = cv2.resize(self.homography_template, (self.homography_template.shape[1]//5, self.homography_template.shape[0]//5), interpolation=cv2.INTER_LINEAR)
        self.homography_blank_canvas_scaled = cv2.resize(self.homography_blank_canvas, (self.homography_blank_canvas.shape[1]//5, self.homography_blank_canvas.shape[0]//5), interpolation=cv2.INTER_LINEAR)

        for key, value in self.widget_dir_map.items():
            self.inspection_config.current_numofPart[key] = self.get_last_entry_currentnumofPart(value)
            self.inspection_config.today_numofPart[key] = self.get_last_entry_total_numofPart(value)


        #INIT all variables

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam1.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam1.yaml") as file:
                self.homography_matrix1 = yaml.load(file, Loader=yaml.FullLoader)
                self.H1 = np.array(self.homography_matrix1)
                print(f"Loaded homography matrix for camera 1")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam2.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam2.yaml") as file:
                self.homography_matrix2 = yaml.load(file, Loader=yaml.FullLoader)
                self.H2 = np.array(self.homography_matrix2)
                print(f"Loaded homography matrix for camera 2")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam3.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam3.yaml") as file:
                self.homography_matrix3 = yaml.load(file, Loader=yaml.FullLoader)
                self.H3 = np.array(self.homography_matrix3)
                print(f"Loaded homography matrix for camera 3")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam4.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam4.yaml") as file:
                self.homography_matrix4 = yaml.load(file, Loader=yaml.FullLoader)
                self.H4 = np.array(self.homography_matrix4)
                print(f"Loaded homography matrix for camera 4")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam5.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam5.yaml") as file:
                self.homography_matrix5 = yaml.load(file, Loader=yaml.FullLoader)
                self.H5 = np.array(self.homography_matrix5)
                print(f"Loaded homography matrix for camera 5")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam1_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam1_scaled.yaml") as file:
                self.homography_matrix1_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H1_scaled = np.array(self.homography_matrix1_scaled)
                print(f"Loaded scaled homography matrix for camera 1")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam2_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam2_scaled.yaml") as file:
                self.homography_matrix2_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H2_scaled = np.array(self.homography_matrix2_scaled)
                print(f"Loaded scaled homography matrix for camera 2")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam3_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam3_scaled.yaml") as file:
                self.homography_matrix3_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H3_scaled = np.array(self.homography_matrix3_scaled)
                print(f"Loaded scaled homography matrix for camera 3")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam4_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam4_scaled.yaml") as file:
                self.homography_matrix4_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H4_scaled = np.array(self.homography_matrix4_scaled)
                print(f"Loaded scaled homography matrix for camera 4")

        if os.path.exists("./aikensa/cameracalibration/homography_param_cam5_scaled.yaml"):
            with open("./aikensa/cameracalibration/homography_param_cam5_scaled.yaml") as file:
                self.homography_matrix5_scaled = yaml.load(file, Loader=yaml.FullLoader)
                self.H5_scaled = np.array(self.homography_matrix5_scaled)
                print(f"Loaded scaled homography matrix for camera 5")

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform = np.array(transform_list)
                print(f"Loaded planarizeTransform matrix")

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_scaled.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_scaled.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_scaled = np.array(transform_list)
                print(f"Loaded scaled planarizeTransform matrix")

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_temp.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_temp.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_temp = np.array(transform_list)

        if os.path.exists("./aikensa/cameracalibration/planarizeTransform_temp_scaled.yaml"):
            with open("./aikensa/cameracalibration/planarizeTransform_temp_scaled.yaml") as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                self.planarizeTransform_temp_scaled = np.array(transform_list)     


        while self.running:

            if self.inspection_config.widget == 0:
                self.inspection_config.cameraID = -1

            if self.inspection_config.widget == 8:

                if self.inspection_config.furyou_plus or self.inspection_config.furyou_minus or self.inspection_config.kansei_plus or self.inspection_config.kansei_minus or self.inspection_config.furyou_plus_10 or self.inspection_config.furyou_minus_10 or self.inspection_config.kansei_plus_10 or self.inspection_config.kansei_minus_10:
                    self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget] = self.manual_adjustment(
                        self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget],
                        self.inspection_config.furyou_plus, 
                        self.inspection_config.furyou_minus, 
                        self.inspection_config.furyou_plus_10, 
                        self.inspection_config.furyou_minus_10, 
                        self.inspection_config.kansei_plus, 
                        self.inspection_config.kansei_minus,
                        self.inspection_config.kansei_plus_10,
                        self.inspection_config.kansei_minus_10)
                    print("Manual Adjustment Done")
                    print(f"Furyou Plus: {self.inspection_config.furyou_plus}")
                    print(f"Furyou Minus: {self.inspection_config.furyou_minus}")
                    print(f"Kansei Plus: {self.inspection_config.kansei_plus}")
                    print(f"Kansei Minus: {self.inspection_config.kansei_minus}")
                    print(f"Furyou Plus 10: {self.inspection_config.furyou_plus_10}")
                    print(f"Furyou Minus 10: {self.inspection_config.furyou_minus_10}")
                    print(f"Kansei Plus 10: {self.inspection_config.kansei_plus_10}")
                    print(f"Kansei Minus 10: {self.inspection_config.kansei_minus_10}")

                if self.inspection_config.counterReset is True:
                    self.inspection_config.current_numofPart[self.inspection_config.widget] = [0, 0]
                    self.inspection_config.counterReset = False
                    self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                            numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget],
                            currentnumofPart = [0, 0], 
                            deltaTime = 0.0,
                            kensainName = self.inspection_config.kensainNumber, 
                            detected_pitch_str = "COUNTERRESET", 
                            delta_pitch_str = "COUNTERRESET", 
                            total_length=0,
                            resultPitch = "COUNTERRESET",
                            status = "COUNTERRESET",
                            NGreason = "COUNTERRESET")
                #check kouden sensor and tenmetsu status

                for i in range (len(self.inspection_config.kouden_sensor)):
                    self.ethernet_status_red_tenmetsu_status[i] = self.inspection_config.kouden_sensor[i]
                    self.ethernet_status_green_hold_status[i] = 0 #Reset at every loop
                    self.ethernet_status_red_hold_status[i] = 0 #Reset at every loop

                for i in range(len(self.ethernet_status_red_tenmetsu_status)):
                    if self.ethernet_status_red_tenmetsu_status[i] == 1:
                        self.InspectionStatus[i] = "製品検出済み"
                    else:
                        self.InspectionStatus[i] = "製品未検出"
                    
                if self.multiCam_stream is False:
                    self.multiCam_stream = True
                    self.initialize_all_camera()

                _, self.bottomframe = self.cap_cam0.read()
                _, self.mergeframe1 = self.cap_cam1.read()
                _, self.mergeframe2 = self.cap_cam2.read()
                _, self.mergeframe3 = self.cap_cam3.read()
                _, self.mergeframe4 = self.cap_cam4.read()
                _, self.mergeframe5 = self.cap_cam5.read()

                self.mergeframe1 = cv2.rotate(self.mergeframe1, cv2.ROTATE_180)
                self.mergeframe2 = cv2.rotate(self.mergeframe2, cv2.ROTATE_180)
                self.mergeframe3 = cv2.rotate(self.mergeframe3, cv2.ROTATE_180)
                self.mergeframe4 = cv2.rotate(self.mergeframe4, cv2.ROTATE_180)
                self.mergeframe5 = cv2.rotate(self.mergeframe5, cv2.ROTATE_180)

                #Downsampled the image
                self.mergeframe1_scaled = self.downSampling(self.mergeframe1, self.scaled_width, self.scaled_height)
                self.mergeframe2_scaled = self.downSampling(self.mergeframe2, self.scaled_width, self.scaled_height)
                self.mergeframe3_scaled = self.downSampling(self.mergeframe3, self.scaled_width, self.scaled_height)
                self.mergeframe4_scaled = self.downSampling(self.mergeframe4, self.scaled_width, self.scaled_height)
                self.mergeframe5_scaled = self.downSampling(self.mergeframe5, self.scaled_width, self.scaled_height)

                if self.inspection_config.mapCalculated[1] is False: #Just checking the first camera to reduce loop time
                    for i in range(1, 6):
                        if os.path.exists(self._save_dir + f"Calibration_camera_{i}.yaml"):
                            camera_matrix, dist_coeffs = self.load_matrix_from_yaml(self._save_dir + f"Calibration_camera_{i}.yaml")
                            # Precompute the undistort and rectify map for faster processing
                            h, w = self.mergeframe1.shape[:2] #use mergeframe1 as reference
                            self.inspection_config.map1[i], self.inspection_config.map2[i] = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2)
                            print(f"map1 and map2 value is calculated")
                            self.inspection_config.mapCalculated[i] = True
                            print(f"Calibration map is calculated for Camera {i}")

                            #Also do the map for the scaled image
                            camera_matrix, dist_coeffs = self.load_matrix_from_yaml(self._save_dir + f"Calibration_camera_scaled_{i}.yaml")
                            h, w = self.mergeframe1_scaled.shape[:2] #use mergeframe1 as reference
                            self.inspection_config.map1_downscaled[i], self.inspection_config.map2_downscaled[i] = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2)
                            print(f"map1 and map2 value is calculated for scaled image")
                            print(f"Calibration map is calculated for Camera {i} for scaled image")

                            #Not idea but the condition use the bigger image

                if self.inspection_config.mapCalculated[1] is True: #Just checking the first camera to reduce loop time

                    #rotate the bottom frame 90deg CCW
                    self.bottomframe = self.downScaledImage(self.bottomframe, self.scale_factor_hole)
                    self.bottomframe = cv2.flip(cv2.rotate(self.bottomframe, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)

                    self.holeFrame1 = self.bottomframe[self.hole1Crop_XYpos_scaled[0]:self.hole1Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole1Crop_XYpos_scaled[1]:self.hole1Crop_XYpos_scaled[1] + self.width_hole_offset]
                    self.holeFrame2 = self.bottomframe[self.hole2Crop_XYpos_scaled[0]:self.hole2Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole2Crop_XYpos_scaled[1]:self.hole2Crop_XYpos_scaled[1] + self.width_hole_offset]
                    self.holeFrame3 = self.bottomframe[self.hole3Crop_XYpos_scaled[0]:self.hole3Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole3Crop_XYpos_scaled[1]:self.hole3Crop_XYpos_scaled[1] + self.width_hole_offset]
                    self.holeFrame4 = self.bottomframe[self.hole4Crop_XYpos_scaled[0]:self.hole4Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole4Crop_XYpos_scaled[1]:self.hole4Crop_XYpos_scaled[1] + self.width_hole_offset]
                    self.holeFrame5 = self.bottomframe[self.hole5Crop_XYpos_scaled[0]:self.hole5Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole5Crop_XYpos_scaled[1]:self.hole5Crop_XYpos_scaled[1] + self.width_hole_offset]
                
                    if self.inspection_config.doInspection is False:

                        self.mergeframe1_scaled = cv2.remap(self.mergeframe1_scaled, self.inspection_config.map1_downscaled[1], self.inspection_config.map2_downscaled[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe2_scaled = cv2.remap(self.mergeframe2_scaled, self.inspection_config.map1_downscaled[2], self.inspection_config.map2_downscaled[2], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe3_scaled = cv2.remap(self.mergeframe3_scaled, self.inspection_config.map1_downscaled[3], self.inspection_config.map2_downscaled[3], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe4_scaled = cv2.remap(self.mergeframe4_scaled, self.inspection_config.map1_downscaled[4], self.inspection_config.map2_downscaled[4], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe5_scaled = cv2.remap(self.mergeframe5_scaled, self.inspection_config.map1_downscaled[5], self.inspection_config.map2_downscaled[5], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                        self.combinedImage_scaled = warpTwoImages_template(self.homography_blank_canvas_scaled, self.mergeframe1_scaled, self.H1_scaled)
                        self.combinedImage_scaled = warpTwoImages_template(self.combinedImage_scaled, self.mergeframe2_scaled, self.H2_scaled)
                        self.combinedImage_scaled = warpTwoImages_template(self.combinedImage_scaled, self.mergeframe3_scaled, self.H3_scaled)
                        self.combinedImage_scaled = warpTwoImages_template(self.combinedImage_scaled, self.mergeframe4_scaled, self.H4_scaled)
                        self.combinedImage_scaled = warpTwoImages_template(self.combinedImage_scaled, self.mergeframe5_scaled, self.H5_scaled)

                        self.combinedImage_scaled = cv2.warpPerspective(self.combinedImage_scaled, self.planarizeTransform_scaled, (int(self.homography_size[1]/self.scale_factor), int(self.homography_size[0]/self.scale_factor)))
                        self.combinedImage_scaled = cv2.resize(self.combinedImage_scaled, (int(self.homography_size[1]/(self.scale_factor*1.48)), int(self.homography_size[0]/(self.scale_factor*1.26*1.48))))#1.48 for the qt, 1.26 for the aspect ratio

                        # cv2.imwrite("combinedImage_scaled_inference.png", self.combinedImage_scaled)

                        #Crop the image scaled for each part
                        self.part1Crop_scaled = self.combinedImage_scaled[self.part1Crop_YPos_scaled : self.part1Crop_YPos_scaled + self.part_height_offset_scaled, 0 : self.homography_size_scaled[1]]
                        self.part2Crop_scaled = self.combinedImage_scaled[self.part2Crop_YPos_scaled : self.part2Crop_YPos_scaled + self.part_height_offset_scaled, 0 : self.homography_size_scaled[1]]
                        self.part3Crop_scaled = self.combinedImage_scaled[self.part3Crop_YPos_scaled : self.part3Crop_YPos_scaled + self.part_height_offset_scaled, 0 : self.homography_size_scaled[1]]
                        self.part4Crop_scaled = self.combinedImage_scaled[self.part4Crop_YPos_scaled : self.part4Crop_YPos_scaled + self.part_height_offset_scaled, 0 : self.homography_size_scaled[1]]
                        self.part5Crop_scaled = self.combinedImage_scaled[self.part5Crop_YPos_scaled : self.part5Crop_YPos_scaled + self.part_height_offset_scaled, 0 : self.homography_size_scaled[1]]

                        self.part1Crop_scaled = self.downSampling(self.part1Crop_scaled, width=1771, height=24)
                        self.part2Crop_scaled = self.downSampling(self.part2Crop_scaled, width=1771, height=24)
                        self.part3Crop_scaled = self.downSampling(self.part3Crop_scaled, width=1771, height=24)
                        self.part4Crop_scaled = self.downSampling(self.part4Crop_scaled, width=1771, height=24)
                        self.part5Crop_scaled = self.downSampling(self.part5Crop_scaled, width=1771, height=24)

                        # self.save_image_hole(self.holeFrame1, False, "P1")
                        # self.save_image_hole(self.holeFrame2, False, "P2")
                        # self.save_image_hole(self.holeFrame3, False, "P3")
                        # self.save_image_hole(self.holeFrame4, False, "P4")
                        # self.save_image_hole(self.holeFrame5, False, "P5")

                        self.holeImageMerge[0] = self.holeFrame1.copy()
                        self.holeImageMerge[1] = self.holeFrame2.copy()
                        self.holeImageMerge[2] = self.holeFrame3.copy()
                        self.holeImageMerge[3] = self.holeFrame4.copy()
                        self.holeImageMerge[4] = self.holeFrame5.copy()

                        self.holeFrame1 = self.downScaledImage(self.holeFrame1, 1.5)
                        self.holeFrame2 = self.downScaledImage(self.holeFrame2, 1.5)
                        self.holeFrame3 = self.downScaledImage(self.holeFrame3, 1.5)
                        self.holeFrame4 = self.downScaledImage(self.holeFrame4, 1.5)
                        self.holeFrame5 = self.downScaledImage(self.holeFrame5, 1.5)

                        for i in range(len(self.holeImageMerge)):
                            self.InspectionResult_HoleDetection[i] = self.hoodFR_holeDetectionModel(
                                cv2.cvtColor(self.holeImageMerge[i], cv2.COLOR_BGR2RGB),
                                stream=True, verbose=False, conf=0.3, imgsz=256
                            )
                            detectedid = []
                            
                            for j, r in enumerate(self.InspectionResult_HoleDetection[i]):
                                for k, box in enumerate(r.boxes):
                                    detectedid.append(box.cls.item())
                            # print(f"Detected ID[i]: {detectedid}")
                            if 0.0 in detectedid:
                                self.DetectionResult_HoleDetection[i] = 1
                            else:
                                self.DetectionResult_HoleDetection[i] = 0

                        # print(self.DetectionResult_HoleDetection)

                        self.InspectionResult_PitchMeasured = [None]*5
                        self.InspectionResult_PitchResult = [None]*5

                        if self.InspectionImages_prev[0] is not None and any(sensor != 0 for sensor in self.inspection_config.kouden_sensor):
                            #convert to bgr
                            self.part1Crop_scaled = cv2.cvtColor(self.InspectionImages_prev[0], cv2.COLOR_RGB2BGR)
                            self.part2Crop_scaled = cv2.cvtColor(self.InspectionImages_prev[1], cv2.COLOR_RGB2BGR)
                            self.part3Crop_scaled = cv2.cvtColor(self.InspectionImages_prev[2], cv2.COLOR_RGB2BGR)
                            self.part4Crop_scaled = cv2.cvtColor(self.InspectionImages_prev[3], cv2.COLOR_RGB2BGR)
                            self.part5Crop_scaled = cv2.cvtColor(self.InspectionImages_prev[4], cv2.COLOR_RGB2BGR)

                            self.ethernet_status_red_tenmetsu_status = self.ethernet_status_red_tenmetsu_status_prev.copy()
                            self.ethernet_status_green_hold_status = self.ethernet_status_green_hold_status_prev.copy()
                            self.ethernet_status_red_hold_status = self.ethernet_status_red_hold_status_prev.copy()

                            self.InspectionResult_PitchMeasured = self.InspectionResult_PitchMeasured_prev.copy()
                            self.InspectionResult_PitchResult = self.InspectionResult_PitchResult_prev.copy()
                            self.InspectionStatus = self.InspectionStatus_prev.copy()

                            ng_exists = False
                            for status in self.InspectionResult_Status:
                                if status == "NG":
                                    ng_exists = True
                                    break

                            if ng_exists:
                                for i, sensor_value in enumerate(self.inspection_config.kouden_sensor):
                                    if sensor_value == 0:  # 0 indicates the part has been removed
                                        if self.InspectionResult_Status[i] == "OK":
                                            play_alarm_sound()  # Alarm if a good part is being removed
                                        elif self.InspectionResult_Status[i] == "NG":
                                            self.InspectionResult_Status[i] = "None"
                                            play_picking_sound()  # Confirm removal of the NG part
            

                        if all(sensor == 0 for sensor in self.inspection_config.kouden_sensor):
                            self.InspectionImages_prev[0] = None

                        if self.part1Crop_scaled is not None:
                            self.part1Cam.emit(self.convertQImage(self.part1Crop_scaled))
                        if self.part2Crop_scaled is not None:
                            self.part2Cam.emit(self.convertQImage(self.part2Crop_scaled))
                        if self.part3Crop_scaled is not None:
                            self.part3Cam.emit(self.convertQImage(self.part3Crop_scaled))
                        if self.part4Crop_scaled is not None:
                            self.part4Cam.emit(self.convertQImage(self.part4Crop_scaled))
                        if self.part5Crop_scaled is not None:
                            self.part5Cam.emit(self.convertQImage(self.part5Crop_scaled))

                        if self.holeFrame1 is not None:
                            self.hole1Cam.emit(self.convertQImage(self.holeFrame1))
                        if self.holeFrame2 is not None:
                            self.hole2Cam.emit(self.convertQImage(self.holeFrame2))
                        if self.holeFrame3 is not None:
                            self.hole3Cam.emit(self.convertQImage(self.holeFrame3))
                        if self.holeFrame4 is not None:
                            self.hole4Cam.emit(self.convertQImage(self.holeFrame4))
                        if self.holeFrame5 is not None:
                            self.hole5Cam.emit(self.convertQImage(self.holeFrame5))

                        #Empty the Inspection Result
                        
                        self.hoodFR_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured)
                        self.hoodFR_InspectionResult_PitchResult.emit(self.InspectionResult_PitchResult)

                    if self.InspectionTimeStart is None:
                        self.InspectionTimeStart = time.time()

                    # print(time.time() - self.InspectionTimeStart)

                    if self.firstTimeInspection is False:
                        if time.time() - self.InspectionTimeStart < self.InspectionWaitTime:
                            self.inspection_config.doInspection = False

                    # print(f"InspectionConfig: {self.inspection_config.doInspection}")

                    if self.inspection_config.doInspection is True:
                        self.inspection_config.doInspection = False

                        if self.inspection_config.kensainNumber is None or self.inspection_config.kensainNumber == "":
                            #Break the bottom if 
                            print("No Kensain Number Input")
                            for i in range (len(self.InspectionStatus)):
                                self.InspectionStatus[i] = "社員番号未入力"
                            self.hoodFR_InspectionStatus.emit(self.InspectionStatus)
                            continue
                        
                        if self.InspectionTimeStart is not None:

                            if time.time() - self.InspectionTimeStart > self.InspectionWaitTime or self.firstTimeInspection is True:
                                self.firstTimeInspection is False
                                print("Inspection Started") 
                                self.InspectionTimeStart = time.time()
                                
                                for i in range (len(self.InspectionStatus)):
                                    self.InspectionStatus[i] = "検査中"
                                self.hoodFR_InspectionStatus.emit(self.InspectionStatus)

                                self.mergeframe1 = cv2.remap(self.mergeframe1, self.inspection_config.map1[1], self.inspection_config.map2[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                                self.mergeframe2 = cv2.remap(self.mergeframe2, self.inspection_config.map1[2], self.inspection_config.map2[2], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                                self.mergeframe3 = cv2.remap(self.mergeframe3, self.inspection_config.map1[3], self.inspection_config.map2[3], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                                self.mergeframe4 = cv2.remap(self.mergeframe4, self.inspection_config.map1[4], self.inspection_config.map2[4], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                                self.mergeframe5 = cv2.remap(self.mergeframe5, self.inspection_config.map1[5], self.inspection_config.map2[5], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                                self.combinedImage = warpTwoImages_template(self.homography_blank_canvas, self.mergeframe1, self.H1)
                                self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe2, self.H2)
                                self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe3, self.H3)
                                self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe4, self.H4)
                                self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe5, self.H5)
                                                                            
                                self.combinedImage = cv2.warpPerspective(self.combinedImage, self.planarizeTransform, (int(self.homography_size[1]), int(self.homography_size[0])))
                                self.combinedImage = cv2.resize(self.combinedImage, (self.homography_size[1], int(self.homography_size[0]/(1.26))))#1.48 for the qt, 1.26 for the aspect ratio

                                # Crop the image scaled for each part
                                self.part1Crop = self.combinedImage[int(self.part1Crop_YPos*1.48) : int((self.part1Crop_YPos + self.part_height_offset)*1.48), 0 : int(self.homography_size[1]*1.48)]
                                self.part2Crop = self.combinedImage[int(self.part2Crop_YPos*1.48) : int((self.part2Crop_YPos + self.part_height_offset)*1.48), 0 : int(self.homography_size[1]*1.48)]
                                self.part3Crop = self.combinedImage[int(self.part3Crop_YPos*1.48) : int((self.part3Crop_YPos + self.part_height_offset)*1.48), 0 : int(self.homography_size[1]*1.48)]
                                self.part4Crop = self.combinedImage[int(self.part4Crop_YPos*1.48) : int((self.part4Crop_YPos + self.part_height_offset)*1.48), 0 : int(self.homography_size[1]*1.48)]
                                self.part5Crop = self.combinedImage[int(self.part5Crop_YPos*1.48) : int((self.part5Crop_YPos + self.part_height_offset)*1.48), 0 : int(self.homography_size[1]*1.48)]

                                #Need to convert to BGR for SAHI Inspection
                                self.part1Crop = cv2.cvtColor(self.part1Crop, cv2.COLOR_RGB2BGR)
                                self.part2Crop = cv2.cvtColor(self.part2Crop, cv2.COLOR_RGB2BGR)
                                self.part3Crop = cv2.cvtColor(self.part3Crop, cv2.COLOR_RGB2BGR)
                                self.part4Crop = cv2.cvtColor(self.part4Crop, cv2.COLOR_RGB2BGR)
                                self.part5Crop = cv2.cvtColor(self.part5Crop, cv2.COLOR_RGB2BGR)

                                #Put the All the image into a list
                                self.InspectionImages[0] = self.part1Crop.copy()
                                self.InspectionImages[1] = self.part2Crop.copy()
                                self.InspectionImages[2] = self.part3Crop.copy()
                                self.InspectionImages[3] = self.part4Crop.copy()
                                self.InspectionImages[4] = self.part5Crop.copy()

                                print(f"Length of Inspection Images : {len(self.InspectionImages)}") 

                                #Emit　検査中 to the status signal

                                # # Do the inspection
                                for i in range(len(self.InspectionImages)):
                                    #Only do inspectino on the one with kouden sensor on
                                    if self.inspection_config.kouden_sensor[i] == 1:
                                        self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                            self.InspectionImages[i], 
                                            self.hoodFR_clipDetectionModel, 
                                            slice_height=180, slice_width=1280, 
                                            overlap_height_ratio=0.0, overlap_width_ratio=0.3,
                                            postprocess_match_metric="IOS",
                                            postprocess_match_threshold=0.005,
                                            postprocess_class_agnostic=True,
                                            postprocess_type="GREEDYNMM",
                                            verbose=0,
                                            perform_standard_pred=True
                                        )

                                        #Crop image from 0 to 1024 width for the left and width-1024 to 1024 for the right
                                        self.InspectionImages_endSegmentation_Left[i] = self.InspectionImages[i][:, :1024, :]
                                        self.InspectionImages_endSegmentation_Right[i] = self.InspectionImages[i][:, -1024:, :]

                                        self.InspectionResult_EndSegmentation_Left[i] = self.hoodFR_endSegmentationModel(source=self.InspectionImages_endSegmentation_Left[i], conf=0.5, imgsz=960, verbose=False)
                                        self.InspectionResult_EndSegmentation_Right[i] = self.hoodFR_endSegmentationModel(source=self.InspectionImages_endSegmentation_Right[i], conf=0.5, imgsz=960, verbose=False)

                                        self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i], self.InspectionResult_NGReason[i] = partcheck(self.InspectionImages[i], 
                                                                                                                                                                                                                                  self.InspectionResult_ClipDetection[i].object_prediction_list,
                                                                                                                                                                                                                                  self.InspectionResult_EndSegmentation_Left[i],
                                                                                                                                                                                                                                  self.InspectionResult_EndSegmentation_Right[i])
                                    else:
                                        #Make pure black image
                                        self.InspectionImages[i] = np.full((24, 1771, 3), (10, 10, 20), dtype=np.uint8)
                                        self.InspectionResult_PitchMeasured[i] = None
                                        self.InspectionResult_PitchResult[i] = None
                                        self.InspectionResult_DetectionID[i] = None
                                        self.InspectionResult_Status[i] = None
                                    
                                    # print(self.InspectionResult_Status[i])
                                    # print(self.InspectionResult_DetectionID[i])

                                    if self.InspectionResult_Status[i] == "OK":
                                        self.ethernet_status_green_hold_status[i] = 1
                                        self.ethernet_status_red_tenmetsu_status[i] = 0
                                        self.ethernet_status_red_hold_status[i] = 0

                                        self.inspection_config.current_numofPart[self.inspection_config.widget][0] += 1
                                        self.inspection_config.today_numofPart[self.inspection_config.widget][0] += 1

                                        #Play konpou sound if the current_numofPart is dividable by 25
                                        if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 25 == 0 and self.inspection_config.current_numofPart[self.inspection_config.widget][0] != 0:
                                            play_konpou_sound()

                                        self.InspectionStatus[i] = "OK"

                                    elif self.InspectionResult_Status[i] == "NG":
                                        self.ethernet_status_green_hold_status[i] = 0
                                        self.ethernet_status_red_tenmetsu_status[i] = 0
                                        self.ethernet_status_red_hold_status[i] = 1
                                        self.inspection_config.current_numofPart[self.inspection_config.widget][1] += 1
                                        self.inspection_config.today_numofPart[self.inspection_config.widget][1] += 1

                                        self.InspectionStatus[i] = "NG"

                                    else:
                                        self.ethernet_status_green_hold_status[i] = 0
                                        self.ethernet_status_red_tenmetsu_status[i] = 0
                                        self.ethernet_status_red_hold_status[i] = 0

                                    self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                                        numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget], 
                                        currentnumofPart = self.inspection_config.current_numofPart[self.inspection_config.widget],
                                        deltaTime = 0.0,
                                        kensainName = self.inspection_config.kensainNumber, 
                                        detected_pitch_str = self.InspectionResult_PitchMeasured[i], 
                                        delta_pitch_str = self.InspectionResult_DeltaPitch[i], 
                                        total_length=0,
                                        resultPitch = self.InspectionResult_PitchResult[i], 
                                        status = self.InspectionResult_Status[i], 
                                        NGreason = self.InspectionResult_NGReason[i])
                                    
                                    
                                    #save hole image
                                    self.save_image_hole(self.holeFrame1, False, "P1")
                                    self.save_image_hole(self.holeFrame2, False, "P2")
                                    self.save_image_hole(self.holeFrame3, False, "P3")
                                    self.save_image_hole(self.holeFrame4, False, "P4")
                                    self.save_image_hole(self.holeFrame5, False, "P5")
                                    
                                    self.hoodFR_InspectionStatus.emit(self.InspectionStatus)

                                self.save_image_result(self.part1Crop, self.InspectionImages[0], self.InspectionResult_Status[0], True, "P1")
                                self.save_image_result(self.part2Crop, self.InspectionImages[1], self.InspectionResult_Status[1], True, "P2")
                                self.save_image_result(self.part3Crop, self.InspectionImages[2], self.InspectionResult_Status[2], True, "P3")
                                self.save_image_result(self.part4Crop, self.InspectionImages[3], self.InspectionResult_Status[3], True, "P4")
                                self.save_image_result(self.part5Crop, self.InspectionImages[4], self.InspectionResult_Status[4], True, "P5")

                                self.InspectionImages[0] = self.downSampling(self.InspectionImages[0], width=1771, height=24)
                                self.InspectionImages[1] = self.downSampling(self.InspectionImages[1], width=1771, height=24)
                                self.InspectionImages[2] = self.downSampling(self.InspectionImages[2], width=1771, height=24)
                                self.InspectionImages[3] = self.downSampling(self.InspectionImages[3], width=1771, height=24)
                                self.InspectionImages[4] = self.downSampling(self.InspectionImages[4], width=1771, height=24)

                                self.hoodFR_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured)
                                self.hoodFR_InspectionResult_PitchResult.emit(self.InspectionResult_PitchResult)
                                print("Inspection Finished")
                                #Remember that list is mutable
                                self.ethernet_status_red_tenmetsu_status_prev = self.ethernet_status_red_tenmetsu_status_prev.copy()
                                self.ethernet_status_green_hold_status_prev = self.ethernet_status_green_hold_status.copy()
                                self.ethernet_status_red_hold_status_prev = self.ethernet_status_red_hold_status.copy()

                                self.InspectionImages_prev[0] = self.InspectionImages[0]
                                self.InspectionImages_prev[1] = self.InspectionImages[1]
                                self.InspectionImages_prev[2] = self.InspectionImages[2]
                                self.InspectionImages_prev[3] = self.InspectionImages[3]
                                self.InspectionImages_prev[4] = self.InspectionImages[4]

                                self.InspectionResult_PitchMeasured_prev = self.InspectionResult_PitchMeasured.copy()
                                self.InspectionResult_PitchResult_prev = self.InspectionResult_PitchResult.copy()
                                self.InspectionStatus_prev = self.InspectionStatus.copy()

                                self.ethernet_status_red_tenmetsu.emit(self.ethernet_status_red_tenmetsu_status)
                                self.ethernet_status_green_hold.emit(self.ethernet_status_green_hold_status)
                                self.ethernet_status_red_hold.emit(self.ethernet_status_red_hold_status)

                                self.part1Cam.emit(self.converQImageRGB(self.InspectionImages[0]))
                                self.part2Cam.emit(self.converQImageRGB(self.InspectionImages[1]))
                                self.part3Cam.emit(self.converQImageRGB(self.InspectionImages[2]))
                                self.part4Cam.emit(self.converQImageRGB(self.InspectionImages[3]))
                                self.part5Cam.emit(self.converQImageRGB(self.InspectionImages[4]))

                                # self.hoodFR_InspectionStatus.emit(self.InspectionStatus)

                #emit the ethernet 
                self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
                self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)
            
                self.ethernet_status_red_tenmetsu.emit(self.ethernet_status_red_tenmetsu_status)
                self.ethernet_status_green_hold.emit(self.ethernet_status_green_hold_status)
                self.ethernet_status_red_hold.emit(self.ethernet_status_red_hold_status)

                # Emit status based on the red tenmetsu status

                
                self.hoodFR_InspectionStatus.emit(self.InspectionStatus)

                # Emit the hole detection
                self.hoodFR_HoleStatus.emit(self.DetectionResult_HoleDetection)

            if self.inspection_config.widget == 9:

                if self.inspection_config.furyou_plus or self.inspection_config.furyou_minus or self.inspection_config.kansei_plus or self.inspection_config.kansei_minus or self.inspection_config.furyou_plus_10 or self.inspection_config.furyou_minus_10 or self.inspection_config.kansei_plus_10 or self.inspection_config.kansei_minus_10:
                    self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget] = self.manual_adjustment(
                        self.inspection_config.current_numofPart[self.inspection_config.widget], self.inspection_config.today_numofPart[self.inspection_config.widget],
                        self.inspection_config.furyou_plus, 
                        self.inspection_config.furyou_minus, 
                        self.inspection_config.furyou_plus_10, 
                        self.inspection_config.furyou_minus_10, 
                        self.inspection_config.kansei_plus, 
                        self.inspection_config.kansei_minus,
                        self.inspection_config.kansei_plus_10,
                        self.inspection_config.kansei_minus_10)
                    print("Manual Adjustment Done")
                    print(f"Furyou Plus: {self.inspection_config.furyou_plus}")
                    print(f"Furyou Minus: {self.inspection_config.furyou_minus}")
                    print(f"Kansei Plus: {self.inspection_config.kansei_plus}")
                    print(f"Kansei Minus: {self.inspection_config.kansei_minus}")
                    print(f"Furyou Plus 10: {self.inspection_config.furyou_plus_10}")
                    print(f"Furyou Minus 10: {self.inspection_config.furyou_minus_10}")
                    print(f"Kansei Plus 10: {self.inspection_config.kansei_plus_10}")
                    print(f"Kansei Minus 10: {self.inspection_config.kansei_minus_10}")

                if self.inspection_config.counterReset is True:
                    self.inspection_config.current_numofPart[self.inspection_config.widget] = [0, 0]
                    self.inspection_config.counterReset = False
                    self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                            numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget],
                            currentnumofPart = [0, 0], 
                            deltaTime = 0.0,
                            kensainName = self.inspection_config.kensainNumber, 
                            detected_pitch_str = "COUNTERRESET", 
                            delta_pitch_str = "COUNTERRESET", 
                            total_length=0,
                            resultPitch = "COUNTERRESET",
                            status = "COUNTERRESET",
                            NGreason = "COUNTERRESET")
                #check kouden sensor and tenmetsu status

                for i in range (len(self.inspection_config.kouden_sensor)):
                    self.ethernet_status_red_tenmetsu_status[i] = self.inspection_config.kouden_sensor[i]
                    self.ethernet_status_green_hold_status[i] = 0 #Reset at every loop
                    self.ethernet_status_red_hold_status[i] = 0 #Reset at every loop

                # for i in range(len(self.ethernet_status_red_tenmetsu_status)):
                #     if self.ethernet_status_red_tenmetsu_status[i] == 1:
                #         self.InspectionStatus[i] = "製品検出済み"
                #     else:
                #         self.InspectionStatus[i] = "製品未検出"

                    self.InspectionStatus[i] = "検査準備完了"
                    self.InspectionStatus[i] = "検査準備完了"
                    
                if self.multiCam_stream is False:
                    self.multiCam_stream = True
                    self.initialize_all_camera()

                # _, self.bottomframe = self.cap_cam0.read()
                _, self.mergeframe1 = self.cap_cam1.read()
                _, self.mergeframe2 = self.cap_cam2.read()
                _, self.mergeframe3 = self.cap_cam3.read()
                _, self.mergeframe4 = self.cap_cam4.read()
                _, self.mergeframe5 = self.cap_cam5.read()

                self.mergeframe1 = cv2.rotate(self.mergeframe1, cv2.ROTATE_180)
                self.mergeframe2 = cv2.rotate(self.mergeframe2, cv2.ROTATE_180)
                self.mergeframe3 = cv2.rotate(self.mergeframe3, cv2.ROTATE_180)
                self.mergeframe4 = cv2.rotate(self.mergeframe4, cv2.ROTATE_180)
                self.mergeframe5 = cv2.rotate(self.mergeframe5, cv2.ROTATE_180)

                #Downsampled the image
                self.mergeframe1_scaled = self.downSampling(self.mergeframe1, self.scaled_width, self.scaled_height)
                self.mergeframe2_scaled = self.downSampling(self.mergeframe2, self.scaled_width, self.scaled_height)
                self.mergeframe3_scaled = self.downSampling(self.mergeframe3, self.scaled_width, self.scaled_height)
                self.mergeframe4_scaled = self.downSampling(self.mergeframe4, self.scaled_width, self.scaled_height)
                self.mergeframe5_scaled = self.downSampling(self.mergeframe5, self.scaled_width, self.scaled_height)

                if self.inspection_config.mapCalculated[1] is False: #Just checking the first camera to reduce loop time
                    for i in range(1, 6):
                        if os.path.exists(self._save_dir + f"Calibration_camera_{i}.yaml"):
                            camera_matrix, dist_coeffs = self.load_matrix_from_yaml(self._save_dir + f"Calibration_camera_{i}.yaml")
                            # Precompute the undistort and rectify map for faster processing
                            h, w = self.mergeframe1.shape[:2] #use mergeframe1 as reference
                            self.inspection_config.map1[i], self.inspection_config.map2[i] = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2)
                            print(f"map1 and map2 value is calculated")
                            self.inspection_config.mapCalculated[i] = True
                            print(f"Calibration map is calculated for Camera {i}")

                            #Also do the map for the scaled image
                            camera_matrix, dist_coeffs = self.load_matrix_from_yaml(self._save_dir + f"Calibration_camera_scaled_{i}.yaml")
                            h, w = self.mergeframe1_scaled.shape[:2] #use mergeframe1 as reference
                            self.inspection_config.map1_downscaled[i], self.inspection_config.map2_downscaled[i] = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2)
                            print(f"map1 and map2 value is calculated for scaled image")
                            print(f"Calibration map is calculated for Camera {i} for scaled image")

                            #Not idea but the condition use the bigger image

                if self.inspection_config.mapCalculated[1] is True: #Just checking the first camera to reduce loop time

                    # #rotate the bottom frame 90deg CCW
                    # self.bottomframe = self.downScaledImage(self.bottomframe, self.scale_factor_hole)
                    # self.bottomframe = cv2.flip(cv2.rotate(self.bottomframe, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)

                    # self.holeFrame1 = self.bottomframe[self.hole1Crop_XYpos_scaled[0]:self.hole1Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole1Crop_XYpos_scaled[1]:self.hole1Crop_XYpos_scaled[1] + self.width_hole_offset]
                    # self.holeFrame2 = self.bottomframe[self.hole2Crop_XYpos_scaled[0]:self.hole2Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole2Crop_XYpos_scaled[1]:self.hole2Crop_XYpos_scaled[1] + self.width_hole_offset]
                    # self.holeFrame3 = self.bottomframe[self.hole3Crop_XYpos_scaled[0]:self.hole3Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole3Crop_XYpos_scaled[1]:self.hole3Crop_XYpos_scaled[1] + self.width_hole_offset]
                    # self.holeFrame4 = self.bottomframe[self.hole4Crop_XYpos_scaled[0]:self.hole4Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole4Crop_XYpos_scaled[1]:self.hole4Crop_XYpos_scaled[1] + self.width_hole_offset]
                    # self.holeFrame5 = self.bottomframe[self.hole5Crop_XYpos_scaled[0]:self.hole5Crop_XYpos_scaled[0] + self.height_hole_offset, self.hole5Crop_XYpos_scaled[1]:self.hole5Crop_XYpos_scaled[1] + self.width_hole_offset]
                
                    if self.inspection_config.doInspection is False:

                        self.mergeframe1_scaled = cv2.remap(self.mergeframe1_scaled, self.inspection_config.map1_downscaled[1], self.inspection_config.map2_downscaled[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe2_scaled = cv2.remap(self.mergeframe2_scaled, self.inspection_config.map1_downscaled[2], self.inspection_config.map2_downscaled[2], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe3_scaled = cv2.remap(self.mergeframe3_scaled, self.inspection_config.map1_downscaled[3], self.inspection_config.map2_downscaled[3], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe4_scaled = cv2.remap(self.mergeframe4_scaled, self.inspection_config.map1_downscaled[4], self.inspection_config.map2_downscaled[4], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                        self.mergeframe5_scaled = cv2.remap(self.mergeframe5_scaled, self.inspection_config.map1_downscaled[5], self.inspection_config.map2_downscaled[5], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                        self.combinedImage_scaled = warpTwoImages_template(self.homography_blank_canvas_scaled, self.mergeframe1_scaled, self.H1_scaled)
                        self.combinedImage_scaled = warpTwoImages_template(self.combinedImage_scaled, self.mergeframe2_scaled, self.H2_scaled)
                        self.combinedImage_scaled = warpTwoImages_template(self.combinedImage_scaled, self.mergeframe3_scaled, self.H3_scaled)
                        self.combinedImage_scaled = warpTwoImages_template(self.combinedImage_scaled, self.mergeframe4_scaled, self.H4_scaled)
                        self.combinedImage_scaled = warpTwoImages_template(self.combinedImage_scaled, self.mergeframe5_scaled, self.H5_scaled)

                        self.combinedImage_scaled = cv2.warpPerspective(self.combinedImage_scaled, self.planarizeTransform_temp_scaled, (int(self.homography_size[1]/self.scale_factor), int(self.homography_size[0]/self.scale_factor)))
                        self.combinedImage_scaled = cv2.resize(self.combinedImage_scaled, (int(self.homography_size[1]/(self.scale_factor*1.48)), int(self.homography_size[0]/(self.scale_factor*1.26*1.48))))#1.48 for the qt, 1.26 for the aspect ratio

                        cv2.imwrite("combinedImage_scaled_inference.png", self.combinedImage_scaled)

                        #Crop the image scaled for each part
                        self.part1Crop_scaled = self.combinedImage_scaled[self.part1Crop_Ypos_hoodFR_scaled : self.part1Crop_Ypos_hoodFR_scaled + self.part_height_offset_nissanhoodFR_scaled, 0 : self.homography_size_scaled[1]]
                        self.part2Crop_scaled = self.combinedImage_scaled[self.part2Crop_Ypos_hoodFR_scaled : self.part2Crop_Ypos_hoodFR_scaled + self.part_height_offset_nissanhoodFR_scaled, 0 : self.homography_size_scaled[1]]
                        self.part3Crop_scaled = self.combinedImage_scaled[self.part3Crop_Ypos_hoodFR_scaled : self.part3Crop_Ypos_hoodFR_scaled + self.part_height_offset_nissanhoodFR_scaled, 0 : self.homography_size_scaled[1]]
                        self.part4Crop_scaled = self.combinedImage_scaled[self.part4Crop_Ypos_hoodFR_scaled : self.part4Crop_Ypos_hoodFR_scaled + self.part_height_offset_nissanhoodFR_scaled, 0 : self.homography_size_scaled[1]]
                        self.part5Crop_scaled = self.combinedImage_scaled[self.part5Crop_Ypos_hoodFR_scaled : self.part5Crop_Ypos_hoodFR_scaled + self.part_height_offset_nissanhoodFR_scaled, 0 : self.homography_size_scaled[1]]

                        # cv2.imwrite("part1Crop_scaled_inference.png", self.part1Crop_scaled)

                        self.part1Crop_scaled = self.downSampling(self.part1Crop_scaled, width=1771, height=24)
                        self.part2Crop_scaled = self.downSampling(self.part2Crop_scaled, width=1771, height=24)
                        self.part3Crop_scaled = self.downSampling(self.part3Crop_scaled, width=1771, height=24)
                        self.part4Crop_scaled = self.downSampling(self.part4Crop_scaled, width=1771, height=24)
                        self.part5Crop_scaled = self.downSampling(self.part5Crop_scaled, width=1771, height=24)

                        # self.save_image_hole(self.holeFrame1, False, "P1")
                        # self.save_image_hole(self.holeFrame2, False, "P2")
                        # self.save_image_hole(self.holeFrame3, False, "P3")
                        # self.save_image_hole(self.holeFrame4, False, "P4")
                        # self.save_image_hole(self.holeFrame5, False, "P5")

                        # self.holeImageMerge[0] = self.holeFrame1.copy()
                        # self.holeImageMerge[1] = self.holeFrame2.copy()
                        # self.holeImageMerge[2] = self.holeFrame3.copy()
                        # self.holeImageMerge[3] = self.holeFrame4.copy()
                        # self.holeImageMerge[4] = self.holeFrame5.copy()

                        # self.holeFrame1 = self.downScaledImage(self.holeFrame1, 1.5)
                        # self.holeFrame2 = self.downScaledImage(self.holeFrame2, 1.5)
                        # self.holeFrame3 = self.downScaledImage(self.holeFrame3, 1.5)
                        # self.holeFrame4 = self.downScaledImage(self.holeFrame4, 1.5)
                        # self.holeFrame5 = self.downScaledImage(self.holeFrame5, 1.5)

                        # for i in range(len(self.holeImageMerge)):
                        #     self.InspectionResult_HoleDetection[i] = self.hoodFR_holeDetectionModel(
                        #         cv2.cvtColor(self.holeImageMerge[i], cv2.COLOR_BGR2RGB),
                        #         stream=True, verbose=False, conf=0.3, imgsz=256
                        #     )
                        #     detectedid = []
                            
                        #     for j, r in enumerate(self.InspectionResult_HoleDetection[i]):
                        #         for k, box in enumerate(r.boxes):
                        #             detectedid.append(box.cls.item())
                        #     # print(f"Detected ID[i]: {detectedid}")
                        #     if 0.0 in detectedid:
                        #         self.DetectionResult_HoleDetection[i] = 1
                        #     else:
                        #         self.DetectionResult_HoleDetection[i] = 0

                        # # print(self.DetectionResult_HoleDetection)

                        self.InspectionResult_PitchMeasured = [None]*5
                        self.InspectionResult_PitchResult = [None]*5

                        # if self.InspectionImages_prev[0] is not None and any(sensor != 0 for sensor in self.inspection_config.kouden_sensor):
                        #     #convert to bgr
                        #     self.part1Crop_scaled = cv2.cvtColor(self.InspectionImages_prev[0], cv2.COLOR_RGB2BGR)
                        #     self.part2Crop_scaled = cv2.cvtColor(self.InspectionImages_prev[1], cv2.COLOR_RGB2BGR)
                        #     self.part3Crop_scaled = cv2.cvtColor(self.InspectionImages_prev[2], cv2.COLOR_RGB2BGR)
                        #     self.part4Crop_scaled = cv2.cvtColor(self.InspectionImages_prev[3], cv2.COLOR_RGB2BGR)
                        #     self.part5Crop_scaled = cv2.cvtColor(self.InspectionImages_prev[4], cv2.COLOR_RGB2BGR)

                        #     self.ethernet_status_red_tenmetsu_status = self.ethernet_status_red_tenmetsu_status_prev.copy()
                        #     self.ethernet_status_green_hold_status = self.ethernet_status_green_hold_status_prev.copy()
                        #     self.ethernet_status_red_hold_status = self.ethernet_status_red_hold_status_prev.copy()

                        #     self.InspectionResult_PitchMeasured = self.InspectionResult_PitchMeasured_prev.copy()
                        #     self.InspectionResult_PitchResult = self.InspectionResult_PitchResult_prev.copy()
                        #     self.InspectionStatus = self.InspectionStatus_prev.copy()

                        #     ng_exists = False
                        #     for status in self.InspectionResult_Status:
                        #         if status == "NG":
                        #             ng_exists = True
                        #             break

                        #     if ng_exists:
                        #         for i, sensor_value in enumerate(self.inspection_config.kouden_sensor):
                        #             if sensor_value == 0:  # 0 indicates the part has been removed
                        #                 if self.InspectionResult_Status[i] == "OK":
                        #                     play_alarm_sound()  # Alarm if a good part is being removed
                        #                 elif self.InspectionResult_Status[i] == "NG":
                        #                     self.InspectionResult_Status[i] = "None"
                        #                     play_picking_sound()  # Confirm removal of the NG part
            

                        # if all(sensor == 0 for sensor in self.inspection_config.kouden_sensor):
                        #     self.InspectionImages_prev[0] = None

                        if self.part1Crop_scaled is not None:
                            self.part1Cam.emit(self.convertQImage(self.part1Crop_scaled))
                        if self.part2Crop_scaled is not None:
                            self.part2Cam.emit(self.convertQImage(self.part2Crop_scaled))
                        if self.part3Crop_scaled is not None:
                            self.part3Cam.emit(self.convertQImage(self.part3Crop_scaled))
                        if self.part4Crop_scaled is not None:
                            self.part4Cam.emit(self.convertQImage(self.part4Crop_scaled))
                        if self.part5Crop_scaled is not None:
                            self.part5Cam.emit(self.convertQImage(self.part5Crop_scaled))

                        # if self.holeFrame1 is not None:
                        #     self.hole1Cam.emit(self.convertQImage(self.holeFrame1))
                        # if self.holeFrame2 is not None:
                        #     self.hole2Cam.emit(self.convertQImage(self.holeFrame2))
                        # if self.holeFrame3 is not None:
                        #     self.hole3Cam.emit(self.convertQImage(self.holeFrame3))
                        # if self.holeFrame4 is not None:
                        #     self.hole4Cam.emit(self.convertQImage(self.holeFrame4))
                        # if self.holeFrame5 is not None:
                        #     self.hole5Cam.emit(self.convertQImage(self.holeFrame5))

                        #Empty the Inspection Result
                        
                        self.P8462284S00_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured)
                        # self.hoodFR_InspectionResult_PitchResult.emit(self.InspectionResult_PitchResult)

                    if self.InspectionTimeStart is None:
                        self.InspectionTimeStart = time.time()

                    # print(time.time() - self.InspectionTimeStart)

                    if self.firstTimeInspection is False:
                        if time.time() - self.InspectionTimeStart < self.InspectionWaitTime:
                            self.inspection_config.doInspection = False

                    # print(f"InspectionConfig: {self.inspection_config.doInspection}")

                    if self.inspection_config.doInspection is True:
                        self.inspection_config.doInspection = False

                        # if self.inspection_config.kensainNumber is None or self.inspection_config.kensainNumber == "":
                        #     #Break the bottom if 
                        #     print("No Kensain Number Input")
                        #     for i in range (len(self.InspectionStatus)):
                        #         self.InspectionStatus[i] = "社員番号未入力"
                        #     self.hoodFR_InspectionStatus.emit(self.InspectionStatus)
                        #     continue
                        
                        if self.InspectionTimeStart is not None:

                            if time.time() - self.InspectionTimeStart > self.InspectionWaitTime or self.firstTimeInspection is True:
                                self.firstTimeInspection is False
                                print("Inspection Started") 
                                self.InspectionTimeStart = time.time()
                                
                                for i in range (len(self.InspectionStatus)):
                                    self.InspectionStatus[i] = "検査中"
                                self.P8462284S00_InspectionStatus.emit(self.InspectionStatus)

                                self.mergeframe1 = cv2.remap(self.mergeframe1, self.inspection_config.map1[1], self.inspection_config.map2[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                                self.mergeframe2 = cv2.remap(self.mergeframe2, self.inspection_config.map1[2], self.inspection_config.map2[2], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                                self.mergeframe3 = cv2.remap(self.mergeframe3, self.inspection_config.map1[3], self.inspection_config.map2[3], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                                self.mergeframe4 = cv2.remap(self.mergeframe4, self.inspection_config.map1[4], self.inspection_config.map2[4], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                                self.mergeframe5 = cv2.remap(self.mergeframe5, self.inspection_config.map1[5], self.inspection_config.map2[5], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                                self.combinedImage = warpTwoImages_template(self.homography_blank_canvas, self.mergeframe1, self.H1)
                                self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe2, self.H2)
                                self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe3, self.H3)
                                self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe4, self.H4)
                                self.combinedImage = warpTwoImages_template(self.combinedImage, self.mergeframe5, self.H5)
                                                                            
                                self.combinedImage = cv2.warpPerspective(self.combinedImage, self.planarizeTransform_temp, (int(self.homography_size[1]), int(self.homography_size[0])))
                                self.combinedImage = cv2.resize(self.combinedImage, (self.homography_size[1], int(self.homography_size[0]/(1.26))))#1.48 for the qt, 1.26 for the aspect ratio

                                # Crop the image scaled for each part
                                self.part1Crop = self.combinedImage[int(self.part1Crop_YPos_hoodFR*1.48) : int((self.part1Crop_YPos_hoodFR + self.part_height_offset_nissanhoodFR)*1.48), 0 : int(self.homography_size[1]*1.48)]
                                self.part2Crop = self.combinedImage[int(self.part2Crop_YPos_hoodFR*1.48) : int((self.part2Crop_YPos_hoodFR + self.part_height_offset_nissanhoodFR)*1.48), 0 : int(self.homography_size[1]*1.48)]
                                self.part3Crop = self.combinedImage[int(self.part3Crop_YPos_hoodFR*1.48) : int((self.part3Crop_YPos_hoodFR + self.part_height_offset_nissanhoodFR)*1.48), 0 : int(self.homography_size[1]*1.48)]
                                self.part4Crop = self.combinedImage[int(self.part4Crop_YPos_hoodFR*1.48) : int((self.part4Crop_YPos_hoodFR + self.part_height_offset_nissanhoodFR)*1.48), 0 : int(self.homography_size[1]*1.48)]
                                self.part5Crop = self.combinedImage[int(self.part5Crop_YPos_hoodFR*1.48) : int((self.part5Crop_YPos_hoodFR + self.part_height_offset_nissanhoodFR)*1.48), 0 : int(self.homography_size[1]*1.48)]

                                # self.save_image(self.part1Crop)
                                # time.sleep(1.5)
                                # self.save_image(self.part2Crop)
                                # time.sleep(1.5)
                                # self.save_image(self.part3Crop)
                                # time.sleep(1.5)
                                # self.save_image(self.part4Crop)
                                # time.sleep(1.5)
                                # self.save_image(self.part5Crop)
                                # time.sleep(1.5)

                                #Need to convert to BGR for SAHI Inspection
                                self.part1Crop = cv2.cvtColor(self.part1Crop, cv2.COLOR_RGB2BGR)
                                self.part2Crop = cv2.cvtColor(self.part2Crop, cv2.COLOR_RGB2BGR)
                                self.part3Crop = cv2.cvtColor(self.part3Crop, cv2.COLOR_RGB2BGR)
                                self.part4Crop = cv2.cvtColor(self.part4Crop, cv2.COLOR_RGB2BGR)
                                self.part5Crop = cv2.cvtColor(self.part5Crop, cv2.COLOR_RGB2BGR)

                                #Put the All the image into a list
                                self.InspectionImages[0] = self.part1Crop.copy()
                                self.InspectionImages[1] = self.part2Crop.copy()
                                self.InspectionImages[2] = self.part3Crop.copy()
                                self.InspectionImages[3] = self.part4Crop.copy()
                                self.InspectionImages[4] = self.part5Crop.copy()

                                print(f"Length of Inspection Images : {len(self.InspectionImages)}") 


                                #Emit　検査中 to the status signal

                                # # Do the inspection
                                for i in range(len(self.InspectionImages)):
                                    #Only do inspectino on the one with kouden sensor ons
                                    self.inspection_config.kouden_sensor[i] = 1 #forcefully make it 1
                                    if self.inspection_config.kouden_sensor[i] == 1:
                                        self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                            self.InspectionImages[i], 
                                            self.P658207YA0A_clipDetectionModel, 
                                            slice_height=497, slice_width=1960, 
                                            overlap_height_ratio=0.0, overlap_width_ratio=0.2,
                                            postprocess_match_metric="IOS",
                                            postprocess_match_threshold=0.005,
                                            postprocess_class_agnostic=True,
                                            postprocess_type="GREEDYNMM",
                                            verbose=0,
                                            perform_standard_pred=False
                                        )

                                        #Crop image from 0 to 1024 width for the left and width-1024 to 1024 for the right
                                        # self.InspectionImages_endSegmentation_Left[i] = self.InspectionImages[i][:, :1024, :]
                                        # self.InspectionImages_endSegmentation_Right[i] = self.InspectionImages[i][:, -1024:, :]

                                        # self.InspectionResult_EndSegmentation_Left[i] = self.hoodFR_endSegmentationModel(source=self.InspectionImages_endSegmentation_Left[i], conf=0.5, imgsz=960, verbose=False)
                                        # self.InspectionResult_EndSegmentation_Right[i] = self.hoodFR_endSegmentationModel(source=self.InspectionImages_endSegmentation_Right[i], conf=0.5, imgsz=960, verbose=False)

                                        self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i], self.InspectionResult_NGReason[i] = P658207YA0A_partcheck(self.InspectionImages[i], self.InspectionResult_ClipDetection[i].object_prediction_list)
                                    else:
                                        #Make pure black image
                                        self.InspectionImages[i] = np.full((24, 1771, 3), (10, 10, 20), dtype=np.uint8)
                                        self.InspectionResult_PitchMeasured[i] = None
                                        self.InspectionResult_PitchResult[i] = None
                                        self.InspectionResult_DetectionID[i] = None
                                        self.InspectionResult_Status[i] = None
                                    
                                    # print(self.InspectionResult_Status[i])
                                    # print(self.InspectionResult_DetectionID[i])

                                    if self.InspectionResult_Status[i] == "OK":
                                        self.ethernet_status_green_hold_status[i] = 1
                                        self.ethernet_status_red_tenmetsu_status[i] = 0
                                        self.ethernet_status_red_hold_status[i] = 0

                                        self.inspection_config.current_numofPart[self.inspection_config.widget][0] += 1
                                        self.inspection_config.today_numofPart[self.inspection_config.widget][0] += 1

                                        #Play konpou sound if the current_numofPart is dividable by 25
                                        if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 25 == 0 and self.inspection_config.current_numofPart[self.inspection_config.widget][0] != 0:
                                            play_konpou_sound()

                                        self.InspectionStatus[i] = "OK"

                                    elif self.InspectionResult_Status[i] == "NG":
                                        self.ethernet_status_green_hold_status[i] = 0
                                        self.ethernet_status_red_tenmetsu_status[i] = 0
                                        self.ethernet_status_red_hold_status[i] = 1
                                        self.inspection_config.current_numofPart[self.inspection_config.widget][1] += 1
                                        self.inspection_config.today_numofPart[self.inspection_config.widget][1] += 1

                                        self.InspectionStatus[i] = "NG"

                                    else:
                                        self.ethernet_status_green_hold_status[i] = 0
                                        self.ethernet_status_red_tenmetsu_status[i] = 0
                                        self.ethernet_status_red_hold_status[i] = 0

                                    self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                                        numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget], 
                                        currentnumofPart = self.inspection_config.current_numofPart[self.inspection_config.widget],
                                        deltaTime = 0.0,
                                        kensainName = self.inspection_config.kensainNumber, 
                                        detected_pitch_str = self.InspectionResult_PitchMeasured[i], 
                                        delta_pitch_str = self.InspectionResult_DeltaPitch[i], 
                                        total_length=0,
                                        resultPitch = self.InspectionResult_PitchResult[i],
                                        status = self.InspectionStatus[i],
                                        NGreason = self.InspectionResult_NGReason[i])
                                    
                                    
                                    # #save hole image
                                    # self.save_image_hole(self.holeFrame1, False, "P1")
                                    # self.save_image_hole(self.holeFrame2, False, "P2")
                                    # self.save_image_hole(self.holeFrame3, False, "P3")
                                    # self.save_image_hole(self.holeFrame4, False, "P4")
                                    # self.save_image_hole(self.holeFrame5, False, "P5")
                                    
                                    self.P8462284S00_InspectionStatus.emit(self.InspectionStatus)

                                self.save_image_result(self.part1Crop, self.InspectionImages[0], self.InspectionResult_Status[0], True, "P1")
                                self.save_image_result(self.part2Crop, self.InspectionImages[1], self.InspectionResult_Status[1], True, "P2")
                                self.save_image_result(self.part3Crop, self.InspectionImages[2], self.InspectionResult_Status[2], True, "P3")
                                self.save_image_result(self.part4Crop, self.InspectionImages[3], self.InspectionResult_Status[3], True, "P4")
                                self.save_image_result(self.part5Crop, self.InspectionImages[4], self.InspectionResult_Status[4], True, "P5")

                                self.InspectionImages[0] = self.downSampling(self.InspectionImages[0], width=1771, height=24)
                                self.InspectionImages[1] = self.downSampling(self.InspectionImages[1], width=1771, height=24)
                                self.InspectionImages[2] = self.downSampling(self.InspectionImages[2], width=1771, height=24)
                                self.InspectionImages[3] = self.downSampling(self.InspectionImages[3], width=1771, height=24)
                                self.InspectionImages[4] = self.downSampling(self.InspectionImages[4], width=1771, height=24)

                                self.P8462284S00_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured)
                                self.P8462284S00_InspectionResult_PitchResult.emit(self.InspectionResult_PitchResult)

                                print("Inspection Finished")
                                #Remember that list is mutable
                                self.ethernet_status_red_tenmetsu_status_prev = self.ethernet_status_red_tenmetsu_status_prev.copy()
                                self.ethernet_status_green_hold_status_prev = self.ethernet_status_green_hold_status.copy()
                                self.ethernet_status_red_hold_status_prev = self.ethernet_status_red_hold_status.copy()

                                self.InspectionImages_prev[0] = self.InspectionImages[0]
                                self.InspectionImages_prev[1] = self.InspectionImages[1]
                                self.InspectionImages_prev[2] = self.InspectionImages[2]
                                self.InspectionImages_prev[3] = self.InspectionImages[3]
                                self.InspectionImages_prev[4] = self.InspectionImages[4]

                                self.InspectionResult_PitchMeasured_prev = self.InspectionResult_PitchMeasured.copy()
                                self.InspectionResult_PitchResult_prev = self.InspectionResult_PitchResult.copy()
                                self.InspectionStatus_prev = self.InspectionStatus.copy()

                                self.ethernet_status_red_tenmetsu.emit(self.ethernet_status_red_tenmetsu_status)
                                self.ethernet_status_green_hold.emit(self.ethernet_status_green_hold_status)
                                self.ethernet_status_red_hold.emit(self.ethernet_status_red_hold_status)

                                self.part1Cam.emit(self.converQImageRGB(self.InspectionImages[0]))
                                self.part2Cam.emit(self.converQImageRGB(self.InspectionImages[1]))
                                self.part3Cam.emit(self.converQImageRGB(self.InspectionImages[2]))
                                self.part4Cam.emit(self.converQImageRGB(self.InspectionImages[3]))
                                self.part5Cam.emit(self.converQImageRGB(self.InspectionImages[4]))

                                time.sleep(10)

                                # self.hoodFR_InspectionStatus.emit(self.InspectionStatus)

                #emit the ethernet 
                self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
                self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)
            
                self.ethernet_status_red_tenmetsu.emit(self.ethernet_status_red_tenmetsu_status)
                self.ethernet_status_green_hold.emit(self.ethernet_status_green_hold_status)
                self.ethernet_status_red_hold.emit(self.ethernet_status_red_hold_status)

                # Emit status based on the red tenmetsu status

                
                self.P8462284S00_InspectionStatus.emit(self.InspectionStatus)



            if self.inspection_config.widget == 21:
              
                if self.multiCam_stream is False:
                    self.multiCam_stream = True
                    self.initialize_all_camera()

                _, self.bottomframe = self.cap_cam0.read()
                _, self.mergeframe1 = self.cap_cam1.read()
                _, self.mergeframe2 = self.cap_cam2.read()
                _, self.mergeframe3 = self.cap_cam3.read()
                _, self.mergeframe4 = self.cap_cam4.read()
                _, self.mergeframe5 = self.cap_cam5.read()

                self.mergeframe1 = cv2.rotate(self.mergeframe1, cv2.ROTATE_180)
                self.mergeframe2 = cv2.rotate(self.mergeframe2, cv2.ROTATE_180)
                self.mergeframe3 = cv2.rotate(self.mergeframe3, cv2.ROTATE_180)
                self.mergeframe4 = cv2.rotate(self.mergeframe4, cv2.ROTATE_180)
                self.mergeframe5 = cv2.rotate(self.mergeframe5, cv2.ROTATE_180)

                #Downsampled the image
                self.mergeframe1_scaled = self.downSampling(self.mergeframe1, self.scaled_width, self.scaled_height)
                self.mergeframe2_scaled = self.downSampling(self.mergeframe2, self.scaled_width, self.scaled_height)
                self.mergeframe3_scaled = self.downSampling(self.mergeframe3, self.scaled_width, self.scaled_height)
                self.mergeframe4_scaled = self.downSampling(self.mergeframe4, self.scaled_width, self.scaled_height)
                self.mergeframe5_scaled = self.downSampling(self.mergeframe5, self.scaled_width, self.scaled_height)

                if self.inspection_config.mapCalculated[1] is False: #Just checking the first camera to reduce loop time
                    for i in range(1, 6):
                        if os.path.exists(self._save_dir + f"Calibration_camera_{i}.yaml"):
                            camera_matrix, dist_coeffs = self.load_matrix_from_yaml(self._save_dir + f"Calibration_camera_{i}.yaml")
                            # Precompute the undistort and rectify map for faster processing
                            h, w = self.mergeframe1.shape[:2] #use mergeframe1 as reference
                            self.inspection_config.map1[i], self.inspection_config.map2[i] = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2)
                            print(f"map1 and map2 value is calculated")
                            self.inspection_config.mapCalculated[i] = True
                            print(f"Calibration map is calculated for Camera {i}")

                            #Also do the map for the scaled image
                            camera_matrix, dist_coeffs = self.load_matrix_from_yaml(self._save_dir + f"Calibration_camera_scaled_{i}.yaml")
                            h, w = self.mergeframe1_scaled.shape[:2] #use mergeframe1 as reference
                            self.inspection_config.map1_downscaled[i], self.inspection_config.map2_downscaled[i] = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_16SC2)
                            print(f"map1 and map2 value is calculated for scaled image")
                            print(f"Calibration map is calculated for Camera {i} for scaled image")

                            #Not idea but the condition use the bigger image

                if self.inspection_config.mapCalculated[1] is True: #Just checking the first camera to reduce loop time
                
                    if self.inspection_config.doInspection is False:

                        self.mergeframe3_scaled = cv2.remap(self.mergeframe3_scaled, self.inspection_config.map1_downscaled[3], self.inspection_config.map2_downscaled[3], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                        self.combinedImage_scaled = warpTwoImages_template(self.homography_blank_canvas_scaled, self.mergeframe3_scaled, self.H3_scaled)

                        self.combinedImage_scaled = cv2.warpPerspective(self.combinedImage_scaled, self.planarizeTransform_scaled, (int(self.homography_size[1]/self.scale_factor), int(self.homography_size[0]/self.scale_factor)))
                        self.combinedImage_scaled = cv2.resize(self.combinedImage_scaled, (int(self.homography_size[1]/(self.scale_factor*1.48)), int(self.homography_size[0]/(self.scale_factor*1.26*1.48))))#1.48 for the qt, 1.26 for the aspect ratio

                        #Crop the image scaled for each part
                        self.part3Crop_scaled = self.combinedImage_scaled[self.part3Crop_YPos_scaled : self.part3Crop_YPos_scaled + self.part_height_offset_scaled, 0 : self.homography_size_scaled[1]]
                        #crop the image again, taking the center area with the width of self.dailyTenken_cropWidth only
                        cv2.imwrite("part3Crop_scaled.png", self.part3Crop_scaled)

                        self.part3Crop_scaled = self.part3Crop_scaled[:, int((self.part3Crop_scaled.shape[1] - self.dailyTenken_cropWidth_scaled)/2) : int((self.part3Crop_scaled.shape[1] + self.dailyTenken_cropWidth_scaled)/2)]
                        cv2.imwrite("part3Crop2_scaled.png", self.part3Crop_scaled)


                        self.part3Crop_scaled = self.downSampling(self.part3Crop_scaled, width=1710, height=198)

                        self.InspectionResult_PitchMeasured = [None]*5
                        self.InspectionResult_PitchResult = [None]*5

                        if self.part3Crop_scaled is not None:
                            self.dailytenkenCam.emit(self.convertQImage(self.part3Crop_scaled))

                    if self.InspectionTimeStart is None:
                        self.InspectionTimeStart = time.time()

                    # print(time.time() - self.InspectionTimeStart)

                    if self.firstTimeInspection is False:
                        if time.time() - self.InspectionTimeStart < self.InspectionWaitTime:
                            self.inspection_config.doInspection = False

                    # print(f"InspectionConfig: {self.inspection_config.doInspection}")

                    if self.inspection_config.doInspection is True:
                        self.inspection_config.doInspection = False

                        # if self.inspection_config.kensainNumber is None or self.inspection_config.kensainNumber == "":
                        #     #Break the bottom if 
                        #     print("No Kensain Number Input")
                        #     for i in range (len(self.InspectionStatus)):
                        #         self.InspectionStatus[i] = "社員番号未入力"
                        #     self.hoodFR_InspectionStatus.emit(self.InspectionStatus)
                        #     continue
                        
                        if self.InspectionTimeStart is not None:

                            if time.time() - self.InspectionTimeStart > self.InspectionWaitTime or self.firstTimeInspection is True:
                                self.firstTimeInspection is False
                                print("Inspection Started") 
                                self.InspectionTimeStart = time.time()

                                self.mergeframe3 = cv2.remap(self.mergeframe3, self.inspection_config.map1[3], self.inspection_config.map2[3], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                                self.combinedImage = warpTwoImages_template(self.homography_blank_canvas, self.mergeframe3, self.H3)
                                                                            
                                self.combinedImage = cv2.warpPerspective(self.combinedImage, self.planarizeTransform, (int(self.homography_size[1]), int(self.homography_size[0])))
                                self.combinedImage = cv2.resize(self.combinedImage, (self.homography_size[1], int(self.homography_size[0]/(1.26))))#1.48 for the qt, 1.26 for the aspect ratio

                                # Crop the image scaled for each part
                                
                                self.part3Crop = self.combinedImage_scaled[self.part3Crop_YPos : self.part3Crop_YPos + self.part_height_offset, 0 : self.homography_size[1]]
                                

                                #Need to convert to BGR for SAHI Inspection
                                self.part1Crop = cv2.cvtColor(self.part1Crop, cv2.COLOR_RGB2BGR)
                                self.part2Crop = cv2.cvtColor(self.part2Crop, cv2.COLOR_RGB2BGR)
                                self.part3Crop = cv2.cvtColor(self.part3Crop, cv2.COLOR_RGB2BGR)
                                self.part4Crop = cv2.cvtColor(self.part4Crop, cv2.COLOR_RGB2BGR)
                                self.part5Crop = cv2.cvtColor(self.part5Crop, cv2.COLOR_RGB2BGR)

                                #Put the All the image into a list
                                self.InspectionImages[0] = self.part1Crop.copy()
                                self.InspectionImages[1] = self.part2Crop.copy()
                                self.InspectionImages[2] = self.part3Crop.copy()
                                self.InspectionImages[3] = self.part4Crop.copy()
                                self.InspectionImages[4] = self.part5Crop.copy()

                                print(f"Length of Inspection Images : {len(self.InspectionImages)}") 

                                #Emit　検査中 to the status signal

                                # # Do the inspection
                                for i in range(len(self.InspectionImages)):
                                    #Only do inspectino on the one with kouden sensor on
                                    if self.inspection_config.kouden_sensor[i] == 1:
                                        self.InspectionResult_ClipDetection[i] = get_sliced_prediction(
                                            self.InspectionImages[i], 
                                            self.hoodFR_clipDetectionModel, 
                                            slice_height=180, slice_width=1280, 
                                            overlap_height_ratio=0.0, overlap_width_ratio=0.3,
                                            postprocess_match_metric="IOS",
                                            postprocess_match_threshold=0.005,
                                            postprocess_class_agnostic=True,
                                            postprocess_type="GREEDYNMM",
                                            verbose=0,
                                            perform_standard_pred=True
                                        )

                                        #Crop image from 0 to 1024 width for the left and width-1024 to 1024 for the right
                                        self.InspectionImages_endSegmentation_Left[i] = self.InspectionImages[i][:, :1024, :]
                                        self.InspectionImages_endSegmentation_Right[i] = self.InspectionImages[i][:, -1024:, :]

                                        self.InspectionResult_EndSegmentation_Left[i] = self.hoodFR_endSegmentationModel(source=self.InspectionImages_endSegmentation_Left[i], conf=0.5, imgsz=960, verbose=False)
                                        self.InspectionResult_EndSegmentation_Right[i] = self.hoodFR_endSegmentationModel(source=self.InspectionImages_endSegmentation_Right[i], conf=0.5, imgsz=960, verbose=False)

                                        self.InspectionImages[i], self.InspectionResult_PitchMeasured[i], self.InspectionResult_PitchResult[i], self.InspectionResult_DetectionID[i], self.InspectionResult_Status[i], self.InspectionResult_NGReason[i] = partcheck(self.InspectionImages[i], 
                                                                                                                                                                                                                                  self.InspectionResult_ClipDetection[i].object_prediction_list,
                                                                                                                                                                                                                                  self.InspectionResult_EndSegmentation_Left[i],
                                                                                                                                                                                                                                  self.InspectionResult_EndSegmentation_Right[i])
                                    else:
                                        #Make pure black image
                                        self.InspectionImages[i] = np.full((24, 1771, 3), (10, 10, 20), dtype=np.uint8)
                                        self.InspectionResult_PitchMeasured[i] = None
                                        self.InspectionResult_PitchResult[i] = None
                                        self.InspectionResult_DetectionID[i] = None
                                        self.InspectionResult_Status[i] = None
                                    
                                    # print(self.InspectionResult_Status[i])
                                    # print(self.InspectionResult_DetectionID[i])

                                    if self.InspectionResult_Status[i] == "OK":
                                        self.ethernet_status_green_hold_status[i] = 1
                                        self.ethernet_status_red_tenmetsu_status[i] = 0
                                        self.ethernet_status_red_hold_status[i] = 0

                                        self.inspection_config.current_numofPart[self.inspection_config.widget][0] += 1
                                        self.inspection_config.today_numofPart[self.inspection_config.widget][0] += 1

                                        #Play konpou sound if the current_numofPart is dividable by 25
                                        if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 25 == 0 and self.inspection_config.current_numofPart[self.inspection_config.widget][0] != 0:
                                            play_konpou_sound()

                                        self.InspectionStatus[i] = "OK"

                                    elif self.InspectionResult_Status[i] == "NG":
                                        self.ethernet_status_green_hold_status[i] = 0
                                        self.ethernet_status_red_tenmetsu_status[i] = 0
                                        self.ethernet_status_red_hold_status[i] = 1
                                        self.inspection_config.current_numofPart[self.inspection_config.widget][1] += 1
                                        self.inspection_config.today_numofPart[self.inspection_config.widget][1] += 1

                                        self.InspectionStatus[i] = "NG"

                                    else:
                                        self.ethernet_status_green_hold_status[i] = 0
                                        self.ethernet_status_red_tenmetsu_status[i] = 0
                                        self.ethernet_status_red_hold_status[i] = 0

                                    self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                                        numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget], 
                                        currentnumofPart = self.inspection_config.current_numofPart[self.inspection_config.widget],
                                        deltaTime = 0.0,
                                        kensainName = self.inspection_config.kensainNumber, 
                                        detected_pitch_str = self.InspectionResult_PitchMeasured[i], 
                                        delta_pitch_str = self.InspectionResult_DeltaPitch[i], 
                                        total_length=0,
                                        resultPitch = self.InspectionResult_PitchResult[i],
                                        status = self.InspectionStatus[i],
                                        NGreason = self.InspectionResult_NGReason[i])
                                    
                                    #save hole image
                                    self.save_image_hole(self.holeFrame1, False, "P1")
                                    self.save_image_hole(self.holeFrame2, False, "P2")
                                    self.save_image_hole(self.holeFrame3, False, "P3")
                                    self.save_image_hole(self.holeFrame4, False, "P4")
                                    self.save_image_hole(self.holeFrame5, False, "P5")
                                    
                                    self.hoodFR_InspectionStatus.emit(self.InspectionStatus)

                                self.save_image_result(self.part1Crop, self.InspectionImages[0], self.InspectionResult_Status[0], True, "P1")
                                self.save_image_result(self.part2Crop, self.InspectionImages[1], self.InspectionResult_Status[1], True, "P2")
                                self.save_image_result(self.part3Crop, self.InspectionImages[2], self.InspectionResult_Status[2], True, "P3")
                                self.save_image_result(self.part4Crop, self.InspectionImages[3], self.InspectionResult_Status[3], True, "P4")
                                self.save_image_result(self.part5Crop, self.InspectionImages[4], self.InspectionResult_Status[4], True, "P5")

                                self.InspectionImages[0] = self.downSampling(self.InspectionImages[0], width=1771, height=24)
                                self.InspectionImages[1] = self.downSampling(self.InspectionImages[1], width=1771, height=24)
                                self.InspectionImages[2] = self.downSampling(self.InspectionImages[2], width=1771, height=24)
                                self.InspectionImages[3] = self.downSampling(self.InspectionImages[3], width=1771, height=24)
                                self.InspectionImages[4] = self.downSampling(self.InspectionImages[4], width=1771, height=24)

                                self.hoodFR_InspectionResult_PitchMeasured.emit(self.InspectionResult_PitchMeasured)
                                self.hoodFR_InspectionResult_PitchResult.emit(self.InspectionResult_PitchResult)
                                print("Inspection Finished")
                                #Remember that list is mutable
                                self.ethernet_status_red_tenmetsu_status_prev = self.ethernet_status_red_tenmetsu_status_prev.copy()
                                self.ethernet_status_green_hold_status_prev = self.ethernet_status_green_hold_status.copy()
                                self.ethernet_status_red_hold_status_prev = self.ethernet_status_red_hold_status.copy()

                                self.InspectionImages_prev[0] = self.InspectionImages[0]
                                self.InspectionImages_prev[1] = self.InspectionImages[1]
                                self.InspectionImages_prev[2] = self.InspectionImages[2]
                                self.InspectionImages_prev[3] = self.InspectionImages[3]
                                self.InspectionImages_prev[4] = self.InspectionImages[4]

                                self.InspectionResult_PitchMeasured_prev = self.InspectionResult_PitchMeasured.copy()
                                self.InspectionResult_PitchResult_prev = self.InspectionResult_PitchResult.copy()
                                self.InspectionStatus_prev = self.InspectionStatus.copy()

                                self.ethernet_status_red_tenmetsu.emit(self.ethernet_status_red_tenmetsu_status)
                                self.ethernet_status_green_hold.emit(self.ethernet_status_green_hold_status)
                                self.ethernet_status_red_hold.emit(self.ethernet_status_red_hold_status)

                                self.part1Cam.emit(self.converQImageRGB(self.InspectionImages[0]))
                                self.part2Cam.emit(self.converQImageRGB(self.InspectionImages[1]))
                                self.part3Cam.emit(self.converQImageRGB(self.InspectionImages[2]))
                                self.part4Cam.emit(self.converQImageRGB(self.InspectionImages[3]))
                                self.part5Cam.emit(self.converQImageRGB(self.InspectionImages[4]))

                                # self.hoodFR_InspectionStatus.emit(self.InspectionStatus)

                #emit the ethernet 
                self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
                self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)
            
                self.ethernet_status_red_tenmetsu.emit(self.ethernet_status_red_tenmetsu_status)
                self.ethernet_status_green_hold.emit(self.ethernet_status_green_hold_status)
                self.ethernet_status_red_hold.emit(self.ethernet_status_red_hold_status)

                # Emit status based on the red tenmetsu status

                
                self.hoodFR_InspectionStatus.emit(self.InspectionStatus)

                # Emit the hole detection
                self.hoodFR_HoleStatus.emit(self.DetectionResult_HoleDetection)

        self.msleep(1)


    def setCounterFalse(self):
        self.inspection_config.furyou_plus = False
        self.inspection_config.furyou_minus = False
        self.inspection_config.kansei_plus = False
        self.inspection_config.kansei_minus = False
        self.inspection_config.furyou_plus_10 = False
        self.inspection_config.furyou_minus_10 = False
        self.inspection_config.kansei_plus_10 = False
        self.inspection_config.kansei_minus_10 = False

    def manual_adjustment(self, currentPart, Totalpart,
                          furyou_plus, furyou_minus, 
                          furyou_plus_10, furyou_minus_10,
                          kansei_plus, kansei_minus,
                          kansei_plus_10, kansei_minus_10):
        
        ok_count_current = currentPart[0]
        ng_count_current = currentPart[1]
        ok_count_total = Totalpart[0]
        ng_count_total = Totalpart[1]
        
        if furyou_plus:
            ng_count_current += 1
            ng_count_total += 1

        if furyou_plus_10:
            ng_count_current += 10
            ng_count_total += 10

        if furyou_minus and ng_count_current > 0 and ng_count_total > 0:
            ng_count_current -= 1
            ng_count_total -= 1
        
        if furyou_minus_10 and ng_count_current > 9 and ng_count_total > 9:
            ng_count_current -= 10
            ng_count_total -= 10

        if kansei_plus:
            ok_count_current += 1
            ok_count_total += 1

        if kansei_plus_10:
            ok_count_current += 10
            ok_count_total += 10

        if kansei_minus and ok_count_current > 0 and ok_count_total > 0:
            ok_count_current -= 1
            ok_count_total -= 1

        if kansei_minus_10 and ok_count_current > 9 and ok_count_total > 9:
            ok_count_current -= 10
            ok_count_total -= 10

        self.setCounterFalse()

        self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                numofPart = [ok_count_total, ng_count_total], 
                currentnumofPart = [ok_count_current, ng_count_current],
                deltaTime = 0.0,
                kensainName = self.inspection_config.kensainNumber, 
                detected_pitch_str = "MANUAL", 
                delta_pitch_str = "MANUAL", 
                total_length=0,
                resultPitch = "MANUAL",
                status = "MANUAL",
                NGreason = "MANUAL")

        return [ok_count_current, ng_count_current], [ok_count_total, ng_count_total]
    
    
    def save_result_database(self, partname, numofPart, 
                             currentnumofPart, deltaTime, 
                             kensainName, detected_pitch_str, 
                             delta_pitch_str, total_length, 
                             resultPitch, status, NGreason):
        # Ensure all inputs are strings or compatible types

        timestamp = datetime.now()
        timestamp_date = timestamp.strftime("%Y%m%d")
        timestamp_hour = timestamp.strftime("%H:%M:%S")

        partname = str(partname)
        numofPart = str(numofPart)
        currentnumofPart = str(currentnumofPart)
        timestamp_hour = str(timestamp_hour)
        timestamp_date = str(timestamp_date)
        deltaTime = float(deltaTime)  # Ensure this is a float
        kensainName = str(kensainName)
        detected_pitch_str = str(detected_pitch_str)
        delta_pitch_str = str(delta_pitch_str)
        total_length = float(total_length)  # Ensure this is a float
        resultPitch = str(resultPitch)
        status = str(status)
        NGreason = str(NGreason)

        self.cursor.execute('''
        INSERT INTO inspection_results (partname, numofPart, currentnumofPart, timestampHour, timestampDate, deltaTime, kensainName, detected_pitch, delta_pitch, total_length, resultpitch, status, NGreason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (partname, numofPart, currentnumofPart, timestamp_hour, timestamp_date, deltaTime, kensainName, detected_pitch_str, delta_pitch_str, total_length, resultPitch, status, NGreason))
        self.conn.commit()

        # Update the totatl part number (Maybe the day has been changed)
        for key, value in self.widget_dir_map.items():
            self.inspection_config.today_numofPart[key] = self.get_last_entry_total_numofPart(value)

        #Also save to mysql cursor
        self.mysql_cursor.execute('''
        INSERT INTO inspection_results (partName, numofPart, currentnumofPart, timestampHour, timestampDate, deltaTime, kensainName, detected_pitch, delta_pitch, total_length, resultpitch, status, NGreason)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (partname, numofPart, currentnumofPart, timestamp_hour, timestamp_date, deltaTime, kensainName, detected_pitch_str, delta_pitch_str, total_length, resultPitch, status, NGreason))
        self.mysql_conn.commit()


    def get_last_entry_currentnumofPart(self, part_name):
        self.cursor.execute('''
        SELECT currentnumofPart 
        FROM inspection_results 
        WHERE partName = ? 
        ORDER BY id DESC 
        LIMIT 1
        ''', (part_name,))
        
        row = self.cursor.fetchone()
        if row:
            currentnumofPart = eval(row[0])
            return currentnumofPart
        else:
            return [0, 0]
            
    def get_last_entry_total_numofPart(self, part_name):
        # Get today's date in yyyymmdd format
        today_date = datetime.now().strftime("%Y%m%d")

        self.cursor.execute('''
        SELECT numofPart 
        FROM inspection_results 
        WHERE partName = ? AND timestampDate = ? 
        ORDER BY id DESC 
        LIMIT 1
        ''', (part_name, today_date))
        
        row = self.cursor.fetchone()
        if row:
            numofPart = eval(row[0])  # Convert the string tuple to an actual tuple
            return numofPart
        else:
            return [0, 0]  # Default values if no entry is found    def get_last_entry_currentnumofPart(self, part_name):
        self.cursor.execute('''
        SELECT currentnumofPart 
        FROM inspection_results 
        WHERE partName = ? 
        ORDER BY id DESC 
        LIMIT 1
        ''', (part_name,))
        
        row = self.cursor.fetchone()
        if row:
            currentnumofPart = eval(row[0])
            return currentnumofPart
        else:
            return [0, 0]
            

    def draw_status_text_PIL(self, image, text, color, size = "normal", x_offset = 0, y_offset = 0):

        center_x = image.shape[1] // 2
        center_y = image.shape[0] // 2

        if size == "large":
            font_scale = 130.0

        if size == "normal":
            font_scale = 100.0

        elif size == "small":
            font_scale = 50.0
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(self.kanjiFontPath, font_scale)

        draw.text((center_x + x_offset, center_y + y_offset), text, font=font, fill=color)  
        # Convert back to BGR for OpenCV compatibility
        image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        return image

    def save_image(self, image):
        dir = "aikensa/inspection/" + self.widget_dir_map[self.inspection_config.widget]
        os.makedirs(dir, exist_ok=True)
        cv2.imwrite(dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png", image)

    def save_image_hole(self, image, BGR = True, id=None):
        if BGR:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        dir = "aikensa/inspection_results/" + self.widget_dir_map[self.inspection_config.widget] + "/" + datetime.now().strftime("%Y%m%d") +  "/hole/"
        os.makedirs(dir, exist_ok=True)
        cv2.imwrite(dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + id + ".png", image)


    def save_image_result(self, image_initial, image_result, result, BGR = True, id = None):
        if BGR:
            image_initial = cv2.cvtColor(image_initial, cv2.COLOR_RGB2BGR)
            image_result = cv2.cvtColor(image_result, cv2.COLOR_RGB2BGR)

        raw_dir = "aikensa/inspection_results/" + self.widget_dir_map[self.inspection_config.widget] + "/" + datetime.now().strftime("%Y%m%d") +  "/" +  str(result) + "/nama/"
        result_dir = "aikensa/inspection_results/" + self.widget_dir_map[self.inspection_config.widget] + "/" + datetime.now().strftime("%Y%m%d") +  "/" + str(result) + "/kekka/"
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        cv2.imwrite(raw_dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + id + ".png", image_initial)
        cv2.imwrite(result_dir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + id + ".png", image_result)

    def minitimerStart(self):
        self.timerStart_mini = time.time()
    
    def minitimerFinish(self, message = "OperationName"):
        self.timerFinish_mini = time.time()
        # self.fps_mini = 1/(self.timerFinish_mini - self.timerStart_mini)
        print(f"Time to {message} : {(self.timerFinish_mini - self.timerStart_mini) * 1000} ms")
        # print(f"FPS of {message} : {self.fps_mini}")

    def convertQImage(self, image):
        h, w, ch = image.shape
        bytesPerLine = ch * w
        processed_image = QImage(image.data, w, h, bytesPerLine, QImage.Format_BGR888)
        return processed_image
    
    def converQImageRGB(self, image):
        h, w, ch = image.shape
        bytesPerLine = ch * w
        processed_image = QImage(image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return processed_image
    
    def downScaledImage(self, image, scaleFactor=1.0):
        #create a copy of the image
        resized_image = cv2.resize(image, (0, 0), fx=1/scaleFactor, fy=1/scaleFactor, interpolation=cv2.INTER_LINEAR)
        return resized_image
    
    def downSampling(self, image, width=384, height=256):
        #create a copy of the image
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        return resized_image

    def load_matrix_from_yaml(self, filename):
        with open(filename, 'r') as file:
            calibration_param = yaml.load(file, Loader=yaml.FullLoader)
            camera_matrix = np.array(calibration_param.get('camera_matrix'))
            distortion_coeff = np.array(calibration_param.get('distortion_coefficients'))
        return camera_matrix, distortion_coeff

    def initialize_model(self):
        #Change based on the widget
        hoodFR_holeDetectionModel = None
        hoodFR_clipDetectionModel = None
        hoodFR_hanireDetectionModel = None
        hoodFR_endSegmentationModel = None

        #Classification Model
        path_hoodFR_clipDetectionModel = "./aikensa/models/65820W030P_CLIP.pt"
        path_hoodFR_holeDetectionModel = "./aikensa/models/65820W030P_MIZUANA.pt"
        path_P658207YA0A_clipDetectionModel = "./aikensa/models/658207YA0A_CLIP.pt"
        #Segmentation Model
        path_hoodFR_endSegmentationModel = "./aikensa/models/65820W030P_END_SEGMENTATION.pt"


        if os.path.exists(path_hoodFR_holeDetectionModel):
            hoodFR_holeDetectionModel = YOLO(path_hoodFR_holeDetectionModel)
        
        if os.path.exists(path_hoodFR_clipDetectionModel):
            hoodFR_clipDetectionModel = AutoDetectionModel.from_pretrained(model_type="yolov8",
                                                                            model_path=path_hoodFR_clipDetectionModel,
                                                                            confidence_threshold=0.5,
                                                                            device="cuda:0")
        if os.path.exists(path_hoodFR_endSegmentationModel):
            hoodFR_endSegmentationModel = YOLO(path_hoodFR_endSegmentationModel)
        
        if os.path.exists(path_P658207YA0A_clipDetectionModel):
            P658207YA0A_clipDetectionModel = AutoDetectionModel.from_pretrained(model_type="yolov8",
                                                                            model_path=path_P658207YA0A_clipDetectionModel,
                                                                            confidence_threshold=0.5,
                                                                            device="cuda:0")

        self.hoodFR_holeDetectionModel = hoodFR_holeDetectionModel
        self.hoodFR_clipDetectionModel = hoodFR_clipDetectionModel
        self.hoodFR_hanireDetectionModel = hoodFR_hanireDetectionModel
        self.hoodFR_endSegmentationModel = hoodFR_endSegmentationModel

        self.P658207YA0A_clipDetectionModel = P658207YA0A_clipDetectionModel


    def stop(self):
        self.running = False
        print("Releasing all cameras.")
        self.release_all_camera()
        self.running = False
        print("Inspection thread stopped.")

    def add_columns(self, cursor, table_name, columns):
        for column_name, column_type in columns:
            try:
                cursor.execute(f'''
                ALTER TABLE {table_name}
                ADD COLUMN {column_name} {column_type};
                ''')
                print(f"Added column: {column_name}")
            except sqlite3.OperationalError as e:
                print(f"Could not add column {column_name}: {e}")
