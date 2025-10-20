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

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

from aikensa.camscripts.cam_init import initialize_camera
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard , calculatecameramatrix, warpTwoImages, calculateHomography_template, warpTwoImages_template
from aikensa.opencv_imgprocessing.arucoplanarize import planarize, planarize_image
from dataclasses import dataclass, field
from typing import List, Tuple

from aikensa.parts_config.sound import play_do_sound, play_picking_sound, play_re_sound, play_mi_sound, play_alarm_sound, play_konpou_sound, play_keisoku_sound

from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image

from aikensa.scripts.scripts import list_to_16bit_int, load_register_map, invert_16bit_int, random_list
from aikensa.scripts.scripts_img_processing import crop_parts

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

    AGC_InspectionStatus = pyqtSignal(list)

    holding_register_signal = pyqtSignal(list)

    today_numofPart_signal = pyqtSignal(list)
    current_numofPart_signal = pyqtSignal(list)

    partNumber_signal = pyqtSignal(int)

    requestModbusWrite = pyqtSignal(int, list)

    def __init__(self, inspection_config: InspectionConfig = None, modbus_thread=None):
        super(InspectionThread, self).__init__()
        self.running = True

        if inspection_config is None:
            self.inspection_config = InspectionConfig()    
        else:
            self.inspection_config = inspection_config

        self.modbus_thread = modbus_thread

        if self.modbus_thread is not None:
            self.modbus_thread.holdingUpdated.connect(self.on_holding_update)
            self.requestModbusWrite.connect(self.modbus_thread.write_holding_registers)


        self.kanjiFontPath = "aikensa/font/NotoSansJP-ExtraBold.ttf"

        self.multiCam_stream = False

        self.cap_cam = None
        self.cap_cam0 = None
        self.camFrame = None

        self.homography_template = None
        self.homography_matrix1 = None

        self.homography_template_scaled = None
        self.homography_matrix1_scaled = None

        self.H1 = None

        self.H1_scaled = None

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

        self.part_height_offset = 205
        self.part_height_offset_scaled = int(self.part_height_offset//self.scale_factor)

        self.part_height_offset_hoodFR = 140
        self.part_height_offset_hoodFR_scaled = int(self.part_height_offset_hoodFR//self.scale_factor)

        self.part_height_offset_nissanhoodFR = 180
        self.part_height_offset_nissanhoodFR_scaled = int(self.part_height_offset_nissanhoodFR//self.scale_factor)

        self.dailyTenken_cropWidth = 950
        self.dailyTenken_cropWidth_scaled = int(self.dailyTenken_cropWidth//self.scale_factor)

        self.qtWindowWidth = 1701
        self.qtWindowHeight = 109
        
        self.part_crops = None

        self.J30LH_part1_Crop = None
        self.J30LH_part2_Crop = None
        self.J30LH_part3_Crop = None
        self.J30LH_part4_Crop = None
        self.J30LH_part5_Crop = None

        self.J30RH_part1_Crop = None
        self.J30RH_part2_Crop = None
        self.J30RH_part3_Crop = None
        self.J30RH_part4_Crop = None
        self.J30RH_part5_Crop = None

        self.J59JLH_part1_Crop = None
        self.J59JLH_part2_Crop = None
        self.J59JLH_part3_Crop = None
        self.J59JLH_part4_Crop = None
        self.J59JLH_part5_Crop = None

        self.J59JRH_part1_Crop = None
        self.J59JRH_part2_Crop = None
        self.J59JRH_part3_Crop = None
        self.J59JRH_part4_Crop = None
        self.J59JRH_part5_Crop = None

        self.J30RH_cropstart = 50
        self.J30RH_part_height_offset = 205
        self.J30RH_part1_Crop_YPos = 170
        self.J30RH_part2_Crop_YPos = 488
        self.J30RH_part3_Crop_YPos = 805
        self.J30RH_part4_Crop_YPos = 1130
        self.J30RH_part5_Crop_YPos = 1445

        self.height_hole_offset = int(120//self.scale_factor_hole)
        self.width_hole_offset = int(370//self.scale_factor_hole)

        self.timerStart = None
        self.timerFinish = None
        self.fps = None

        self.timerStart_mini = None
        self.timerFinish_mini = None
        self.fps_mini = None

        self.InspectionImages = [None]*5

        self.SetInspectionImages = [None]*5
        self.PartInspectionImages = [None]*5

        self.InspectionResult_DetectionID = [None]*5
        self.InspectionResult_DetectionID_int = None

        self.InspectionResult_SetID_OK = [None]*5
        self.InspectionResult_SetID_OK_int = None

        self.InspectionResult_SetID_NG = [None]*5
        self.InspectionResult_SetID_NG_int = [None]*5

        self.InspectionResult_TapeID_OK = [None]*5
        self.InspectionResult_TapeID_OK_int = [None]*5

        self.InspectionResult_TapeID_NG = [None]*5
        self.InspectionResult_TapeID_NG_int = [None]*5

        self.InspectionResult_NGReason = [None]*5
        
        self.InspectionStatus = [None]*5

        self.widget_dir_map = {
            5: "J59JRH",
            6: "J59JLH",
            7: "J30RH",
            8: "J30LH",
        }

        self.InspectionWaitTime = 5.0
        self.InspectionTimeStart = None

        self.test = 0
        self.firstTimeInspection = True

        self.partNumber = None
        self.partNumber_prev = None
        self.serialNumber_front = None
        self.serialNumber_back = None
        self.InstructionCode = None
        self.InstructionCode_prev = None

        # "Read mysql id and password from yaml file"
        with open("aikensa/mysql/id.yaml") as file:
            credentials = yaml.load(file, Loader=yaml.FullLoader)
            self.mysqlID = credentials["id"]
            self.mysqlPassword = credentials["pass"]
            self.mysqlHost = credentials["host"]
            self.mysqlHostPort = credentials["port"]

        self.holding_register_path = "./aikensa/modbus/holding_register_map.yaml"
        self.holding_register_map = load_register_map(self.holding_register_path)
        self.camera_angle = 180.65
        self.last_valid_part_number = 0


    @pyqtSlot(dict)
    def on_holding_update(self, reg_dict):
        # Only called whenever the Modbus thread emits new data.
        self.partNumber = reg_dict.get(50, 0)
        # self.partNumber = 8 #For testing only
        self.serialNumber_front = reg_dict.get(62, 0)
        self.serialNumber_back  = reg_dict.get(63, 0)
        self.InstructionCode    = reg_dict.get(100, 0)

        print(f"Part Number: {self.partNumber}")
        print(f"Serial Number Front: {self.serialNumber_front}")
        print(f"Serial Number Back:  {self.serialNumber_back}")
        print(f"Instruction Code:     {self.InstructionCode}")

    def initialize_single_camera(self, camID):
        if self.cap_cam is not None:
            self.cap_cam.release()  # Release the previous camera if it's already open
            print(f"Camera {self.inspection_config.cameraID} released.")

        if camID == -1:
            print("No valid camera selected, displaying placeholder.")
            self.cap_cam = None  # No camera initialized
            # self.frame = self.create_placeholder_image()
        else:
            print(f"Initializing camera with ID {camID}")
            self.cap_cam = initialize_camera(camID)
            #find the center of the image
            self.center_image = (self.cap_cam.get(cv2.CAP_PROP_FRAME_WIDTH) // 2, self.cap_cam.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)

            if not self.cap_cam.isOpened():
                print(f"Failed to open camera with ID {camID}")
                self.cap_cam = None
            else:
                print(f"Initialized Camera on ID {camID}")

    def release_camera(self):
        if self.cap_cam is not None:
            self.cap_cam.release()
            self.cap_cam = None
            print("Camera released.")

        if self.cap_cam0 is not None:
            self.cap_cam0.release()
            self.cap_cam0 = None
            print("Camera 0 released.")

    def run(self):
        #initialize the database
        if not os.path.exists("./aikensa/inspection_results"):
            os.makedirs("./aikensa/inspection_results")

        self.conn = sqlite3.connect('./aikensa/inspection_results/agc_database_results.db')
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
            status TEXT,
            NGreason TEXT,
            PPMS TEXT
        )
        ''')

        self.conn.commit()
        # #Initialize connection to mysql server if available
        # try:
        #     self.mysql_conn = mysql.connector.connect(
        #         host=self.mysqlHost,
        #         user=self.mysqlID,
        #         password=self.mysqlPassword,
        #         port=self.mysqlHostPort,
        #         database="AIKENSAresults"
        #     )
        #     print(f"Connected to MySQL database at {self.mysqlHost}")
        # except Exception as e:
        #     print(f"Error connecting to MySQL database: {e}")
        #     self.mysql_conn = None

        # #try adding data to the schema in mysql
        # if self.mysql_conn is not None:
        #     self.mysql_cursor = self.mysql_conn.cursor()
        #     self.mysql_cursor.execute('''
        #         CREATE TABLE IF NOT EXISTS AGC_tapehari_inspection_results (
        #             id INTEGER PRIMARY KEY AUTO_INCREMENT,
        #             partName TEXT,
        #             numofPart TEXT,
        #             currentnumofPart TEXT,
        #             timestampHour TEXT,
        #             timestampDate TEXT,
        #             deltaTime REAL,
        #             kensainName TEXT,
        #             status TEXT,
        #             NGreason TEXT,
        #             PPMS TEXT
        #         )
        #     ''')
        #     self.mysql_conn.commit()

        #print thread started
        print("Inspection Thread Started")
        self.initialize_model()
        print("AI Model Initialized")

        self.current_cameraID = self.inspection_config.cameraID
        self.initialize_single_camera(self.current_cameraID)
        self.initialize_single_camera(0)

        self.load_crop_coords("aikensa/cameraconfig/part_pos.yaml")

        while self.running:

            if self.inspection_config.widget == 0:
                self.inspection_config.cameraID = -1

            if self.partNumber is not None:
                self.handle_part_number_update()

            if self.inspection_config.widget in [0, 5, 6, 7, 8]:
                # render the camera first
                if self.cap_cam is None:
                    print("Camera 0 is not initialized, skipping frame capture.")
                    continue
                # Read the frame from the camera
                _, self.camFrame = self.cap_cam.read()
                # angle = self.camera_angle
                # M = cv2.getRotationMatrix2D(self.center_image, angle, 1.0)  # 1.0 is the scale factor
                # self.camFrame = cv2.warpAffine(self.camFrame, M, (self.camFrame.shape[1], self.camFrame.shape[0]))
                #rotate camera by 180 degrees
                self.camFrame = cv2.rotate(self.camFrame, cv2.ROTATE_180)

            #J30 RH Inspection
            if self.inspection_config.widget == 7:
                self.handle_adjustments_and_counterreset()
                self.part_crops = crop_parts(
                    img=self.camFrame,
                    crop_start=int(self.J30RH_cropstart),
                    crop_height=int(self.J30RH_part_height_offset),
                    crop_y_positions=[
                        int(self.J30RH_part1_Crop_YPos),
                        int(self.J30RH_part2_Crop_YPos),
                        int(self.J30RH_part3_Crop_YPos),
                        int(self.J30RH_part4_Crop_YPos),
                        int(self.J30RH_part5_Crop_YPos),
                    ],
                )
                self.part1Crop, self.part2Crop, self.part3Crop, self.part4Crop, self.part5Crop = self.part_crops

                self.process_and_emit_parts(width=self.qtWindowWidth, height=self.qtWindowHeight)

                if self.firstTimeInspection is False:
                    if self.inspection_config.doInspection is False:
                        self.InspectionTimeStart = time.time()
                        self.firstTimeInspection = True
                        self.inspection_config.doInspection = True
                
                self.partNumber_signal.emit(self.partNumber)

                if self.InstructionCode != 0:
                    self.InspectionImages[0] = self.part1Crop
                    self.InspectionImages[1] = self.part2Crop
                    self.InspectionImages[2] = self.part3Crop
                    self.InspectionImages[3] = self.part4Crop
                    self.InspectionImages[4] = self.part5Crop

                if self.inspection_config.doInspection is True:
                    self.inspection_config.doInspection = False
                    #Save images, time and part number to ./training_images
                    if not os.path.exists("./aikensa/training_images"):
                        os.makedirs("./aikensa/training_images")
                        #Use time for the file name
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    #save InspectionImages with cv2.write
                    self.InspectionImages[0] = self.part1Crop
                    self.InspectionImages[1] = self.part2Crop
                    self.InspectionImages[2] = self.part3Crop
                    self.InspectionImages[3] = self.part4Crop
                    self.InspectionImages[4] = self.part5Crop

                    for i, img in enumerate(self.InspectionImages):
                        if img is not None:
                            filename = f"./aikensa/training_images/part{i+1}_{timestamp}.jpg"
                            cv2.imwrite(filename, img)
                            print(f"Saved {filename}")

                if self.InstructionCode == 0:
                    self.requestModbusWrite.emit(self.holding_register_map["return_state_code"], [0])
                    if self.InstructionCode_prev == 0:
                        pass  # Skip, already processed
                    else:
                        self.InstructionCode_prev = self.InstructionCode

                        self.InspectionResult_DetectionID = [0, 0, 0, 0, 0]

                        self.InspectionResult_SetID_OK = [0, 0, 0, 0, 0]
                        self.InspectionResult_SetID_NG = [0, 0, 0, 0, 0]

                        self.InspectionResult_TapeID_OK = [0, 0, 0, 0, 0]
                        self.InspectionResult_TapeID_NG = [0, 0, 0, 0, 0]

                        self.InspectionResult_DetectionID_int = list_to_16bit_int(self.InspectionResult_DetectionID)
                        self.InspectionResult_SetID_OK_int = list_to_16bit_int(self.InspectionResult_SetID_OK)
                        self.InspectionResult_SetID_NG_int = list_to_16bit_int(self.InspectionResult_SetID_NG)
                        self.InspectionResult_TapeID_OK_int = list_to_16bit_int(self.InspectionResult_TapeID_OK)
                        self.InspectionResult_TapeID_NG_int = list_to_16bit_int(self.InspectionResult_TapeID_NG)

                        print(f"Inspection Result Detection ID: {self.InspectionResult_DetectionID_int}")
                        print(f"Inspection Result Set ID: {self.InspectionResult_SetID_OK_int}")
                        print(f"Inspection Result Tape ID: {self.InspectionResult_TapeID_OK_int}")
                        
                        # Send all zeros to the holding registers
                        self.requestModbusWrite.emit(self.holding_register_map["return_serialNumber_front"], [self.serialNumber_front])
                        self.requestModbusWrite.emit(self.holding_register_map["return_serialNumber_back"], [self.serialNumber_back])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_set_partexist"], [self.InspectionResult_DetectionID_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_set_results_OK"], [self.InspectionResult_SetID_OK_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_set_results_NG"], [self.InspectionResult_SetID_NG_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_tapeinspection_partexist"], [self.InspectionResult_DetectionID_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_tapeinspection_results_OK"], [self.InspectionResult_TapeID_OK_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_tapeinspection_results_NG"], [self.InspectionResult_TapeID_NG_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_state_code"], [0])
                        print("All zeros Emitted to Holding Registers")
                        #wait
                        time.sleep(0.5)

                # 1 = Set Inspection
                if self.InstructionCode == 1:
                    if self.InstructionCode_prev == 1:
                        pass
                    else:
                        self.InstructionCode_prev = self.InstructionCode

                        self.InspectionResult_DetectionID = [0, 0, 0, 0, 0]
                        # self.InspectionResult_SetID_OK = random_list(5) #Dummy values for testing
                        self.InspectionResult_SetID_OK = [1, 1, 1, 1, 1]
                        self.InspectionResult_SetID_NG = [1 - x for x in self.InspectionResult_SetID_OK]  # Invert the OK values for NG
                        
                        self.InspectionResult_DetectionID_int = list_to_16bit_int(self.InspectionResult_DetectionID)
                        self.InspectionResult_SetID_OK_int = list_to_16bit_int(self.InspectionResult_SetID_OK)
                        self.InspectionResult_SetID_NG_int = list_to_16bit_int(self.InspectionResult_SetID_NG)

                        print(f"Inspection Result Detection ID: {self.InspectionResult_DetectionID}")
                        print(f"Inspection Result Set OK ID: {self.InspectionResult_SetID_OK}")
                        print(f"Inspection Result Set NG ID: {self.InspectionResult_SetID_NG}")

                        #Emit the inspection result and serial number to holding registers
                        self.requestModbusWrite.emit(self.holding_register_map["return_serialNumber_front"],[self.serialNumber_front])
                        self.requestModbusWrite.emit(self.holding_register_map["return_serialNumber_back"], [self.serialNumber_back])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_set_partexist"], [self.InspectionResult_DetectionID_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_set_results_OK"], [self.InspectionResult_SetID_OK_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_set_results_NG"], [self.InspectionResult_SetID_NG_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_state_code"],[1])
                        print("Inspection Result Set ID Emitted")
                        # Wait for 0.5 sec then emit return state code of 0 to show that it can accept the next instruction
                        time.sleep(0.5)
                        # self.requestModbusWrite.emit(self.holding_register_map["return_state_code"], [0])
                        # print("0 State Code Emitted, ready for next instruction")

                        if not os.path.exists("./aikensa/training_images/set"):
                            os.makedirs("./aikensa/training_images/set")
                            #Use time for the file name
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        #save InspectionImages with cv2.write
                        self.InspectionImages[0] = self.part1Crop
                        self.InspectionImages[1] = self.part2Crop
                        self.InspectionImages[2] = self.part3Crop
                        self.InspectionImages[3] = self.part4Crop
                        self.InspectionImages[4] = self.part5Crop

                        for i, img in enumerate(self.InspectionImages):
                            if img is not None:
                                filename = f"./aikensa/training_images/set/part{i+1}_{timestamp}.jpg"
                                cv2.imwrite(filename, img)
                                print(f"Saved {filename}")

                # 2 = Tape Inspection
                if self.InstructionCode == 2:
                    if self.InstructionCode_prev == 2:
                        pass
                    else:
                        self.InstructionCode_prev = self.InstructionCode
                        self.InspectionResult_DetectionID = [0, 0, 0, 0, 0]
                        # self.InspectionResult_TapeID_OK = random_list(5) #Dummy values for testing
                        self.InspectionResult_TapeID_OK = [1, 1, 1, 1, 1]
                        self.InspectionResult_TapeID_NG = [1 - x for x in self.InspectionResult_TapeID_OK]

                        self.InspectionResult_DetectionID_int = list_to_16bit_int(self.InspectionResult_DetectionID)
                        self.InspectionResult_TapeID_OK_int = list_to_16bit_int(self.InspectionResult_TapeID_OK)
                        self.InspectionResult_TapeID_NG_int = list_to_16bit_int(self.InspectionResult_TapeID_NG)

                        print(f"Inspection Result Detection ID: {self.InspectionResult_DetectionID}")
                        print(f"Inspection Result Tape OK ID: {self.InspectionResult_TapeID_OK}")
                        print(f"Inspection Result Tape NG ID: {self.InspectionResult_TapeID_NG}")

                        #Emit the inspection result and serial number to holding registers
                        self.requestModbusWrite.emit(self.holding_register_map["return_serialNumber_front"], [self.serialNumber_front])
                        self.requestModbusWrite.emit(self.holding_register_map["return_serialNumber_back"], [self.serialNumber_back])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_tapeinspection_partexist"], [self.InspectionResult_DetectionID_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_tapeinspection_results_OK"], [self.InspectionResult_TapeID_OK_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_tapeinspection_results_NG"], [self.InspectionResult_TapeID_NG_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_state_code"], [2])
                        print("Inspection Result Tape ID Emitted")
                        # Wait for 0.5 sec then emit return state code of 0 to show that it can accept the next instruction
                        time.sleep(0.5)
                        # self.requestModbusWrite.emit(self.holding_register_map["return_state_code"], [0])
                        # print("0 State Code Emitted, ready for next instruction")

                        if not os.path.exists("./aikensa/training_images/tape"):
                            os.makedirs("./aikensa/training_images/tape")
                            #Use time for the file name
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        #save InspectionImages with cv2.write
                        self.InspectionImages[0] = self.part1Crop
                        self.InspectionImages[1] = self.part2Crop
                        self.InspectionImages[2] = self.part3Crop
                        self.InspectionImages[3] = self.part4Crop
                        self.InspectionImages[4] = self.part5Crop

                        for i, img in enumerate(self.InspectionImages):
                            if img is not None:
                                filename = f"./aikensa/training_images/tape/part{i+1}_{timestamp}.jpg"
                                cv2.imwrite(filename, img)
                                print(f"Saved {filename}")
                        
                #emit the ethernet 
                self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
                self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)
            
                self.AGC_InspectionStatus.emit(self.InspectionStatus)

            #J59 JLH Inspection
            if self.inspection_config.widget == 6:
                self.handle_adjustments_and_counterreset()

                self.part1Crop = self.crop_part(self.camFrame, "J59JLH_part1_Crop", out_w=1771, out_h=121)
                self.part2Crop = self.crop_part(self.camFrame, "J59JLH_part2_Crop", out_w=1771, out_h=121)
                self.part3Crop = self.crop_part(self.camFrame, "J59JLH_part3_Crop", out_w=1771, out_h=121)
                self.part4Crop = self.crop_part(self.camFrame, "J59JLH_part4_Crop", out_w=1771, out_h=121)
                self.part5Crop = self.crop_part(self.camFrame, "J59JLH_part5_Crop", out_w=1771, out_h=121)
                # cv2.imwrite("part1_crop_test.jpg", self.part1Crop)
                # cv2.imwrite("part2_crop_test.jpg", self.part2Crop)
                # cv2.imwrite("part3_crop_test.jpg", self.part3Crop)
                # cv2.imwrite("part4_crop_test.jpg", self.part4Crop)
                # cv2.imwrite("part5_crop_test.jpg", self.part5Crop)

                self.process_and_emit_parts(width=self.qtWindowWidth, height=self.qtWindowHeight)
                

                if self.firstTimeInspection is False:
                    if self.inspection_config.doInspection is False:
                        self.InspectionTimeStart = time.time()
                        self.firstTimeInspection = True
                        self.inspection_config.doInspection = True
                
                self.partNumber_signal.emit(self.partNumber)

                if self.InstructionCode != 0:
                    self.InspectionImages[0] = self.part1Crop
                    self.InspectionImages[1] = self.part2Crop
                    self.InspectionImages[2] = self.part3Crop
                    self.InspectionImages[3] = self.part4Crop
                    self.InspectionImages[4] = self.part5Crop

                if self.inspection_config.doInspection is True:
                    self.inspection_config.doInspection = False
                    #Save images, time and part number to ./training_images
                    if not os.path.exists("./aikensa/training_images"):
                        os.makedirs("./aikensa/training_images")
                        #Use time for the file name
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    #save InspectionImages with cv2.write
                    self.InspectionImages[0] = self.part1Crop
                    self.InspectionImages[1] = self.part2Crop
                    self.InspectionImages[2] = self.part3Crop
                    self.InspectionImages[3] = self.part4Crop
                    self.InspectionImages[4] = self.part5Crop

                    for i, img in enumerate(self.InspectionImages):
                        if img is not None:
                            filename = f"./aikensa/training_images/part{i+1}_{timestamp}.jpg"
                            cv2.imwrite(filename, img)
                            print(f"Saved {filename}")

                if self.InstructionCode == 0:
                    self.requestModbusWrite.emit(self.holding_register_map["return_state_code"], [0])
                    if self.InstructionCode_prev == 0:
                        pass  # Skip, already processed
                    else:
                        self.InstructionCode_prev = self.InstructionCode

                        self.InspectionResult_DetectionID = [0, 0, 0, 0, 0]

                        self.InspectionResult_SetID_OK = [0, 0, 0, 0, 0]
                        self.InspectionResult_SetID_NG = [0, 0, 0, 0, 0]

                        self.InspectionResult_TapeID_OK = [0, 0, 0, 0, 0]
                        self.InspectionResult_TapeID_NG = [0, 0, 0, 0, 0]

                        self.InspectionResult_DetectionID_int = list_to_16bit_int(self.InspectionResult_DetectionID)
                        self.InspectionResult_SetID_OK_int = list_to_16bit_int(self.InspectionResult_SetID_OK)
                        self.InspectionResult_SetID_NG_int = list_to_16bit_int(self.InspectionResult_SetID_NG)
                        self.InspectionResult_TapeID_OK_int = list_to_16bit_int(self.InspectionResult_TapeID_OK)
                        self.InspectionResult_TapeID_NG_int = list_to_16bit_int(self.InspectionResult_TapeID_NG)

                        print(f"Inspection Result Detection ID: {self.InspectionResult_DetectionID_int}")
                        print(f"Inspection Result Set ID: {self.InspectionResult_SetID_OK_int}")
                        print(f"Inspection Result Tape ID: {self.InspectionResult_TapeID_OK_int}")
                        
                        # Send all zeros to the holding registers
                        self.requestModbusWrite.emit(self.holding_register_map["return_serialNumber_front"], [self.serialNumber_front])
                        self.requestModbusWrite.emit(self.holding_register_map["return_serialNumber_back"], [self.serialNumber_back])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_set_partexist"], [self.InspectionResult_DetectionID_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_set_results_OK"], [self.InspectionResult_SetID_OK_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_set_results_NG"], [self.InspectionResult_SetID_NG_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_tapeinspection_partexist"], [self.InspectionResult_DetectionID_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_tapeinspection_results_OK"], [self.InspectionResult_TapeID_OK_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_tapeinspection_results_NG"], [self.InspectionResult_TapeID_NG_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_state_code"], [0])
                        print("All zeros Emitted to Holding Registers")
                        #wait
                        time.sleep(0.5)

                # 1 = Set Inspection
                if self.InstructionCode == 1:
                    if self.InstructionCode_prev == 1:
                        pass
                    else:
                        self.InstructionCode_prev = self.InstructionCode

                        self.InspectionResult_DetectionID = [0, 0, 0, 0, 0]
                        # self.InspectionResult_SetID_OK = random_list(5) #Dummy values for testing
                        self.InspectionResult_SetID_OK = [1, 1, 1, 1, 1]
                        self.InspectionResult_SetID_NG = [1 - x for x in self.InspectionResult_SetID_OK]  # Invert the OK values for NG
                        
                        self.InspectionResult_DetectionID_int = list_to_16bit_int(self.InspectionResult_DetectionID)
                        self.InspectionResult_SetID_OK_int = list_to_16bit_int(self.InspectionResult_SetID_OK)
                        self.InspectionResult_SetID_NG_int = list_to_16bit_int(self.InspectionResult_SetID_NG)

                        print(f"Inspection Result Detection ID: {self.InspectionResult_DetectionID}")
                        print(f"Inspection Result Set OK ID: {self.InspectionResult_SetID_OK}")
                        print(f"Inspection Result Set NG ID: {self.InspectionResult_SetID_NG}")

                        #Emit the inspection result and serial number to holding registers
                        self.requestModbusWrite.emit(self.holding_register_map["return_serialNumber_front"],[self.serialNumber_front])
                        self.requestModbusWrite.emit(self.holding_register_map["return_serialNumber_back"], [self.serialNumber_back])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_set_partexist"], [self.InspectionResult_DetectionID_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_set_results_OK"], [self.InspectionResult_SetID_OK_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_set_results_NG"], [self.InspectionResult_SetID_NG_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_state_code"],[1])
                        print("Inspection Result Set ID Emitted")
                        # Wait for 0.5 sec then emit return state code of 0 to show that it can accept the next instruction
                        time.sleep(0.5)
                        # self.requestModbusWrite.emit(self.holding_register_map["return_state_code"], [0])
                        # print("0 State Code Emitted, ready for next instruction")

                        if not os.path.exists("./aikensa/training_images/set"):
                            os.makedirs("./aikensa/training_images/set")
                            #Use time for the file name
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        #save InspectionImages with cv2.write
                        self.InspectionImages[0] = self.part1Crop
                        self.InspectionImages[1] = self.part2Crop
                        self.InspectionImages[2] = self.part3Crop
                        self.InspectionImages[3] = self.part4Crop
                        self.InspectionImages[4] = self.part5Crop

                        for i, img in enumerate(self.InspectionImages):
                            if img is not None:
                                filename = f"./aikensa/training_images/set/part{i+1}_{timestamp}.jpg"
                                cv2.imwrite(filename, img)
                                print(f"Saved {filename}")

                # 2 = Tape Inspection
                if self.InstructionCode == 2:
                    if self.InstructionCode_prev == 2:
                        pass
                    else:
                        self.InstructionCode_prev = self.InstructionCode
                        self.InspectionResult_DetectionID = [0, 0, 0, 0, 0]
                        # self.InspectionResult_TapeID_OK = random_list(5) #Dummy values for testing
                        self.InspectionResult_TapeID_OK = [1, 1, 1, 1, 1]
                        self.InspectionResult_TapeID_NG = [1 - x for x in self.InspectionResult_TapeID_OK]

                        self.InspectionResult_DetectionID_int = list_to_16bit_int(self.InspectionResult_DetectionID)
                        self.InspectionResult_TapeID_OK_int = list_to_16bit_int(self.InspectionResult_TapeID_OK)
                        self.InspectionResult_TapeID_NG_int = list_to_16bit_int(self.InspectionResult_TapeID_NG)

                        print(f"Inspection Result Detection ID: {self.InspectionResult_DetectionID}")
                        print(f"Inspection Result Tape OK ID: {self.InspectionResult_TapeID_OK}")
                        print(f"Inspection Result Tape NG ID: {self.InspectionResult_TapeID_NG}")

                        #Emit the inspection result and serial number to holding registers
                        self.requestModbusWrite.emit(self.holding_register_map["return_serialNumber_front"], [self.serialNumber_front])
                        self.requestModbusWrite.emit(self.holding_register_map["return_serialNumber_back"], [self.serialNumber_back])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_tapeinspection_partexist"], [self.InspectionResult_DetectionID_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_tapeinspection_results_OK"], [self.InspectionResult_TapeID_OK_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_tapeinspection_results_NG"], [self.InspectionResult_TapeID_NG_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_state_code"], [2])
                        print("Inspection Result Tape ID Emitted")
                        # Wait for 0.5 sec then emit return state code of 0 to show that it can accept the next instruction
                        time.sleep(0.5)
                        # self.requestModbusWrite.emit(self.holding_register_map["return_state_code"], [0])
                        # print("0 State Code Emitted, ready for next instruction")

                        if not os.path.exists("./aikensa/training_images/tape"):
                            os.makedirs("./aikensa/training_images/tape")
                            #Use time for the file name
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        #save InspectionImages with cv2.write
                        self.InspectionImages[0] = self.part1Crop
                        self.InspectionImages[1] = self.part2Crop
                        self.InspectionImages[2] = self.part3Crop
                        self.InspectionImages[3] = self.part4Crop
                        self.InspectionImages[4] = self.part5Crop

                        for i, img in enumerate(self.InspectionImages):
                            if img is not None:
                                filename = f"./aikensa/training_images/tape/part{i+1}_{timestamp}.jpg"
                                cv2.imwrite(filename, img)
                                print(f"Saved {filename}")
                        
                #emit the ethernet 
                self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
                self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)
            
                self.AGC_InspectionStatus.emit(self.InspectionStatus)

            #J59 JRH Inspection
            if self.inspection_config.widget == 5:
                self.handle_adjustments_and_counterreset()
                self.part1Crop = self.crop_part(self.camFrame, "J59JRH_part1_Crop", out_w=1771, out_h=121)
                self.part2Crop = self.crop_part(self.camFrame, "J59JRH_part2_Crop", out_w=1771, out_h=121)
                self.part3Crop = self.crop_part(self.camFrame, "J59JRH_part3_Crop", out_w=1771, out_h=121)
                self.part4Crop = self.crop_part(self.camFrame, "J59JRH_part4_Crop", out_w=1771, out_h=121)
                self.part5Crop = self.crop_part(self.camFrame, "J59JRH_part5_Crop", out_w=1771, out_h=121)

                self.process_and_emit_parts(width=self.qtWindowWidth, height=self.qtWindowHeight)

                if self.firstTimeInspection is False:
                    if self.inspection_config.doInspection is False:
                        self.InspectionTimeStart = time.time()
                        self.firstTimeInspection = True
                        self.inspection_config.doInspection = True
                
                self.partNumber_signal.emit(self.partNumber)

                if self.InstructionCode != 0:
                    self.InspectionImages[0] = self.part1Crop
                    self.InspectionImages[1] = self.part2Crop
                    self.InspectionImages[2] = self.part3Crop
                    self.InspectionImages[3] = self.part4Crop
                    self.InspectionImages[4] = self.part5Crop

                if self.inspection_config.doInspection is True:
                    self.inspection_config.doInspection = False
                    #Save images, time and part number to ./training_images
                    if not os.path.exists("./aikensa/training_images"):
                        os.makedirs("./aikensa/training_images")
                        #Use time for the file name
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    #save InspectionImages with cv2.write
                    self.InspectionImages[0] = self.part1Crop
                    self.InspectionImages[1] = self.part2Crop
                    self.InspectionImages[2] = self.part3Crop
                    self.InspectionImages[3] = self.part4Crop
                    self.InspectionImages[4] = self.part5Crop

                    for i, img in enumerate(self.InspectionImages):
                        if img is not None:
                            filename = f"./aikensa/training_images/part{i+1}_{timestamp}.jpg"
                            cv2.imwrite(filename, img)
                            print(f"Saved {filename}")

                if self.InstructionCode == 0:
                    self.requestModbusWrite.emit(self.holding_register_map["return_state_code"], [0])
                    if self.InstructionCode_prev == 0:
                        pass  # Skip, already processed
                    else:
                        self.InstructionCode_prev = self.InstructionCode

                        self.InspectionResult_DetectionID = [0, 0, 0, 0, 0]

                        self.InspectionResult_SetID_OK = [0, 0, 0, 0, 0]
                        self.InspectionResult_SetID_NG = [0, 0, 0, 0, 0]

                        self.InspectionResult_TapeID_OK = [0, 0, 0, 0, 0]
                        self.InspectionResult_TapeID_NG = [0, 0, 0, 0, 0]

                        self.InspectionResult_DetectionID_int = list_to_16bit_int(self.InspectionResult_DetectionID)
                        self.InspectionResult_SetID_OK_int = list_to_16bit_int(self.InspectionResult_SetID_OK)
                        self.InspectionResult_SetID_NG_int = list_to_16bit_int(self.InspectionResult_SetID_NG)
                        self.InspectionResult_TapeID_OK_int = list_to_16bit_int(self.InspectionResult_TapeID_OK)
                        self.InspectionResult_TapeID_NG_int = list_to_16bit_int(self.InspectionResult_TapeID_NG)

                        print(f"Inspection Result Detection ID: {self.InspectionResult_DetectionID_int}")
                        print(f"Inspection Result Set ID: {self.InspectionResult_SetID_OK_int}")
                        print(f"Inspection Result Tape ID: {self.InspectionResult_TapeID_OK_int}")
                        
                        # Send all zeros to the holding registers
                        self.requestModbusWrite.emit(self.holding_register_map["return_serialNumber_front"], [self.serialNumber_front])
                        self.requestModbusWrite.emit(self.holding_register_map["return_serialNumber_back"], [self.serialNumber_back])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_set_partexist"], [self.InspectionResult_DetectionID_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_set_results_OK"], [self.InspectionResult_SetID_OK_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_set_results_NG"], [self.InspectionResult_SetID_NG_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_tapeinspection_partexist"], [self.InspectionResult_DetectionID_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_tapeinspection_results_OK"], [self.InspectionResult_TapeID_OK_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_tapeinspection_results_NG"], [self.InspectionResult_TapeID_NG_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_state_code"], [0])
                        print("All zeros Emitted to Holding Registers")
                        #wait
                        time.sleep(0.5)

                # 1 = Set Inspection
                if self.InstructionCode == 1:
                    if self.InstructionCode_prev == 1:
                        pass
                    else:
                        self.InstructionCode_prev = self.InstructionCode
                        #resize to 512 512
                        # self.SetInspectionImages[0] = cv2.resize(self.part1Crop, (512, 512))
                        # self.SetInspectionImages[1] = cv2.resize(self.part2Crop, (512, 512))
                        # self.SetInspectionImages[2] = cv2.resize(self.part3Crop, (512, 512))
                        # self.SetInspectionImages[3] = cv2.resize(self.part4Crop, (512, 512))
                        # self.SetInspectionImages[4] = cv2.resize(self.part5Crop, (512, 512))

                        
                        # self.InspectionResult_SetID_OK[0] = self.AGCJ59JRH_setDetectionModel(cv2.cvtColor(self.SetInspectionImages[0], cv2.COLOR_BGR2RGB), stream=True, verbose=False)
                        # self.InspectionResult_SetID_OK[1] = self.AGCJ59JRH_setDetectionModel(cv2.cvtColor(self.SetInspectionImages[1], cv2.COLOR_BGR2RGB), stream=True, verbose=False)  
                        # self.InspectionResult_SetID_OK[2] = self.AGCJ59JRH_setDetectionModel(cv2.cvtColor(self.SetInspectionImages[2], cv2.COLOR_BGR2RGB), stream=True, verbose=False)
                        # self.InspectionResult_SetID_OK[3] = self.AGCJ59JRH_setDetectionModel(cv2.cvtColor(self.SetInspectionImages[3], cv2.COLOR_BGR2RGB), stream=True, verbose=False)
                        # self.InspectionResult_SetID_OK[4] = self.AGCJ59JRH_setDetectionModel(cv2.cvtColor(self.SetInspectionImages[4], cv2.COLOR_BGR2RGB), stream=True, verbose=False)
                        
                        # self.InspectionResult_SetID_OK[0] = list(self.InspectionResult_SetID_OK[0])[0].probs.data.argmax().item()
                        # self.InspectionResult_SetID_OK[1] = list(self.InspectionResult_SetID_OK[1])[0].probs.data.argmax().item()
                        # self.InspectionResult_SetID_OK[2] = list(self.InspectionResult_SetID_OK[2])[0].probs.data.argmax().item()
                        # self.InspectionResult_SetID_OK[3] = list(self.InspectionResult_SetID_OK[3])[0].probs.data.argmax().item()
                        # self.InspectionResult_SetID_OK[4] = list(self.InspectionResult_SetID_OK[4])[0].probs.data.argmax().item()


                        for i, crop in enumerate([
                            self.part1Crop, self.part2Crop, self.part3Crop, self.part4Crop, self.part5Crop
                        ]):
                            # Resize and store
                            self.SetInspectionImages[i] = cv2.resize(crop, (512, 512))
                            # Run detection model
                            result = self.AGCJ59JRH_setDetectionModel(
                                self.SetInspectionImages[i],
                                stream=True,
                                verbose=False
                            )
                            # Extract the argmax result
                            self.InspectionResult_DetectionID[i] = list(result)[0].probs.data.argmax().item()
                        
                        self.InspectionResult_DetectionID = np.flip(self.InspectionResult_DetectionID)

                        self.InspectionResult_DetectionID = [int(x) for x in self.InspectionResult_DetectionID]
                        # self.InspectionResult_DetectionID = [1 - x for x in self.InspectionResult_DetectionID]
                        self.InspectionResult_SetID_OK = self.InspectionResult_DetectionID.copy()
                        self.InspectionResult_SetID_OK = [1 - x for x in self.InspectionResult_DetectionID]
                        self.InspectionResult_SetID_NG = [1 - x for x in self.InspectionResult_SetID_OK]

                        for i, d in enumerate(self.InspectionResult_DetectionID):
                            if d == 1:
                                self.InspectionResult_SetID_OK[i] = 0
                                self.InspectionResult_SetID_NG[i] = 0

                        self.InspectionResult_DetectionID_int = list_to_16bit_int(self.InspectionResult_DetectionID)
                        self.InspectionResult_SetID_OK_int = list_to_16bit_int(self.InspectionResult_SetID_OK)
                        self.InspectionResult_SetID_NG_int = list_to_16bit_int(self.InspectionResult_SetID_NG)

                        print(f"Inspection Result Detection ID: {self.InspectionResult_DetectionID}")
                        print(f"Inspection Result Set OK ID: {self.InspectionResult_SetID_OK}")
                        print(f"Inspection Result Set NG ID: {self.InspectionResult_SetID_NG}")

                        #Emit the inspection result and serial number to holding registers
                        self.requestModbusWrite.emit(self.holding_register_map["return_serialNumber_front"],[self.serialNumber_front])
                        self.requestModbusWrite.emit(self.holding_register_map["return_serialNumber_back"], [self.serialNumber_back])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_set_partexist"], [self.InspectionResult_DetectionID_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_set_results_OK"], [self.InspectionResult_SetID_OK_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_set_results_NG"], [self.InspectionResult_SetID_NG_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_state_code"],[1])
                        print("Inspection Result Set ID Emitted")
                        # Wait for 0.5 sec then emit return state code of 0 to show that it can accept the next instruction
                        time.sleep(0.5)
                        # self.requestModbusWrite.emit(self.holding_register_map["return_state_code"], [0])
                        # print("0 State Code Emitted, ready for next instruction")

                        if not os.path.exists("./aikensa/training_images/set"):
                            os.makedirs("./aikensa/training_images/set")
                            #Use time for the file name
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        #save InspectionImages with cv2.write
                        self.InspectionImages[0] = self.part1Crop
                        self.InspectionImages[1] = self.part2Crop
                        self.InspectionImages[2] = self.part3Crop
                        self.InspectionImages[3] = self.part4Crop
                        self.InspectionImages[4] = self.part5Crop

                        for i, img in enumerate(self.InspectionImages):
                            if img is not None:
                                filename = f"./aikensa/training_images/set/part{i+1}_{timestamp}.jpg"
                                cv2.imwrite(filename, img)
                                print(f"Saved {filename}")

                # 2 = Tape Inspection
                if self.InstructionCode == 2:
                    if self.InstructionCode_prev == 2:
                        pass
                    else:
                        self.InstructionCode_prev = self.InstructionCode
                        self.InspectionResult_DetectionID = [0, 0, 0, 0, 0]
                        # self.InspectionResult_TapeID_OK = random_list(5) #Dummy values for testing
                        self.InspectionResult_TapeID_OK = [1, 1, 1, 1, 1]
                        self.InspectionResult_TapeID_NG = [1 - x for x in self.InspectionResult_TapeID_OK]

                        self.InspectionResult_DetectionID_int = list_to_16bit_int(self.InspectionResult_DetectionID)
                        self.InspectionResult_TapeID_OK_int = list_to_16bit_int(self.InspectionResult_TapeID_OK)
                        self.InspectionResult_TapeID_NG_int = list_to_16bit_int(self.InspectionResult_TapeID_NG)

                        print(f"Inspection Result Detection ID: {self.InspectionResult_DetectionID}")
                        print(f"Inspection Result Tape OK ID: {self.InspectionResult_TapeID_OK}")
                        print(f"Inspection Result Tape NG ID: {self.InspectionResult_TapeID_NG}")

                        #Emit the inspection result and serial number to holding registers
                        self.requestModbusWrite.emit(self.holding_register_map["return_serialNumber_front"], [self.serialNumber_front])
                        self.requestModbusWrite.emit(self.holding_register_map["return_serialNumber_back"], [self.serialNumber_back])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_tapeinspection_partexist"], [self.InspectionResult_DetectionID_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_tapeinspection_results_OK"], [self.InspectionResult_TapeID_OK_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_tapeinspection_results_NG"], [self.InspectionResult_TapeID_NG_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_state_code"], [2])
                        print("Inspection Result Tape ID Emitted")
                        # Wait for 0.5 sec then emit return state code of 0 to show that it can accept the next instruction
                        time.sleep(0.5)
                        # self.requestModbusWrite.emit(self.holding_register_map["return_state_code"], [0])
                        # print("0 State Code Emitted, ready for next instruction")

                        if not os.path.exists("./aikensa/training_images/tape"):
                            os.makedirs("./aikensa/training_images/tape")
                            #Use time for the file name
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        #save InspectionImages with cv2.write
                        self.InspectionImages[0] = self.part1Crop
                        self.InspectionImages[1] = self.part2Crop
                        self.InspectionImages[2] = self.part3Crop
                        self.InspectionImages[3] = self.part4Crop
                        self.InspectionImages[4] = self.part5Crop

                        for i, img in enumerate(self.InspectionImages):
                            if img is not None:
                                filename = f"./aikensa/training_images/tape/part{i+1}_{timestamp}.jpg"
                                cv2.imwrite(filename, img)
                                print(f"Saved {filename}")
                        
                #emit the ethernet 
                self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
                self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)
            
                self.AGC_InspectionStatus.emit(self.InspectionStatus)


            if self.InstructionCode == 5:
                # Close app and forcefully turn off PC
                print("Instruction Code 5 received, closing app and turning off PC.")
                os.system("shutdown /s /t 1")
                self.running = False
                break


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
                status = "MANUAL",
                NGreason = "MANUAL",
                PPMS = "MANUAL")
        return [ok_count_current, ng_count_current], [ok_count_total, ng_count_total]
    
    def save_result_database(self, partname, numofPart, 
                             currentnumofPart, deltaTime, 
                             kensainName, 
                             status, NGreason, PPMS):

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
        status = str(status)
        NGreason = str(NGreason)
        PPMS = str(PPMS)

        self.cursor.execute('''
        INSERT INTO inspection_results (partname, numofPart, currentnumofPart, timestampHour, timestampDate, deltaTime, kensainName, status, NGreason, PPMS)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (partname, numofPart, currentnumofPart, timestamp_hour, timestamp_date, deltaTime, kensainName, status, NGreason, PPMS))
        self.conn.commit()

        # Update the totatl part number (Maybe the day has been changed)
        for key, value in self.widget_dir_map.items():
            self.inspection_config.today_numofPart[key] = self.get_last_entry_total_numofPart(value)

        # try:
        #     self.mysql_cursor.execute('''
        #     INSERT INTO inspection_results (partName, numofPart, currentnumofPart, timestampHour, timestampDate, deltaTime, kensainName, detected_pitch, delta_pitch, total_length, resultpitch, status, NGreason)
        #     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        #     ''', (partname, numofPart, currentnumofPart, timestamp_hour, timestamp_date, deltaTime, kensainName, detected_pitch_str, delta_pitch_str, total_length, resultPitch, status, NGreason))
        #     self.mysql_conn.commit()
        # except Exception as e:
        #     print(f"Error saving to MySQL: {e}")

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
        
    def _resolve_yaml_path(self, yaml_path: str) -> str:
        abs_from_cwd = os.path.abspath(yaml_path)
        if os.path.exists(abs_from_cwd):
            return abs_from_cwd
        here = os.path.dirname(os.path.abspath(__file__))
        abs_from_file = os.path.abspath(os.path.join(here, os.path.normpath(yaml_path)))
        if os.path.exists(abs_from_file):
            return abs_from_file
        raise FileNotFoundError(
            f"YAML not found.\n  Tried:\n    {abs_from_cwd}\n    {abs_from_file}"
        )
    
    def load_crop_coords(self, yaml_path: str):
        abs_path = self._resolve_yaml_path(yaml_path)
        print(f"[CropCoords] Loading YAML: {abs_path}")
        with open(abs_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise TypeError(f"YAML must be a dict, got {type(data).__name__}")
        for k, v in data.items():
            setattr(self, k, v)
        print(f"[CropCoords] Loaded keys: {list(data.keys())[:5]} ...")

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
        AGCJ30LH_partDetectionModel = None
        AGCJ30LH_setDetectionModel = None
        AGCJ30RH_partDetectionModel = None
        AGCJ30RH_setDetectionModel = None

        AGCJ59JLH_partDetectionModel = None
        AGCJ59JLH_setDetectionModel = None
        AGCJ59JRH_partDetectionModel = None
        AGCJ59JRH_setDetectionModel = None

        path_AGCJ30LH_partDetectionModel = "./aikensa/models/AGCJ30LH_PART.pt"
        path_AGCJ30LH_setDetectionModel = "./aikensa/models/AGCJ30LH_SET.pt"
        path_AGCJ30RH_partDetectionModel = "./aikensa/models/AGCJ30RH_PART.pt"
        path_AGCJ30RH_setDetectionModel = "./aikensa/models/AGCJ30RH_SET.pt"

        path_AGCJ59JLH_partDetectionModel = "./aikensa/models/AGCJ59JLH_PART.pt"
        path_AGCJ59JLH_setDetectionModel = "./aikensa/models/AGCJ59JLH_SET.pt"
        path_AGCJ59JRH_partDetectionModel = "./aikensa/models/AGCJ59JRH_PART.pt"
        path_AGCJ59JRH_setDetectionModel = "./aikensa/models/AGCJ59JRH_SET.pt"

        # Initialize models as None if file does not exist, otherwise load the model
        if os.path.exists(path_AGCJ30LH_partDetectionModel):
            AGCJ30LH_partDetectionModel = YOLO(path_AGCJ30LH_partDetectionModel)
        else:
            print(f"Model file {path_AGCJ30LH_partDetectionModel} does not exist. Initializing as None.")
            AGCJ30LH_partDetectionModel = None

        if os.path.exists(path_AGCJ30LH_setDetectionModel):
            AGCJ30LH_setDetectionModel = YOLO(path_AGCJ30LH_setDetectionModel)
        else:
            print(f"Model file {path_AGCJ30LH_setDetectionModel} does not exist. Initializing as None.")
            AGCJ30LH_setDetectionModel = None

        if os.path.exists(path_AGCJ30RH_partDetectionModel):
            AGCJ30RH_partDetectionModel = YOLO(path_AGCJ30RH_partDetectionModel)
        else:
            print(f"Model file {path_AGCJ30RH_partDetectionModel} does not exist. Initializing as None.")
            AGCJ30RH_partDetectionModel = None

        if os.path.exists(path_AGCJ30RH_setDetectionModel):
            AGCJ30RH_setDetectionModel = YOLO(path_AGCJ30RH_setDetectionModel)
        else:
            print(f"Model file {path_AGCJ30RH_setDetectionModel} does not exist. Initializing as None.")
            AGCJ30RH_setDetectionModel = None

        if os.path.exists(path_AGCJ59JLH_partDetectionModel):
            AGCJ59JLH_partDetectionModel = YOLO(path_AGCJ59JLH_partDetectionModel)
        else:
            print(f"Model file {path_AGCJ59JLH_partDetectionModel} does not exist. Initializing as None.")
            AGCJ59JLH_partDetectionModel = None

        if os.path.exists(path_AGCJ59JLH_setDetectionModel):
            AGCJ59JLH_setDetectionModel = YOLO(path_AGCJ59JLH_setDetectionModel)
        else:
            print(f"Model file {path_AGCJ59JLH_setDetectionModel} does not exist. Initializing as None.")
            AGCJ59JLH_setDetectionModel = None

        if os.path.exists(path_AGCJ59JRH_partDetectionModel):
            AGCJ59JRH_partDetectionModel = YOLO(path_AGCJ59JRH_partDetectionModel)
        else:
            print(f"Model file {path_AGCJ59JRH_partDetectionModel} does not exist. Initializing as None.")
            AGCJ59JRH_partDetectionModel = None

        if os.path.exists(path_AGCJ59JRH_setDetectionModel):
            AGCJ59JRH_setDetectionModel = YOLO(path_AGCJ59JRH_setDetectionModel)
        else:
            print(f"Model file {path_AGCJ59JRH_setDetectionModel} does not exist. Initializing as None.")
            AGCJ59JRH_setDetectionModel = None

        self.AGCJ30LH_partDetectionModel = AGCJ30LH_partDetectionModel
        self.AGCJ30LH_setDetectionModel = AGCJ30LH_setDetectionModel
        self.AGCJ30RH_partDetectionModel = AGCJ30RH_partDetectionModel
        self.AGCJ30RH_setDetectionModel = AGCJ30RH_setDetectionModel

        self.AGCJ59JLH_partDetectionModel = AGCJ59JLH_partDetectionModel
        self.AGCJ59JLH_setDetectionModel = AGCJ59JLH_setDetectionModel
        self.AGCJ59JRH_partDetectionModel = AGCJ59JRH_partDetectionModel
        self.AGCJ59JRH_setDetectionModel = AGCJ59JRH_setDetectionModel

    def stop(self):
        self.running = False
        print("Releasing all cameras.")
        self.release_camera()
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

    def process_and_emit_parts(self, width: int, height: int):
        """
        Downsample and emit all part crops dynamically.

        Args:
            width (int): Target width for downsampling.
            height (int): Target height for downsampling.
        """
        part_crops = [
            ("part1Crop", self.part1Cam),
            ("part2Crop", self.part2Cam),
            ("part3Crop", self.part3Cam),
            ("part4Crop", self.part4Cam),
            ("part5Crop", self.part5Cam),
        ]

        for attr_name, signal in part_crops:
            crop = getattr(self, attr_name, None)
            if crop is not None:
                downsampled = self.downSampling(crop, width=width, height=height)
                qimage = self.convertQImage(downsampled)
                signal.emit(qimage)
                setattr(self, attr_name, downsampled)

    def handle_part_number_update(self):
        """
        Keep last valid part number (1..4). Ignore transient 0.
        Non-zero invalid codes fall back to widget=0 as before.
        """
        code = self.partNumber
        if code is None:
            return

        # Ignore transient zeros: use last valid if available
        if code == 0:
            code = self.last_valid_part_number
            if code is None:
                return  # nothing to do yet if we don't have a valid one

        if code in (1, 2, 3, 4):
            if code != self.last_valid_part_number:
                self.last_valid_part_number = code
                new_widget = 5 + (code - 1)  # maps 1..4 -> 5..8
                if new_widget != self.inspection_config.widget:
                    self.inspection_config.widget = new_widget
                    print(f"Switching to widget {new_widget} based on part number {code}")
                    self.firstTimeInspection = True
                    self.inspection_config.doInspection = False
            return

        # Any other non-zero, non-1..4 code = unrecognized => reset widget
        # print(f"Part number {self.partNumber} not recognized for inspection.")
        self.inspection_config.widget = 0
        self.firstTimeInspection = True
        self.inspection_config.doInspection = False

    def handle_adjustments_and_counterreset(self):
        """
        Apply manual +/- adjustments if any flag is set, then handle counter reset.
        Keeps logic scoped to the currently selected widget.
        """
        cfg = self.inspection_config
        w = cfg.widget

        # Any manual adjustment flags on?
        any_adjust = (
            cfg.furyou_plus or cfg.furyou_minus or
            cfg.kansei_plus or cfg.kansei_minus or
            cfg.furyou_plus_10 or cfg.furyou_minus_10 or
            cfg.kansei_plus_10 or cfg.kansei_minus_10
        )

        if any_adjust:
            cur = cfg.current_numofPart[w]
            today = cfg.today_numofPart[w]
            new_cur, new_today = self.manual_adjustment(
                cur, today,
                cfg.furyou_plus,
                cfg.furyou_minus,
                cfg.furyou_plus_10,
                cfg.furyou_minus_10,
                cfg.kansei_plus,
                cfg.kansei_minus,
                cfg.kansei_plus_10,
                cfg.kansei_minus_10,
            )
            cfg.current_numofPart[w], cfg.today_numofPart[w] = new_cur, new_today
            print("Manual Adjustment Done")

        if cfg.counterReset is True:
            cfg.current_numofPart[w] = [0, 0]
            cfg.counterReset = False
            self.save_result_database(
                partname=self.widget_dir_map[w],
                numofPart=cfg.today_numofPart[w],
                currentnumofPart=[0, 0],
                deltaTime=0.0,
                kensainName=cfg.kensainNumber,
                status="COUNTERRESET",
                NGreason="COUNTERRESET",
                PPMS="COUNTERRESET",
            )

    def crop_part(self, image, attr_name: str, out_w: int = 512, out_h: int = 512):
        """
        Crop a quadrilateral region from the input image using self.<attr_name>
        and map it into a rectangle of size (out_w x out_h).

        The attribute must be a list of 4 points:
            [TopLeft, BottomLeft, BottomRight, TopRight]

        Example:
            cropped = self.crop_part(image, "J30RH_part1_Crop", 512, 512)
        """
        # Retrieve the coordinate list from the instance
        if not hasattr(self, attr_name):
            raise AttributeError(f"Attribute '{attr_name}' not found in self.")
        quad = getattr(self, attr_name)
        if quad is None:
            raise ValueError(f"'{attr_name}' is None (not loaded yet).")
        if len(quad) != 4:
            raise ValueError(f"'{attr_name}' must contain 4 corner points, got {len(quad)}.")

        # Convert to NumPy array and reorder for OpenCV
        tl, bl, br, tr = quad  # Your order
        src = np.array([tl, tr, br, bl], dtype=np.float32)

        # Destination rectangle (top-left → bottom-right)
        dst = np.array([
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1]
        ], dtype=np.float32)

        # Compute the perspective transform matrix and warp
        M = cv2.getPerspectiveTransform(src, dst)
        cropped = cv2.warpPerspective(
            image, M, (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        return cropped
    
        """
        Crop a quadrilateral region from the input image using self.<attr_name>
        and map it into a rectangle of size (out_w x out_h).

        The attribute must be a list of 4 points:
            [TopLeft, BottomLeft, BottomRight, TopRight]

        Example:
            cropped = self.crop_part(image, "J30RH_part1_Crop", 512, 512)
        """
        # Retrieve the coordinate list from the instance
        if not hasattr(self, attr_name):
            raise AttributeError(f"Attribute '{attr_name}' not found in self.")
        quad = getattr(self, attr_name)
        if quad is None:
            raise ValueError(f"'{attr_name}' is None (not loaded yet).")
        if len(quad) != 4:
            raise ValueError(f"'{attr_name}' must contain 4 corner points, got {len(quad)}.")

        # Convert to NumPy array and reorder for OpenCV
        tl, bl, br, tr = quad  # Your order
        src = np.array([tl, tr, br, bl], dtype=np.float32)

        # Destination rectangle (top-left → bottom-right)
        dst = np.array([
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1]
        ], dtype=np.float32)

        # Compute the perspective transform matrix and warp
        M = cv2.getPerspectiveTransform(src, dst)
        cropped = cv2.warpPerspective(
            image, M, (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        return cropped