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

        self.part1Crop_YPos = 175
        self.part2Crop_YPos = 480
        self.part3Crop_YPos = 800
        self.part4Crop_YPos = 1120
        self.part5Crop_YPos = 1440

        self.part1Crop_YPos_scaled = int(self.part1Crop_YPos//self.scale_factor)
        self.part2Crop_YPos_scaled = int(self.part2Crop_YPos//self.scale_factor)
        self.part3Crop_YPos_scaled = int(self.part3Crop_YPos//self.scale_factor)
        self.part4Crop_YPos_scaled = int(self.part4Crop_YPos//self.scale_factor)
        self.part5Crop_YPos_scaled = int(self.part5Crop_YPos//self.scale_factor)

        self.height_hole_offset = int(120//self.scale_factor_hole)
        self.width_hole_offset = int(370//self.scale_factor_hole)

        self.timerStart = None
        self.timerFinish = None
        self.fps = None

        self.timerStart_mini = None
        self.timerFinish_mini = None
        self.fps_mini = None

        self.InspectionImages = [None]*5

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
            5: "AGC_Line"
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


    @pyqtSlot(dict)
    def on_holding_update(self, reg_dict):
        # Only called whenever the Modbus thread emits new data.
        self.partNumber = reg_dict.get(50, 0)
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
                CREATE TABLE IF NOT EXISTS AGC_tapehari_inspection_results (
                    id INTEGER PRIMARY KEY AUTO_INCREMENT,
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
            self.mysql_conn.commit()


        #print thread started
        print("Inspection Thread Started")
        self.initialize_model()
        print("AI Model Initialized")

        self.current_cameraID = self.inspection_config.cameraID
        self.initialize_single_camera(self.current_cameraID)
        self.initialize_single_camera(0)

        # for key, value in self.widget_dir_map.items():
        #     self.inspection_config.current_numofPart[key] = self.get_last_entry_currentnumofPart(value)
        #     self.inspection_config.today_numofPart[key] = self.get_last_entry_total_numofPart(value)

        while self.running:


            if self.inspection_config.widget == 0:
                self.inspection_config.cameraID = -1

            if self.inspection_config.widget == 5:
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
                            status = "COUNTERRESET",
                            NGreason = "COUNTERRESET",
                            PPMS = "COUNTERRESET")  

                #initialize single camera with id 0

                if self.cap_cam is None:
                    print("Camera 0 is not initialized, skipping frame capture.")
                    continue
                # Read the frame from the camera
                _, self.camFrame = self.cap_cam.read()

                self.camFrame = cv2.rotate(self.camFrame, cv2.ROTATE_180)

                self.part1Crop = self.camFrame[int(self.part1Crop_YPos) : int((self.part1Crop_YPos + self.part_height_offset)), 0 : int(self.camFrame.shape[1])]
                self.part2Crop = self.camFrame[int(self.part2Crop_YPos) : int((self.part2Crop_YPos + self.part_height_offset)), 0 : int(self.camFrame.shape[1])]
                self.part3Crop = self.camFrame[int(self.part3Crop_YPos) : int((self.part3Crop_YPos + self.part_height_offset)), 0 : int(self.camFrame.shape[1])]
                self.part4Crop = self.camFrame[int(self.part4Crop_YPos) : int((self.part4Crop_YPos + self.part_height_offset)), 0 : int(self.camFrame.shape[1])]
                self.part5Crop = self.camFrame[int(self.part5Crop_YPos) : int((self.part5Crop_YPos + self.part_height_offset)), 0 : int(self.camFrame.shape[1])]

                if self.part1Crop is not None:
                    self.part1Crop = self.downSampling(self.part1Crop, width = self.qtWindowWidth, height = self.qtWindowHeight)
                    self.part1Cam.emit(self.convertQImage(self.part1Crop))
                if self.part2Crop is not None:
                    self.part2Crop = self.downSampling(self.part2Crop, width = self.qtWindowWidth, height = self.qtWindowHeight)
                    self.part2Cam.emit(self.convertQImage(self.part2Crop))
                if self.part3Crop is not None:
                    self.part3Crop = self.downSampling(self.part3Crop, width = self.qtWindowWidth, height = self.qtWindowHeight)
                    self.part3Cam.emit(self.convertQImage(self.part3Crop))
                if self.part4Crop is not None:
                    self.part4Crop = self.downSampling(self.part4Crop, width = self.qtWindowWidth, height = self.qtWindowHeight)
                    self.part4Cam.emit(self.convertQImage(self.part4Crop))
                if self.part5Crop is not None:
                    self.part5Crop = self.downSampling(self.part5Crop, width = self.qtWindowWidth, height = self.qtWindowHeight)
                    self.part5Cam.emit(self.convertQImage(self.part5Crop))

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

                if self.InstructionCode == 1:
                    if self.InstructionCode_prev == 1:
                        pass
                    else:
                        self.InstructionCode_prev = self.InstructionCode

                        self.InspectionResult_DetectionID = [0, 0, 0, 0, 0]
                        self.InspectionResult_SetID_OK = random_list(5) #Dummy values for testing
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
                        self.requestModbusWrite.emit(self.holding_register_map["return_state_code"], [0])
                        print("0 State Code Emitted, ready for next instruction")

                if self.InstructionCode == 2:
                    if self.InstructionCode_prev == 2:
                        pass
                    else:
                        self.InstructionCode_prev = self.InstructionCode
                        self.InspectionResult_DetectionID = [0, 0, 0, 0, 0]
                        self.InspectionResult_TapeID_OK = random_list(5) #Dummy values for testing
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
                        self.requestModbusWrite.emit(self.holding_register_map["return_state_code"], [0])
                        print("0 State Code Emitted, ready for next instruction")
                        




                # if self.inspection_config.doInspection is True:
                #     self.inspection_config.doInspection = False

                #     if self.inspection_config.kensainNumber is None or self.inspection_config.kensainNumber == "":
                #         print("No Kensain Number Input")
                #         continue
                    
                #     if self.InspectionTimeStart is not None:
                #         if time.time() - self.InspectionTimeStart > self.InspectionWaitTime or self.firstTimeInspection is True:

                #             # # Do the inspection
                #             for i in range(len(self.InspectionImages)):
                #                 #Do YOLO inference to check whether part exist

                #                 print(f"Part {i+1} Inference Start")

                #                 #if part exists, do another inference to check whether the part is positioned correctly





                #                 # print(self.InspectionResult_Status[i])
                #                 # print(self.InspectionResult_DetectionID[i])

                #                 if self.InspectionResult_Status[i] == "OK":
                #                     self.inspection_config.current_numofPart[self.inspection_config.widget][0] += 1
                #                     self.inspection_config.today_numofPart[self.inspection_config.widget][0] += 1
                #                     self.InspectionStatus[i] = "OK"

                #                 elif self.InspectionResult_Status[i] == "NG":
                #                     self.inspection_config.current_numofPart[self.inspection_config.widget][1] += 1
                #                     self.inspection_config.today_numofPart[self.inspection_config.widget][1] += 1
                #                     self.InspectionStatus[i] = "NG"

                #                 self.save_result_database(partname = self.widget_dir_map[self.inspection_config.widget],
                #                     numofPart = self.inspection_config.today_numofPart[self.inspection_config.widget], 
                #                     currentnumofPart = self.inspection_config.current_numofPart[self.inspection_config.widget],
                #                     deltaTime = 0.0,
                #                     kensainName = self.inspection_config.kensainNumber, 
                #                     status = self.InspectionResult_Status[i], 
                #                     NGreason = self.InspectionResult_NGReason[i],
                #                     PPMS = "PPMS")

                #                 self.hoodFR_InspectionStatus.emit(self.InspectionStatus)

                #             self.save_image_result(self.part1Crop, self.InspectionImages[0], self.InspectionResult_Status[0], True, "P1")
                #             self.save_image_result(self.part2Crop, self.InspectionImages[1], self.InspectionResult_Status[1], True, "P2")
                #             self.save_image_result(self.part3Crop, self.InspectionImages[2], self.InspectionResult_Status[2], True, "P3")
                #             self.save_image_result(self.part4Crop, self.InspectionImages[3], self.InspectionResult_Status[3], True, "P4")
                #             self.save_image_result(self.part5Crop, self.InspectionImages[4], self.InspectionResult_Status[4], True, "P5")

                #             self.InspectionImages[0] = self.downSampling(self.InspectionImages[0], width=1701, height=121)
                #             self.InspectionImages[1] = self.downSampling(self.InspectionImages[1], width=1701, height=121)
                #             self.InspectionImages[2] = self.downSampling(self.InspectionImages[2], width=1701, height=121)
                #             self.InspectionImages[3] = self.downSampling(self.InspectionImages[3], width=1701, height=121)
                #             self.InspectionImages[4] = self.downSampling(self.InspectionImages[4], width=1701, height=121)

                #             print("Inspection Finished")
                #             #Remember that list is mutable

                #             self.part1Cam.emit(self.converQImageRGB(self.InspectionImages[0]))
                #             self.part2Cam.emit(self.converQImageRGB(self.InspectionImages[1]))
                #             self.part3Cam.emit(self.converQImageRGB(self.InspectionImages[2]))
                #             self.part4Cam.emit(self.converQImageRGB(self.InspectionImages[3]))
                #             self.part5Cam.emit(self.converQImageRGB(self.InspectionImages[4]))

                #             # self.hoodFR_InspectionStatus.emit(self.InspectionStatus)

                #emit the ethernet 
                self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
                self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)
            
                # Emit status based on the red tenmetsu status

                
                self.AGC_InspectionStatus.emit(self.InspectionStatus)

                # Emit the hole detection

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
