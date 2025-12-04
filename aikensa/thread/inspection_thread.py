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


from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

from aikensa.camscripts.cam_init import initialize_camera
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard , calculatecameramatrix, warpTwoImages, calculateHomography_template, warpTwoImages_template
from aikensa.opencv_imgprocessing.arucoplanarize import planarize, planarize_image
from dataclasses import dataclass, field
from typing import List, Tuple

from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
from aikensa.parts_config.sound import play_keisoku_sound

from aikensa.scripts.scripts import list_to_16bit_int, load_register_map, invert_16bit_int, random_list, combine_by_and
import time

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
    part6Cam = pyqtSignal(QImage)
    part7Cam = pyqtSignal(QImage)
    part8Cam = pyqtSignal(QImage)
    part9Cam = pyqtSignal(QImage)
    part10Cam = pyqtSignal(QImage)


    P668307UA0A_InspectionStatus = pyqtSignal(list)

    holding_register_signal = pyqtSignal(list)

    today_numofPart_signal = pyqtSignal(list)
    current_numofPart_signal = pyqtSignal(list)

    partNumber_signal = pyqtSignal(int)

    requestModbusWrite = pyqtSignal(int, list)

    def __init__(self, inspection_config: InspectionConfig = None, modbus_thread=None):
        super(InspectionThread, self).__init__()
        self.running = True
        self.latest_frame = None
        
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

        self.part1Crop = None
        self.part2Crop = None
        self.part3Crop = None
        self.part4Crop = None
        self.part5Crop = None
        self.part6Crop = None
        self.part7Crop = None
        self.part8Crop = None
        self.part9Crop = None
        self.part10Crop = None

        self.frame_width = 2048
        self.frame_height = 2048


        self.planarizeTransform = None
        self.planarizeTransform_scaled = None

        self.planarizeTransform_temp = None
        self.planarizeTransform_temp_scaled = None
        
        self.part1Crop_Pos = (335, 505)
        self.part2Crop_Pos = (336, 730)
        self.part3Crop_Pos = (340, 960)
        self.part4Crop_Pos = (345, 1185)
        self.part5Crop_Pos = (350, 1410)
        self.part6Crop_Pos = (1610, 490)
        self.part7Crop_Pos = (1610, 717)
        self.part8Crop_Pos = (1616, 943)
        self.part9Crop_Pos = (1615, 1170)
        self.part10Crop_Pos = (1625, 1395)

        self.partCrop_width_height = (150, 150)

        self.qtWindowWidth = 110
        self.qtWindowHeight = 110

        self.timerStart = None
        self.timerFinish = None
        self.fps = None

        self.timerStart_mini = None
        self.timerFinish_mini = None
        self.fps_mini = None

        self.InspectionImages = [None]*10

        self.InspectionResult_DetectionID = [None]*10
        self.InspectionResult_DetectionID_int = None

        self.InspectionResult_SetID_OK = [None]*10
        self.InspectionResult_SetID_OK_int = None

        self.InspectionResult_SetID_NG = [None]*10
        self.InspectionResult_SetID_NG_int = [None]*10

        self.InspectionResult_BouseiID_OK = [None]*10
        self.InspectionResult_BouseiID_OK_int = [None]*10

        self.InspectionResult_BouseiID_NG = [None]*10
        self.InspectionResult_BouseiID_NG_int = [None]*10

        self.InspectionResult_NGReason = [None]*10
        
        self.InspectionStatus = [None]*5

        self.widget_dir_map = {
            5: "P668307UA0A_COWL_TOP",
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
        self.TurnOffCommand = False

        self.prev_numofPart = None

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
        self.TurnOffCommand   = reg_dict.get(101, 0)

        # print(f"Part Number: {self.partNumber}")
        # print(f"Serial Number Front: {self.serialNumber_front}")
        # print(f"Serial Number Back:  {self.serialNumber_back}")
        print(f"Instruction Code:     {self.InstructionCode}")

    @pyqtSlot(np.ndarray)
    def receive_frame(self, frame):
        self.latest_frame = frame

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
        # print("Inspection Thread Started")
        self.initialize_model()
        # print("AI Model Initialized")

        # for key, value in self.widget_dir_map.items():
        #     self.inspection_config.current_numofPart[key] = self.get_last_entry_currentnumofPart(value)
        #     self.inspection_config.today_numofPart[key] = self.get_last_entry_total_numofPart(value)

        last_seen_id = 0

        while self.running:

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


                if self.latest_frame is not None:
                    cur_id = id(self.latest_frame)
                    if cur_id != last_seen_id:
                        last_seen_id = cur_id
                        self.camFrame = self.latest_frame

                else:
                    print("Camera 0 is not opened, creating placeholder image.")
                    self.camFrame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                    self.camFrame[:] = (0, 255, 0)

                self.part1Crop = self.camFrame[self.part1Crop_Pos[1]:self.part1Crop_Pos[1]+self.partCrop_width_height[1], 
                                                self.part1Crop_Pos[0]:self.part1Crop_Pos[0]+self.partCrop_width_height[0]]
                self.part2Crop = self.camFrame[self.part2Crop_Pos[1]:self.part2Crop_Pos[1]+self.partCrop_width_height[1],
                                                self.part2Crop_Pos[0]:self.part2Crop_Pos[0]+self.partCrop_width_height[0]]
                self.part3Crop = self.camFrame[self.part3Crop_Pos[1]:self.part3Crop_Pos[1]+self.partCrop_width_height[1],
                                                self.part3Crop_Pos[0]:self.part3Crop_Pos[0]+self.partCrop_width_height[0]]
                self.part4Crop = self.camFrame[self.part4Crop_Pos[1]:self.part4Crop_Pos[1]+self.partCrop_width_height[1],
                                                self.part4Crop_Pos[0]:self.part4Crop_Pos[0]+self.partCrop_width_height[0]]
                self.part5Crop = self.camFrame[self.part5Crop_Pos[1]:self.part5Crop_Pos[1]+self.partCrop_width_height[1],
                                                self.part5Crop_Pos[0]:self.part5Crop_Pos[0]+self.partCrop_width_height[0]]
                self.part6Crop = self.camFrame[self.part6Crop_Pos[1]:self.part6Crop_Pos[1]+self.partCrop_width_height[1],
                                                self.part6Crop_Pos[0]:self.part6Crop_Pos[0]+self.partCrop_width_height[0]]
                self.part7Crop = self.camFrame[self.part7Crop_Pos[1]:self.part7Crop_Pos[1]+self.partCrop_width_height[1],
                                                self.part7Crop_Pos[0]:self.part7Crop_Pos[0]+self.partCrop_width_height[0]]
                self.part8Crop = self.camFrame[self.part8Crop_Pos[1]:self.part8Crop_Pos[1]+self.partCrop_width_height[1],
                                                self.part8Crop_Pos[0]:self.part8Crop_Pos[0]+self.partCrop_width_height[0]]
                self.part9Crop = self.camFrame[self.part9Crop_Pos[1]:self.part9Crop_Pos[1]+self.partCrop_width_height[1],
                                                self.part9Crop_Pos[0]:self.part9Crop_Pos[0]+self.partCrop_width_height[0]]
                self.part10Crop = self.camFrame[self.part10Crop_Pos[1]:self.part10Crop_Pos[1]+self.partCrop_width_height[1],
                                                self.part10Crop_Pos[0]:self.part10Crop_Pos[0]+self.partCrop_width_height[0]]
         

                #print the part number
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
                if self.part6Crop is not None:
                    self.part6Crop = self.downSampling(self.part6Crop, width = self.qtWindowWidth, height = self.qtWindowHeight)
                    self.part6Cam.emit(self.convertQImage(self.part6Crop))
                if self.part7Crop is not None:
                    self.part7Crop = self.downSampling(self.part7Crop, width = self.qtWindowWidth, height = self.qtWindowHeight)
                    self.part7Cam.emit(self.convertQImage(self.part7Crop))
                if self.part8Crop is not None:
                    self.part8Crop = self.downSampling(self.part8Crop, width = self.qtWindowWidth, height = self.qtWindowHeight)
                    self.part8Cam.emit(self.convertQImage(self.part8Crop))
                if self.part9Crop is not None:
                    self.part9Crop = self.downSampling(self.part9Crop, width = self.qtWindowWidth, height = self.qtWindowHeight)
                    self.part9Cam.emit(self.convertQImage(self.part9Crop))
                if self.part10Crop is not None:
                    self.part10Crop = self.downSampling(self.part10Crop, width = self.qtWindowWidth, height = self.qtWindowHeight)
                    self.part10Cam.emit(self.convertQImage(self.part10Crop))

                    if self.firstTimeInspection is False:
                        if self.inspection_config.doInspection is False:
                            self.InspectionTimeStart = time.time()
                            self.firstTimeInspection = True
                            self.inspection_config.doInspection = True
                
                self.partNumber_signal.emit(self.partNumber)


                if self.inspection_config.doInspection is True:
                    self.inspection_config.doInspection = False
                    self.InstructionCode = 2


                if self.InstructionCode != 0:
                    self.InspectionImages[0] = self.part1Crop
                    self.InspectionImages[1] = self.part2Crop
                    self.InspectionImages[2] = self.part3Crop
                    self.InspectionImages[3] = self.part4Crop
                    self.InspectionImages[4] = self.part5Crop
                    self.InspectionImages[5] = self.part6Crop
                    self.InspectionImages[6] = self.part7Crop
                    self.InspectionImages[7] = self.part8Crop
                    self.InspectionImages[8] = self.part9Crop
                    self.InspectionImages[9] = self.part10Crop
                    # Do Inference


                if self.InstructionCode == 0:
                    if self.InstructionCode_prev == 0:
                        # print("State Code 0 Already Processed, Skipping")
                        pass
                    else:
                        self.InstructionCode_prev = self.InstructionCode

                        self.InspectionResult_DetectionID = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

                        self.InspectionResult_BouseiID_OK = [0, 0, 0, 0, 0]
                        self.InspectionResult_BouseiID_NG = [0, 0, 0, 0, 0]

                        self.InspectionResult_DetectionID_int = list_to_16bit_int(self.InspectionResult_DetectionID)
                        self.InspectionResult_BouseiID_OK_int = list_to_16bit_int(self.InspectionResult_BouseiID_OK)
                        self.InspectionResult_BouseiID_NG_int = list_to_16bit_int(self.InspectionResult_BouseiID_NG)

                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_bouseiinspection_partexist"], [self.InspectionResult_DetectionID_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_bouseiinspection_results_OK"], [self.InspectionResult_BouseiID_OK_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_bouseiinspection_results_NG"], [self.InspectionResult_BouseiID_NG_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_state_code"], [0])
                        
                        # print(f"Inspection Result Bousei OK ID: {self.InspectionResult_BouseiID_OK}")
                        # print(f"Inspection Result Bousei NG ID: {self.InspectionResult_BouseiID_NG}")

                        self.InspectionStatus = ["待機"] * 5

                if self.InstructionCode == 2:
                    if self.InstructionCode_prev == 2:
                        print("State Code 2 Already Processed, Skipping")
                    else:
                        self.InstructionCode_prev = self.InstructionCode

                        print("State Code 2 Received, Starting Inspection")

                        for i in range(len(self.InspectionImages)):
                            image = self.InspectionImages[i]
                            if i < 5:
                                image = cv2.rotate(image, cv2.ROTATE_180)
                            print(f"Part {i+1} Image: {image is not None}")

                            #AI INFER
                            if image is not None:
                                # Do YOLO inference to check whether part exists
                                _ = self.P668307UA0A_kensaModel(image, stream=True, verbose=False, imgsz = 128)
                                self.InspectionResult_DetectionID[i] = list(_)[0].probs.data.argmax().item()
                                print (f"Part {i+1} Detection ID: {self.InspectionResult_DetectionID[i]}")
                                #save image too
                                self.save_image(image, self.InspectionResult_DetectionID[i])

                                
                                #Detection ID is as follows:
                                # {0: 'NOPART', 1: 'NURIWASURE', 2: 'OK'}
                                # If Detection ID is 2, then it is OK, otherwise it is NG

                                #remap the detection ID like this: 
                                # {0: 'NOPART' into 1, 1: 'NURIWASURE' into 1, 2: 'OK' into 0} -> so Basically if its 2 its 0, others will be 1
                                self.InspectionResult_DetectionID[i] = 0 if self.InspectionResult_DetectionID[i] != 2 else 1
                                print(f"Part {i+1} Remapped Detection ID: {self.InspectionResult_DetectionID[i]}")

                        #HARD CODE THE OK HERE
                        self.InspectionResult_DetectionID = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

                        self.InspectionResult_BouseiID_OK = combine_by_and(self.InspectionResult_DetectionID)
                        self.InspectionResult_BouseiID_NG = [1 - x for x in self.InspectionResult_BouseiID_OK]

                        prev = self.inspection_config.current_numofPart[self.inspection_config.widget][0]   

                        for i in range (len(self.InspectionResult_BouseiID_OK)):
                            if self.InspectionResult_BouseiID_OK[i] == 1:
                                self.InspectionStatus[i] = "OK"
                                self.inspection_config.current_numofPart[self.inspection_config.widget][0] += 1
                                self.inspection_config.today_numofPart[self.inspection_config.widget][0] += 1
                            elif self.InspectionResult_BouseiID_NG[i] == 1:
                                self.InspectionStatus[i] = "NG"
                                self.inspection_config.current_numofPart[self.inspection_config.widget][1] += 1
                                self.inspection_config.today_numofPart[self.inspection_config.widget][1] += 1
                            
                        #PLAY SOUND IF THE TOTAL OF OK FOR THE CURRENT NUM OF PART IS IN MULTIPLICATION OF 40
                        # if self.inspection_config.current_numofPart[self.inspection_config.widget][0] % 40 == 0 and self.inspection_config.current_numofPart[self.inspection_config.widget][0] != 0:
                        #     play_keisoku_sound()

                        curr = self.inspection_config.current_numofPart[self.inspection_config.widget][0]


                        if curr > prev:
                            # Block index: 0 = 0?39, 1 = 40?79, 2 = 80?119, ...
                            prev_block = prev // 40
                            curr_block = curr // 40

                            # If we entered a new block (40, 80, 120, ...) then play sound
                            if curr_block > prev_block and curr_block > 0:
                                play_keisoku_sound()

                        self.InspectionResult_DetectionID_int = list_to_16bit_int(self.InspectionResult_DetectionID)
                        self.InspectionResult_BouseiID_OK_int = list_to_16bit_int(self.InspectionResult_BouseiID_OK)
                        self.InspectionResult_BouseiID_NG_int = list_to_16bit_int(self.InspectionResult_BouseiID_NG)

                        # print(f"Inspection Result Detection ID: {self.InspectionResult_DetectionID}")
                        # print(f"Inspection Result Bousei OK ID: {self.InspectionResult_BouseiID_OK}")
                        # print(f"Inspection Result Bousei NG ID: {self.InspectionResult_BouseiID_NG}")

                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_bouseiinspection_partexist"], [self.InspectionResult_DetectionID_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_bouseiinspection_results_OK"], [self.InspectionResult_BouseiID_OK_int])
                        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_bouseiinspection_results_NG"], [self.InspectionResult_BouseiID_NG_int])
                        time.sleep(0.1)
                        self.requestModbusWrite.emit(self.holding_register_map["return_state_code"], [2])
                        print("Inspection Result Tape ID Emitted")
                        # Wait for 0.5 sec then emit return state code of 0 to show that it can accept the next instruction
                        self.P668307UA0A_InspectionStatus.emit(self.InspectionStatus)
                        time.sleep(0.5)
                        self.requestModbusWrite.emit(self.holding_register_map["return_state_code"], [0])
                        print("0 State Code Emitted, ready for next instruction")

                self.today_numofPart_signal.emit(self.inspection_config.today_numofPart)
                self.current_numofPart_signal.emit(self.inspection_config.current_numofPart)
            
                self.P668307UA0A_InspectionStatus.emit(self.InspectionStatus)

                # self.msleep(5)


            if self.TurnOffCommand == 1:
                #turn pc off
                os.system("shutdown now")

        # self.msleep(5)


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
        status = str(status)
        NGreason = str(NGreason)

        self.cursor.execute('''
        INSERT INTO inspection_results (partname, numofPart, currentnumofPart, timestampHour, timestampDate, deltaTime, kensainName, status, NGreason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (partname, numofPart, currentnumofPart, timestamp_hour, timestamp_date, deltaTime, kensainName, status, NGreason))
        self.conn.commit()

        # Update the totatl part number (Maybe the day has been changed)
        for key, value in self.widget_dir_map.items():
            self.inspection_config.today_numofPart[key] = self.get_last_entry_total_numofPart(value)

        #Also save to mysql cursor
        # self.mysql_cursor.execute('''
        # INSERT INTO inspection_results (partName, numofPart, currentnumofPart, timestampHour, timestampDate, deltaTime, kensainName, status, NGreason)
        # VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        # ''', (partname, numofPart, currentnumofPart, timestamp_hour, timestamp_date, deltaTime, kensainName, status, NGreason))
        # self.mysql_conn.commit()

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

    def save_image(self, image, detection_id):
        #Detection ID is as follows:
        # {0: 'NOPART', 1: 'NURIWASURE', 2: 'OK'}
        #Save the image in the directory based on the detection ID
        if detection_id == 0:
            dir = f"aikensa/inspection/{self.widget_dir_map[self.inspection_config.widget]}/nopart/"
        elif detection_id == 1:
            dir = f"aikensa/inspection/{self.widget_dir_map[self.inspection_config.widget]}/nuriwasure/"
        elif detection_id == 2:
            dir = f"aikensa/inspection/{self.widget_dir_map[self.inspection_config.widget]}/ok/"
        else:
            print(f"Unknown detection ID: {detection_id}, saving in default directory.")

        os.makedirs(dir, exist_ok=True)
        # Create a unique filename based on the current timestamp

        base_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_filename}.png"
        filepath = os.path.join(dir, filename)
        counter = 1
        # If file exists, add a number suffix
        while os.path.exists(filepath):
            filename = f"{base_filename}_{counter}.png"
            filepath = os.path.join(dir, filename)
            counter += 1
        cv2.imwrite(filepath, image)

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

    def convertQImage(self, frame: np.ndarray) -> QImage:
        """Safely convert a NumPy image (from BGRA or BGR) to QImage."""
        # If BGRA (4 channels)
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            height, width, _ = frame.shape
            return QImage(frame.data, width, height, width * 3, QImage.Format_RGB888).copy()

        # If BGR (3 channels)
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, _ = frame.shape
            return QImage(frame.data, width, height, width * 3, QImage.Format_RGB888).copy()

        else:
            raise ValueError("Unsupported image format")

    
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

        P668307UA0A_partDetectionModel_path = "./aikensa/models/P668307UA0A_partDetectionModel.pt"
        P668307UA0A_setSegmentationModel_path = "./aikensa/models/P668307UA0A_setSegmentationModel.pt"
        P668307UA0A_bouseiSegmentationModel_path = "./aikensa/models/P668307UA0A_bouseiSegmentationModel.pt"
        P668307UA0A_kensaModel_path = "./aikensa/models/P668307UA0A_kensaModel.pt"

        # Initialize models as None if file does not exist, otherwise load the model
        if os.path.exists(P668307UA0A_partDetectionModel_path):
            P668307UA0A_partDetectionModel = YOLO(P668307UA0A_partDetectionModel_path)
        else:
            print(f"Model file {P668307UA0A_partDetectionModel_path} does not exist. Initializing as None.")
            P668307UA0A_partDetectionModel = None

        if os.path.exists(P668307UA0A_setSegmentationModel_path):
            P668307UA0A_setSegmentationModel = YOLO(P668307UA0A_setSegmentationModel_path)
        else:
            print(f"Model file {P668307UA0A_setSegmentationModel_path} does not exist. Initializing as None.")
            P668307UA0A_setSegmentationModel = None

        if os.path.exists(P668307UA0A_bouseiSegmentationModel_path):
            P668307UA0A_bouseiSegmentationModel = YOLO(P668307UA0A_bouseiSegmentationModel_path)
        else:
            print(f"Model file {P668307UA0A_bouseiSegmentationModel_path} does not exist. Initializing as None.")
            P668307UA0A_bouseiSegmentationModel = None

        if os.path.exists(P668307UA0A_kensaModel_path):
            P668307UA0A_kensaModel = YOLO(P668307UA0A_kensaModel_path)
        else:
            print(f"Model file {P668307UA0A_kensaModel_path} does not exist. Initializing as None.")
            P668307UA0A_kensaModel = None

        self.P668307UA0A_partDetectionModel = P668307UA0A_partDetectionModel
        self.P668307UA0A_setSegmentationModel = P668307UA0A_setSegmentationModel
        self.P668307UA0A_bouseiSegmentationModel = P668307UA0A_bouseiSegmentationModel
        self.P668307UA0A_kensaModel = P668307UA0A_kensaModel

    def stop(self):
        self.running = False
        print("Releasing all cameras.")
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
