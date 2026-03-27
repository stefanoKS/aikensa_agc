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
import threading

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

from aikensa.camscripts.cam_init import initialize_camera
from aikensa.camscripts.cam_init_ic4 import initialize_camera_ic4
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard , calculatecameramatrix, warpTwoImages, calculateHomography_template, warpTwoImages_template
from aikensa.opencv_imgprocessing.arucoplanarize import planarize, planarize_image
from dataclasses import dataclass, field
from typing import List, Tuple

from aikensa.parts_config.sound import play_do_sound, play_picking_sound, play_re_sound, play_mi_sound, play_alarm_sound, play_konpou_sound, play_keisoku_sound

from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image

from aikensa.scripts.scripts import list_to_16bit_int, load_register_map, invert_16bit_int, random_list
from aikensa.scripts.scripts_img_processing import crop_parts, aruco_detect, aruco_detect_yolo

from aikensa.parts_config.AGC.JXX_KENSA import JXX_Check as JXX_Check

from typing import Iterable, Any, Optional, Dict, Tuple, List, Callable
import re


logger = logging.getLogger(__name__)

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
    doTapeInspection: bool = False

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

    nichijoutenken_mode: bool = False

    manual_part_selection: str = None
    debug_mode_selection: str = None

    debug_mode: bool = False

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

    SerialNumber_signal = pyqtSignal(str)
    LotNumber_signal = pyqtSignal(str)

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

        self.models_initialized = False
        self.camera_read_timeout_ms = 100


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

        self.Tray_detection_left_crop = None
        self.Tray_detection_right_crop = None

        self.Tray_detection_left_image = None
        self.Tray_detection_right_image = None

        self.Tray_detection_left_result = None
        self.Tray_detection_right_result = None

        self.height_hole_offset = int(120//self.scale_factor_hole)
        self.width_hole_offset = int(370//self.scale_factor_hole)

        self.timerStart = None
        self.timerFinish = None
        self.fps = None

        self.timerStart_mini = None
        self.timerFinish_mini = None
        self.fps_mini = None

        self.InspectionImages = [None]*5

        self.SetExistInspectionImages = [None]*5
        self.SetCorrectInspectionImages = [None]*5
        self.SetCorrectInspectionImages_result = [None]*5

        self.TapeExistInspectionImages = [None]*5
        self.TapeCorrectInspectionImages = [None]*5
        self.TapeCorrectInspectionImages_result = [None]*5

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

        self.InspectionResult_Tray_NG  = None #1 = NG, 0 = OK
        
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

        self.partNumber_modbus = None
        self.InstructionCode_modbus = None

        self.partNumber = None
        self.partNumber_prev = None
        self.serialNumber_front = None
        self.serialNumber_back = None
        self.IV4_KensaResults_OK = None
        self.IV4_KensaResults_NG = None
        self.lotNumber_front = None
        self.InstructionCode = None
        self.InstructionCode_prev = None

        self.current_SerialNumber = None #combination of serial number front and back
        self.current_LotNumber = None 

        self.prev_SerialNumber = None
        self.prev_LotNumber = None

        self.currentLot_NOP = [0]*2 #current lot number num of parts [OK, NG]
        self.prevLot_NOP = [0]*2
        self.todayPart_NOP = [0]*2
        self.current_counter_offsets = {
            widget: [0, 0] for widget in self.widget_dir_map
        }
        self.today_counter_offsets = {
            widget: [0, 0] for widget in self.widget_dir_map
        }
        self.current_counter_offset_lot = {
            widget: None for widget in self.widget_dir_map
        }
        self.today_counter_offset_day = datetime.now().strftime("%Y%m%d")
        self.last_emitted_serial_number = None
        self.last_emitted_lot_number = None

        self.temp_prev_OK = 0
        self.temp_prev_NG = 0

        # "Read mysql id and password from yaml file"
        with open("aikensa/mysql/id.yaml") as file:
            credentials = yaml.load(file, Loader=yaml.FullLoader)
            self.mysqlID = credentials["id"]
            self.mysqlPassword = credentials["pass"]
            self.mysqlHost = credentials["host"]
            self.mysqlHostPort = credentials["port"]

        self.holding_register_path = "./aikensa/modbus/holding_register_map.yaml"
        self.widget_inspection_config_path = "./aikensa/parts_config/AGC/widget_inspection_config.yaml"
        self.widget_inspection_configs = self.load_widget_inspection_config(self.widget_inspection_config_path)
        self.command_pause_ms = 150
        self.holding_register_map = load_register_map(self.holding_register_path)
        self.camera_angle = 180.65
        self.last_valid_part_number = 0


    @pyqtSlot(dict)
    def on_holding_update(self, reg_dict):
        # Only called whenever the Modbus thread emits new data.
        self.partNumber = reg_dict.get(50, 0)

        # DEBUG
        self.lotASCIICode_1 = reg_dict.get(52, 0)
        self.lotASCIICode_2 = reg_dict.get(53, 0)
        self.lotASCIICode_3 = reg_dict.get(54, 0)
        self.lotASCIICode_4 = reg_dict.get(55, 0)
        self.lotASCIICode_5 = reg_dict.get(56, 0)
        self.lotASCIICode_6 = reg_dict.get(57, 0)
        self.lotASCIICode_7 = reg_dict.get(58, 0)
        self.lotASCIICode_8 = reg_dict.get(59, 0)

        self.lotNumber_front   = reg_dict.get(60, 0)
        self.lotNumber_back    = reg_dict.get(61, 0)

        self.IV4_KensaResults_OK = reg_dict.get(85, 0)
        self.IV4_KensaResults_NG  = reg_dict.get(86, 0)

        self.serialNumber_front = reg_dict.get(62, 0)
        self.serialNumber_back  = reg_dict.get(63, 0)

        self.InstructionCode_modbus = reg_dict.get(100, 0)

        #combine lot number front and back by appending them into singular number
        #Convert all the lotASCII Code to characters
        self.lotASCIICode_chars = self.convert_lotASCII_to_chars(
            [
            self.lotASCIICode_1,
            self.lotASCIICode_2,
            self.lotASCIICode_3,
            self.lotASCIICode_4,
            self.lotASCIICode_5,
            self.lotASCIICode_6,
            self.lotASCIICode_7,
            self.lotASCIICode_8]
        )
        self.current_LotNumber = f"{self.lotASCIICode_chars}{self.lotNumber_back:05d}{self.lotNumber_front:05d}"
        self.current_SerialNumber = f"{self.serialNumber_back:05d}{self.serialNumber_front:05d}"

        if self.current_LotNumber != self.last_emitted_lot_number:
            self.last_emitted_lot_number = self.current_LotNumber
            self.LotNumber_signal.emit(self.current_LotNumber)

        if self.current_SerialNumber != self.last_emitted_serial_number:
            self.last_emitted_serial_number = self.current_SerialNumber
            self.SerialNumber_signal.emit(self.current_SerialNumber)


    def initialize_single_camera(self, camID):

        self.cap_cam_ic4 = initialize_camera_ic4("37420968",
            width=3072, height=2048, fps=25,
            color=True,
            exposure_us=15000, gain_db=10, wb_temperature=6500,
            auto_exposure=False, auto_gain=False, auto_wb=False,
            first_frame_timeout_ms=200)

        if not self.cap_cam_ic4.isOpened():
            print("Failed to open IC4 camera ")
            self.cap_cam_ic4 = None
        else:
            print("Initialized IC4 camera ")

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

        db_dir = "./aikensa/inspection_results"
        os.makedirs(db_dir, exist_ok=True)

        db_path = os.path.join(db_dir, "agc_database_results.db")

        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        # good defaults for app usage
        self.cursor.execute("PRAGMA journal_mode=WAL;")
        self.cursor.execute("PRAGMA synchronous=NORMAL;")

        # Log table (one row per inspection)
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS inspection_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            partName     INTEGER NOT NULL,  -- 5/6/7/8
            lotNumber    TEXT    NOT NULL,
            serialNumber TEXT    NOT NULL,
            ok_add       INTEGER NOT NULL DEFAULT 0,
            ng_add       INTEGER NOT NULL DEFAULT 0,
            timestamp    TEXT    NOT NULL,
            kensainName  TEXT
        )
        """)

        # Prevent duplicates: at most one row per (part, lot, serial)
        self.cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_part_lot_serial
        ON inspection_results(partName, lotNumber, serialNumber)
        """)

        # Speed up SUM + latest queries
        self.cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_part_lot_time
        ON inspection_results(partName, lotNumber, timestamp)
        """)

        self.conn.commit()
        


        print("Inspection Thread Started")

        self.current_cameraID = self.inspection_config.cameraID
        self.initialize_single_camera(0)

        self.load_crop_coords("aikensa/cameraconfig/part_pos.yaml")


        while self.running:

            if self.inspection_config.debug_mode_selection == 1:
                self.debug = True
                # print("Debug Mode Enabled")
            else:
                self.debug = False
                # print("Debug Mode Disabled")

            if self.inspection_config.widget == 0:
                self.inspection_config.cameraID = -1

            if self.partNumber is not None:
                self.handle_part_number_update()
                # self.partNumber = self.partNumber_modbus
                self.InstructionCode = self.InstructionCode_modbus

            if self.inspection_config.nichijoutenken_mode == True:
                if self.inspection_config.manual_part_selection == 1:
                    self.partNumber = 4
                if self.inspection_config.manual_part_selection == 2:
                    self.partNumber = 3
                if self.inspection_config.manual_part_selection == 3:
                    self.partNumber = 2
                if self.inspection_config.manual_part_selection == 4:
                    self.partNumber = 1

            if self._inspection_command_pending():
                self.ensure_models_initialized()


            if self.inspection_config.widget in [0, 5, 6, 7, 8]:
                ok, self.camFrame_ic4 = self.cap_cam_ic4.read(timeout_ms=self.camera_read_timeout_ms)
                if self.camFrame_ic4 is None:
                    continue
                # self.camFrame_ic4 = cv2.rotate(self.camFrame_ic4, cv2.ROTATE_180)
                #invert rgb to bgr
                self.camFrame_ic4 = cv2.cvtColor(self.camFrame_ic4, cv2.COLOR_BGR2RGB)
            widget_config = self.widget_inspection_configs.get(self.inspection_config.widget)
            if widget_config is not None:
                if self.process_widget_inspection(widget_config):
                    continue
                                         
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

        ok_delta = 0
        ng_delta = 0

        if furyou_plus:
            ng_delta += 1
        if furyou_plus_10:
            ng_delta += 10
        if furyou_minus:
            ng_delta -= 1
        if furyou_minus_10:
            ng_delta -= 10

        if kansei_plus:
            ok_delta += 1
        if kansei_plus_10:
            ok_delta += 10
        if kansei_minus:
            ok_delta -= 1
        if kansei_minus_10:
            ok_delta -= 10

        ok_count_current = max(int(currentPart[0]) + ok_delta, 0)
        ng_count_current = max(int(currentPart[1]) + ng_delta, 0)
        ok_count_total = max(int(Totalpart[0]) + ok_delta, 0)
        ng_count_total = max(int(Totalpart[1]) + ng_delta, 0)

        self.setCounterFalse()
        return [ok_count_current, ng_count_current], [ok_count_total, ng_count_total]
    
    def save_result_database(
        self,
        partName: int,          # 5/6/7/8
        lotNumber: str,
        serialNumber: str,
        currentLOTNOP: list[int],   # [ok_total, ng_total] (optional usage)
        timestampDate: str | None,  # you can pass None; we'll compute
        kensainName: str,
        ok_add: int,
        ng_add: int,
    ):
        partName = int(partName)
        lotNumber = str(lotNumber)
        serialNumber = str(serialNumber)
        ok_add = int(ok_add)
        ng_add = int(ng_add)

        ts = datetime.now()
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")

        try:
            self.cursor.execute("BEGIN")

            # delete old row if exists (same lot+serial+part)
            self.cursor.execute("""
                DELETE FROM inspection_results
                WHERE partName=? AND lotNumber=? AND serialNumber=?
            """, (partName, lotNumber, serialNumber))

            # insert new row
            self.cursor.execute("""
                INSERT INTO inspection_results
                    (partName, lotNumber, serialNumber, ok_add, ng_add, timestamp, kensainName)
                VALUES
                    (?,       ?,        ?,           ?,      ?,      ?,         ?)
            """, (partName, lotNumber, serialNumber, ok_add, ng_add, ts_str, str(kensainName)))

            self.conn.commit()

        except Exception:
            self.conn.rollback()
            raise

    def get_last_entry_currentnumofPart(self, partName: int, lotNumber: str, serialNumber: str) -> list[int]:
        partName = int(partName)
        lotNumber = str(lotNumber)
        serialNumber = str(serialNumber)

        # total for this lot (all serials)
        self.cursor.execute("""
            SELECT COALESCE(SUM(ok_add), 0), COALESCE(SUM(ng_add), 0)
            FROM inspection_results
            WHERE partName=? AND lotNumber=?
        """, (partName, lotNumber))
        total_ok, total_ng = self.cursor.fetchone()
        total_ok, total_ng = int(total_ok), int(total_ng)

        # if this serial already exists, subtract it -> baseline
        self.cursor.execute("""
            SELECT ok_add, ng_add
            FROM inspection_results
            WHERE partName=? AND lotNumber=? AND serialNumber=?
            LIMIT 1
        """, (partName, lotNumber, serialNumber))
        old = self.cursor.fetchone()

        if old:
            old_ok, old_ng = int(old[0]), int(old[1])
            total_ok = max(total_ok - old_ok, 0)
            total_ng = max(total_ng - old_ng, 0)

        return [total_ok, total_ng]
        
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

    def load_widget_inspection_config(self, yaml_path: str) -> dict[int, dict]:
        abs_path = self._resolve_yaml_path(yaml_path)
        with open(abs_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        widget_configs = {}
        for widget_key, config in (data.get("widgets") or {}).items():
            normalized = dict(config)
            normalized["widget"] = int(widget_key)

            tray_config = dict(normalized.get("tray") or {})
            mismatch_ng = tray_config.get("mismatch_ng") or {}
            tray_config["mismatch_ng"] = {
                int(command): int(value)
                for command, value in mismatch_ng.items()
            }
            normalized["tray"] = tray_config

            widget_configs[normalized["widget"]] = normalized

        return widget_configs

    def _get_current_part_crops(self) -> list:
        return [self.part1Crop, self.part2Crop, self.part3Crop, self.part4Crop, self.part5Crop]

    def _set_current_part_crops(self, crops: list) -> None:
        for index, crop in enumerate(crops[:5], start=1):
            setattr(self, f"part{index}Crop", crop)

    def _sync_inspection_images(self) -> None:
        if self.InstructionCode == 0:
            return

        for index, crop in enumerate(self._get_current_part_crops()[:5]):
            self.InspectionImages[index] = crop

    def _pause_for_plc(self) -> None:
        self.msleep(self.command_pause_ms)

    def _prepare_widget_inspection(self, widget_config: dict) -> None:
        self.handle_adjustments_and_counterreset()

        cropped_parts = [
            self.crop_part(self.camFrame_ic4, crop_name, out_w=1771, out_h=121)
            for crop_name in widget_config["part_crop_names"]
        ]
        self._set_current_part_crops(cropped_parts)
        self.process_and_emit_parts(width=self.qtWindowWidth, height=self.qtWindowHeight)

        if self.firstTimeInspection is False and self.inspection_config.doInspection is False:
            self.InspectionTimeStart = time.time()
            self.firstTimeInspection = True
            self.inspection_config.doInspection = True

        self.partNumber_signal.emit(self.partNumber)
        self._sync_inspection_images()

    def _handle_zero_instruction(self) -> None:
        self.requestModbusWrite.emit(self.holding_register_map["return_state_code"], [0])
        if self.InstructionCode_prev == 0:
            return

        self.InstructionCode_prev = self.InstructionCode
        self._reset_inspection_results()
        self._emit_zero_registers()
        self._pause_for_plc()

    def _run_tray_detection(self, widget_config: dict, command: int) -> int:
        tray_config = widget_config["tray"]
        crop_names = tray_config["crop_names"]

        self.Tray_detection_left_image = self.crop_part(self.camFrame_ic4, crop_names["left"], out_w=512, out_h=512)
        self.Tray_detection_right_image = self.crop_part(self.camFrame_ic4, crop_names["right"], out_w=512, out_h=512)

        self.Tray_detection_left_result = aruco_detect_yolo(self.Tray_detection_left_image, model=self.arucoClassificer_model)
        self.Tray_detection_right_result = aruco_detect_yolo(self.Tray_detection_right_image, model=self.arucoClassificer_model)
        print(f"Tray Detection Left Result: {self.Tray_detection_left_result} Tray Detection Right Result: {self.Tray_detection_right_result}")

        expected_left, expected_right = tray_config["expected_ids"]
        if (self.Tray_detection_left_result, self.Tray_detection_right_result) == (expected_left, expected_right):
            print(f"Tray detected as {widget_config['part_name']} correctly.")
            self.InspectionResult_Tray_NG = 0
        else:
            print(f"Tray detection failed or incorrect tray for {widget_config['part_name']}.")
            self.InspectionResult_Tray_NG = tray_config["mismatch_ng"].get(command, 1)

        return self.InspectionResult_Tray_NG

    def _resolve_phase_models(self, phase_config: dict) -> dict:
        models = phase_config.get("models") or {}
        resolved_models = {
            "model_left": getattr(self, models["left"]),
            "model_right": getattr(self, models["right"]),
        }

        center_model = models.get("center")
        if center_model:
            resolved_models["model_center"] = getattr(self, center_model)

        return resolved_models

    def _run_phase_checks(self, phase_name: str, phase_config: dict) -> tuple[list, list, list]:
        parts = self._get_current_part_crops()
        part_count = len(parts)
        exist_images = [None] * part_count
        source_images = list(parts)
        corrected_images = [None] * part_count
        detection_ids = [0] * part_count
        ok_ids = [0] * part_count

        detection_kwargs = {"stream": True, "verbose": False}
        detection_imgsz = phase_config.get("detection_imgsz")
        if detection_imgsz is not None:
            detection_kwargs["imgsz"] = detection_imgsz

        check_kwargs = dict(phase_config.get("check") or {})
        if "center_ok_class" in check_kwargs:
            check_kwargs["center_ok_class"] = check_kwargs.pop("center_ok_class")
        elif "center_class_id" in check_kwargs:
            check_kwargs["center_ok_class"] = check_kwargs.pop("center_class_id")
        check_kwargs.pop("center_bbox_height_range", None)
        check_kwargs["debug_mode"] = self.debug
        model_kwargs = self._resolve_phase_models(phase_config)

        for index, crop in enumerate(parts):
            exist_images[index] = cv2.resize(crop, (512, 512))
            part_exist_result = self.AGC_ALL_WS_DETECTION_model(exist_images[index], **detection_kwargs)
            detection_ids[index] = list(part_exist_result)[0].probs.data.argmax().item()
            corrected_images[index], ok_ids[index], _ = JXX_Check(
                crop,
                **model_kwargs,
                **check_kwargs,
            )

        if phase_name == "set":
            self.SetExistInspectionImages = exist_images
            self.SetCorrectInspectionImages = source_images
            self.SetCorrectInspectionImages_result = corrected_images
        else:
            self.TapeExistInspectionImages = exist_images
            self.TapeCorrectInspectionImages = source_images
            self.TapeCorrectInspectionImages_result = corrected_images

        return corrected_images, detection_ids, ok_ids

    def _finalize_phase_results(self, detection_ids: list, ok_ids: list) -> tuple[list, list, list]:
        detection_result = [int(value) for value in reversed(detection_ids)]
        ok_result = [int(value) for value in reversed(ok_ids)]
        ng_result = [1 - value for value in ok_result]

        for index, detected in enumerate(detection_result):
            if detected == 1:
                ok_result[index] = 0
                ng_result[index] = 0

        return detection_result, ok_result, ng_result

    def _emit_serial_registers(self) -> None:
        self.requestModbusWrite.emit(self.holding_register_map["return_serialNumber_front"], [self.serialNumber_front])
        self.requestModbusWrite.emit(self.holding_register_map["return_serialNumber_back"], [self.serialNumber_back])

    def _emit_serial_and_lot(self) -> None:
        self.SerialNumber_signal.emit(self.current_SerialNumber)
        self.LotNumber_signal.emit(self.current_LotNumber)
        self.last_emitted_serial_number = self.current_SerialNumber
        self.last_emitted_lot_number = self.current_LotNumber

    def _emit_set_results(self) -> None:
        self._emit_serial_registers()
        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_set_partexist"], [self.InspectionResult_DetectionID_int])
        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_set_results_OK"], [self.InspectionResult_SetID_OK_int])
        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_set_results_NG"], [self.InspectionResult_SetID_NG_int])
        self.requestModbusWrite.emit(self.holding_register_map["return_pallet_Error"], [self.InspectionResult_Tray_NG])
        self.requestModbusWrite.emit(self.holding_register_map["return_state_code"], [1])

    def _emit_tape_results(self, emit_pallet_error: bool) -> None:
        self._emit_serial_registers()
        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_tapeinspection_partexist"], [self.InspectionResult_DetectionID_int])
        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_tapeinspection_results_OK"], [self.InspectionResult_TapeID_OK_int])
        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_tapeinspection_results_NG"], [self.InspectionResult_TapeID_NG_int])
        if emit_pallet_error:
            self.requestModbusWrite.emit(self.holding_register_map["return_pallet_Error"], [self.InspectionResult_Tray_NG])
        self.requestModbusWrite.emit(self.holding_register_map["return_state_code"], [2])

    def _emit_iv4_bypass_results(self, emit_pallet_error: bool) -> None:
        self._emit_serial_registers()
        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_tapeinspection_results_OK"], [self.IV4_KensaResults_OK])
        self.requestModbusWrite.emit(self.holding_register_map["return_AIKENSA_KensaResults_tapeinspection_results_NG"], [self.IV4_KensaResults_NG])
        if emit_pallet_error:
            self.requestModbusWrite.emit(self.holding_register_map["return_pallet_Error"], [self.InspectionResult_Tray_NG])
        self.requestModbusWrite.emit(self.holding_register_map["return_state_code"], [2])

    def _save_inspection_images(self, widget_config: dict, phase: str, images: list, ok_results: list, ng_results: list) -> None:
        timestamp = datetime.now()
        date_dir = timestamp.strftime("%Y%m%d")
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        part_name = widget_config["part_name"]
        lot_number = str(self.current_LotNumber or "NOLOT")
        serial_number = str(self.current_SerialNumber or "NOSERIAL")

        for index, image in enumerate(images):
            if image is None:
                continue

            is_ok = index < len(ok_results) and int(ok_results[index]) == 1
            is_ng = index < len(ng_results) and int(ng_results[index]) == 1

            if is_ok:
                result_dir = "OK"
            elif is_ng:
                result_dir = "NG"
            else:
                continue

            save_dir = os.path.join(
                "./aikensa/inspection_results",
                part_name,
                date_dir,
                phase,
                result_dir,
                f"part{index + 1}",
            )
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(
                save_dir,
                f"{timestamp_str}_lot-{lot_number}_serial-{serial_number}.jpg",
            )
            cv2.imwrite(filename, image)
            print(f"Saved {filename}")

    def _refresh_current_lot_totals(self, widget: int) -> None:
        if not self.current_LotNumber:
            self.currentLot_NOP = [0, 0]
            return

        self.cursor.execute(
            """
            SELECT COALESCE(SUM(ok_add), 0), COALESCE(SUM(ng_add), 0)
            FROM inspection_results
            WHERE partName=? AND lotNumber=?
            """,
            (int(widget), str(self.current_LotNumber)),
        )
        ok_total, ng_total = self.cursor.fetchone()
        self.currentLot_NOP = [int(ok_total), int(ng_total)]

    def _refresh_today_part_totals(self, widget: int) -> None:
        self.cursor.execute(
            """
            SELECT COALESCE(SUM(ok_add), 0), COALESCE(SUM(ng_add), 0)
            FROM inspection_results
            WHERE partName=? AND DATE(timestamp) = DATE('now', 'localtime')
            """,
            (int(widget),),
        )
        ok_total, ng_total = self.cursor.fetchone()
        self.todayPart_NOP = [int(ok_total), int(ng_total)]

    def _get_scoped_current_offset(self, widget: int) -> list[int]:
        current_lot = self.current_LotNumber or ""
        if self.current_counter_offset_lot.get(widget) != current_lot:
            self.current_counter_offset_lot[widget] = current_lot
            self.current_counter_offsets[widget] = [0, 0]
        return self.current_counter_offsets[widget]

    def _get_scoped_today_offset(self, widget: int) -> list[int]:
        today_key = datetime.now().strftime("%Y%m%d")
        if self.today_counter_offset_day != today_key:
            self.today_counter_offset_day = today_key
            self.today_counter_offsets = {
                key: [0, 0] for key in self.widget_dir_map
            }
        return self.today_counter_offsets[widget]

    @staticmethod
    def _apply_count_offset(base_counts: list[int], offset_counts: list[int]) -> list[int]:
        return [
            max(int(base_counts[0]) + int(offset_counts[0]), 0),
            max(int(base_counts[1]) + int(offset_counts[1]), 0),
        ]

    def _get_display_current_counts(self, widget: int) -> list[int]:
        return self._apply_count_offset(self.currentLot_NOP, self._get_scoped_current_offset(widget))

    def _get_display_today_counts(self, widget: int) -> list[int]:
        return self._apply_count_offset(self.todayPart_NOP, self._get_scoped_today_offset(widget))

    def _persist_tape_counts(self, widget: int) -> None:
        ok_add = int(sum(self.InspectionResult_TapeID_OK))
        ng_add = int(sum(self.InspectionResult_TapeID_NG))

        self.save_result_database(
            partName=int(widget),
            lotNumber=str(self.current_LotNumber),
            serialNumber=str(self.current_SerialNumber),
            currentLOTNOP=self.currentLot_NOP,
            timestampDate=None,
            kensainName=str(getattr(self, "kensainName", "")),
            ok_add=ok_add,
            ng_add=ng_add,
        )
        self._refresh_current_lot_totals(widget)
        self.prev_LotNumber = self.current_LotNumber
        self.prev_SerialNumber = self.current_SerialNumber
        self.temp_prev_OK = ok_add
        self.temp_prev_NG = ng_add

    def _iv4_bypass_active(self) -> bool:
        return self.lotASCIICode_1 == 22089 and self.lotASCIICode_2 == 52

    def _run_set_command(self, widget_config: dict) -> None:
        if self.InstructionCode_prev == 1:
            return

        self.InstructionCode_prev = self.InstructionCode
        self.inspection_config.doInspection = False
        self._run_tray_detection(widget_config, 1)

        corrected_parts, detection_ids, ok_ids = self._run_phase_checks("set", widget_config["set"])
        self._set_current_part_crops(corrected_parts)
        self.process_and_emit_parts(width=self.qtWindowWidth, height=self.qtWindowHeight)
        self._pause_for_plc()

        self.InspectionResult_DetectionID, self.InspectionResult_SetID_OK, self.InspectionResult_SetID_NG = self._finalize_phase_results(detection_ids, ok_ids)
        self.InspectionResult_DetectionID_int = list_to_16bit_int(self.InspectionResult_DetectionID)
        self.InspectionResult_SetID_OK_int = list_to_16bit_int(self.InspectionResult_SetID_OK)
        self.InspectionResult_SetID_NG_int = list_to_16bit_int(self.InspectionResult_SetID_NG)

        self._emit_set_results()
        if widget_config.get("emit_serial_lot_on_set"):
            self._emit_serial_and_lot()
        print("Inspection Result Set ID Emitted")
        self._pause_for_plc()
        self._save_inspection_images(
            widget_config,
            "set",
            self.SetCorrectInspectionImages,
            self.InspectionResult_SetID_OK,
            self.InspectionResult_SetID_NG,
        )

    def _run_tape_command(self, widget_config: dict, phase_name: str, command: int) -> bool:
        if self.InstructionCode_prev == command:
            print("Already processed Tape Inspection command, skipping...")
            return False

        self.InstructionCode_prev = self.InstructionCode
        self.inspection_config.doTapeInspection = False
        self._run_tray_detection(widget_config, command)

        phase_config = widget_config[phase_name]
        phase_dir = "tape" if phase_name == "tape" else "reinspection"
        emit_pallet_error = widget_config.get("emit_pallet_error_for_tape", True)

        if command == 2 and phase_config.get("iv4_bypass") and self._iv4_bypass_active():
            print("IV4 detected, skipping Tape Inspection...")
            self._emit_iv4_bypass_results(emit_pallet_error)
            print("Inspection Result Tape ID Emitted")
            self._pause_for_plc()
            return True

        if phase_config.get("persist_counts"):
            self.prevLot_NOP = getattr(self, "currentLot_NOP", [0, 0]).copy()
            self.currentLot_NOP = self.get_last_entry_currentnumofPart(
                partName=int(self.inspection_config.widget),
                lotNumber=str(self.current_LotNumber),
                serialNumber=str(self.current_SerialNumber),
            )

        corrected_parts, detection_ids, ok_ids = self._run_phase_checks("tape", phase_config)
        self._set_current_part_crops(corrected_parts)
        self.process_and_emit_parts(width=self.qtWindowWidth, height=self.qtWindowHeight)
        self._pause_for_plc()

        self.InspectionResult_DetectionID, self.InspectionResult_TapeID_OK, self.InspectionResult_TapeID_NG = self._finalize_phase_results(detection_ids, ok_ids)
        self.InspectionResult_DetectionID_int = list_to_16bit_int(self.InspectionResult_DetectionID)
        self.InspectionResult_TapeID_OK_int = list_to_16bit_int(self.InspectionResult_TapeID_OK)
        self.InspectionResult_TapeID_NG_int = list_to_16bit_int(self.InspectionResult_TapeID_NG)

        if phase_config.get("persist_counts"):
            self._persist_tape_counts(self.inspection_config.widget)

        self._emit_tape_results(emit_pallet_error)
        self._emit_serial_and_lot()
        print("Inspection Result Tape ID Emitted")
        self._pause_for_plc()
        self._save_inspection_images(
            widget_config,
            phase_dir,
            self.TapeCorrectInspectionImages,
            self.InspectionResult_TapeID_OK,
            self.InspectionResult_TapeID_NG,
        )
        return False

    def _emit_widget_counts(self, widget_config: dict) -> None:
        widget = int(widget_config["widget"])
        self._refresh_current_lot_totals(widget)
        self._refresh_today_part_totals(widget)
        self.current_numofPart_signal.emit(self._get_display_current_counts(widget))
        self.today_numofPart_signal.emit(self._get_display_today_counts(widget))

    def process_widget_inspection(self, widget_config: dict) -> bool:
        self._prepare_widget_inspection(widget_config)

        if self.InstructionCode == 0:
            self._handle_zero_instruction()

        if self.InstructionCode == 1 or self.inspection_config.doInspection is True:
            self._run_set_command(widget_config)

        if self.InstructionCode == 2 or self.inspection_config.doTapeInspection is True:
            if self._run_tape_command(widget_config, "tape", 2):
                return True

        if self.InstructionCode == 3:
            self._run_tape_command(widget_config, "reinspection", 3)

        self._emit_widget_counts(widget_config)
        self.AGC_InspectionStatus.emit(self.InspectionStatus)
        return False

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

        self.AGC_ALL_WS_DETECTION_model = YOLO("./aikensa/models/WS_ALL_DETECTION.pt")

        self.AGCJ59JRH_SET_LEFT_model = YOLO("./aikensa/models/AGCJ59JRH/SET/AGCJ59JRH_SET_LEFT.pt")
        self.AGCJ59JRH_SET_RIGHT_model = YOLO("./aikensa/models/AGCJ59JRH/SET/AGCJ59JRH_SET_RIGHT.pt")

        self.AGCJ59JRH_TAPE_LEFT_model = YOLO("./aikensa/models/AGCJ59JRH/TAPE/AGCJ59JRH_TAPE_LEFT.pt")
        self.AGCJ59JRH_TAPE_CENTER_model = YOLO("./aikensa/models/AGCJ59JRH/TAPE/AGCJ59JRH_TAPE_CENTER_DETECTION.pt")
        self.AGCJ59JRH_TAPE_RIGHT_model = YOLO("./aikensa/models/AGCJ59JRH/TAPE/AGCJ59JRH_TAPE_RIGHT.pt")

        self.AGCJ59JLH_SET_LEFT_model = YOLO("./aikensa/models/AGCJ59JLH/SET/AGCJ59JLH_SET_LEFT.pt")
        self.AGCJ59JLH_SET_RIGHT_model = YOLO("./aikensa/models/AGCJ59JLH/SET/AGCJ59JLH_SET_RIGHT.pt")

        self.AGCJ59JLH_TAPE_LEFT_model = YOLO("./aikensa/models/AGCJ59JLH/TAPE/AGCJ59JLH_TAPE_LEFT.pt")
        self.AGCJ59JLH_TAPE_CENTER_model = YOLO("./aikensa/models/AGCJ59JLH/TAPE/AGCJ59JLH_TAPE_CENTER_DETECTION.pt")
        self.AGCJ59JLH_TAPE_RIGHT_model = YOLO("./aikensa/models/AGCJ59JLH/TAPE/AGCJ59JLH_TAPE_RIGHT.pt")

        self.AGCJ30LH_SET_LEFT_model = YOLO("./aikensa/models/AGCJ30LH/SET/AGCJ30LH_SET_LEFT.pt")
        self.AGCJ30LH_SET_RIGHT_model = YOLO("./aikensa/models/AGCJ30LH/SET/AGCJ30LH_SET_RIGHT.pt")

        self.AGCJ30LH_TAPE_LEFT_model = YOLO("./aikensa/models/AGCJ30LH/TAPE/AGCJ30LH_TAPE_LEFT.pt")
        self.AGCJ30LH_TAPE_RIGHT_model = YOLO("./aikensa/models/AGCJ30LH/TAPE/AGCJ30LH_TAPE_RIGHT.pt")
        self.AGCJ30LH_TAPE_CENTER_model = YOLO("./aikensa/models/AGCJ30LH/TAPE/AGCJ30LH_TAPE_CENTER_DETECTION.pt")

        self.AGCJ30RH_SET_LEFT_model = YOLO("./aikensa/models/AGCJ30RH/SET/AGCJ30RH_SET_LEFT.pt")
        self.AGCJ30RH_SET_RIGHT_model = YOLO("./aikensa/models/AGCJ30RH/SET/AGCJ30RH_SET_RIGHT.pt")

        self.AGCJ30RH_TAPE_LEFT_model = YOLO("./aikensa/models/AGCJ30RH/TAPE/AGCJ30RH_TAPE_LEFT.pt")
        self.AGCJ30RH_TAPE_RIGHT_model = YOLO("./aikensa/models/AGCJ30RH/TAPE/AGCJ30RH_TAPE_RIGHT.pt")
        self.AGCJ30RH_TAPE_CENTER_model = YOLO("./aikensa/models/AGCJ30RH/TAPE/AGCJ30RH_TAPE_CENTER_DETECTION.pt")

        self.arucoClassificer_model = YOLO("./aikensa/models/ARUCO/aruco.pt")
        self.models_initialized = True

    def ensure_models_initialized(self):
        if self.models_initialized:
            return

        start_time = time.perf_counter()
        logger.info("Initializing inspection models on first inspection request")
        self.initialize_model()
        logger.info("Inspection models initialized in %.2f seconds", time.perf_counter() - start_time)

    def _inspection_command_pending(self) -> bool:
        return (
            self.inspection_config.widget in [5, 6, 7, 8]
            and (
                self.InstructionCode_modbus in [1, 2, 3]
                or self.inspection_config.doInspection is True
                or self.inspection_config.doTapeInspection is True
            )
        )


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
            cur = self._get_display_current_counts(w)
            today = self._get_display_today_counts(w)
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

            self.current_counter_offsets[w] = [
                int(new_cur[0]) - int(self.currentLot_NOP[0]),
                int(new_cur[1]) - int(self.currentLot_NOP[1]),
            ]
            self.today_counter_offsets[w] = [
                int(new_today[0]) - int(self.todayPart_NOP[0]),
                int(new_today[1]) - int(self.todayPart_NOP[1]),
            ]
            cfg.current_numofPart[w], cfg.today_numofPart[w] = new_cur, new_today
            print("Manual Adjustment Done")

        if cfg.counterReset is True:
            current_display = self._get_display_current_counts(w)
            self.current_counter_offsets[w] = [
                -int(self.currentLot_NOP[0]),
                -int(self.currentLot_NOP[1]),
            ]
            cfg.current_numofPart[w] = [0, 0]
            cfg.today_numofPart[w] = self._get_display_today_counts(w)
            cfg.counterReset = False
            print(f"Counter reset applied for widget {w}: {current_display} -> [0, 0]")

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

    def _reset_inspection_results(self):
        """Reset all inspection result arrays to zeros and cache their 16-bit ints."""
        zeros = [0] * 5
        # raw arrays
        self.InspectionResult_DetectionID = zeros.copy()
        self.InspectionResult_SetID_OK    = zeros.copy()
        self.InspectionResult_SetID_NG    = zeros.copy()
        self.InspectionResult_TapeID_OK   = zeros.copy()
        self.InspectionResult_TapeID_NG   = zeros.copy()
        # int encodings
        self.InspectionResult_DetectionID_int = list_to_16bit_int(self.InspectionResult_DetectionID)
        self.InspectionResult_SetID_OK_int    = list_to_16bit_int(self.InspectionResult_SetID_OK)
        self.InspectionResult_SetID_NG_int    = list_to_16bit_int(self.InspectionResult_SetID_NG)
        self.InspectionResult_TapeID_OK_int   = list_to_16bit_int(self.InspectionResult_TapeID_OK)
        self.InspectionResult_TapeID_NG_int   = list_to_16bit_int(self.InspectionResult_TapeID_NG)

    def _emit_zero_registers(self):
        """Emit current zeroed results + serials + state_code to holding registers."""
        write = self.requestModbusWrite.emit
        m = self.holding_register_map

        write(m["return_serialNumber_front"], [self.serialNumber_front])
        write(m["return_serialNumber_back"],  [self.serialNumber_back])

        write(m["return_AIKENSA_KensaResults_set_partexist"],        [self.InspectionResult_DetectionID_int])
        write(m["return_AIKENSA_KensaResults_set_results_OK"],       [self.InspectionResult_SetID_OK_int])
        write(m["return_AIKENSA_KensaResults_set_results_NG"],       [self.InspectionResult_SetID_NG_int])

        write(m["return_AIKENSA_KensaResults_tapeinspection_partexist"],  [self.InspectionResult_DetectionID_int])
        write(m["return_AIKENSA_KensaResults_tapeinspection_results_OK"], [self.InspectionResult_TapeID_OK_int])
        write(m["return_AIKENSA_KensaResults_tapeinspection_results_NG"], [self.InspectionResult_TapeID_NG_int])
        write(m["return_pallet_Error"], [0])
        write(m["return_state_code"], [0])


    def _normalize_text(self, s: str,
                        *,
                        whitespace: str = "none",
                        custom_pattern: Optional[str] = None,
                        transform: Optional[Callable[[str], str]] = None) -> str:
        """Normalize a single string according to the requested whitespace handling."""
        if transform is not None:
            return transform(s)

        if whitespace == "strip":
            return s.strip()
        elif whitespace == "collapse":
            return re.sub(r"\s+", " ", s).strip()
        elif whitespace == "remove":
            return re.sub(r"\s+", "", s)
        elif whitespace == "custom" and custom_pattern:
            return re.sub(custom_pattern, "", s)
        # default: no change
        return s

    def collect_ocr_concat(self,
        ocr,
        images: Iterable[Any],                       # list of np.ndarray or file paths
        image_ids: Optional[Iterable[str]] = None,   # optional labels (e.g., "left_name", "right_pos")
        *,
        # NEW: whitespace controls (pick one)
        whitespace: str = "none",                    # "none"|"strip"|"collapse"|"remove"|"custom"
        custom_pattern: Optional[str] = None,        # used when whitespace="custom" (regex)
        transform: Optional[Callable[[str], str]] = None  # custom function(s) -> str
    ) -> Dict[str, Any]:
        """
        Runs ocr.predict() for each image and concatenates:
        - texts:  List[str]   (normalized if requested)
        - scores: List[float]
        - boxes:  np.ndarray (N, 4/8/...) or None
        - polys:  List[np.ndarray] (each Nx2)
        - index_map: List[(image_id, local_index)]
        Whitespace handling:
        whitespace="strip"    -> trim ends
        whitespace="collapse" -> collapse runs to single space
        whitespace="remove"   -> delete all whitespace
        whitespace="custom"   -> remove regex matches from custom_pattern
        transform=callable    -> apply your own function (overrides whitespace/custom)
        """
        if image_ids is None:
            image_ids = [f"img_{i}" for i, _ in enumerate(images)]

        texts_all: List[str] = []
        scores_all: List[float] = []
        boxes_chunks: List[np.ndarray] = []
        polys_all: List[np.ndarray] = []
        index_map: List[Tuple[str, int]] = []

        for img_id, img in zip(image_ids, images):
            results = ocr.predict(img)   # PaddleOCR v3 pipeline -> List[Result]
            if not results:
                continue

            res = results[0]
            j = res.json["res"]

            texts  = j.get("rec_texts", []) or []
            scores = j.get("rec_scores", []) or []
            boxes  = j.get("rec_boxes", None)
            polys  = j.get("rec_polys", None) or j.get("dt_polys", []) or []

            # normalize + extend
            for k, t in enumerate(texts):
                t_norm = self._normalize_text(
                    t,
                    whitespace=whitespace,
                    custom_pattern=custom_pattern,
                    transform=transform
                )
                texts_all.append(t_norm)
                s = float(scores[k]) if k < len(scores) else float("nan")
                scores_all.append(s)
                index_map.append((img_id, k))

            if boxes is not None and hasattr(boxes, "shape"):
                boxes_chunks.append(np.atleast_2d(boxes))

            if polys:
                polys_all.extend(polys)

        boxes_all = None
        if boxes_chunks:
            try:
                boxes_all = np.vstack(boxes_chunks)
            except Exception:
                boxes_all = np.concatenate(
                    [b.reshape(1, -1) if b.ndim == 1 else b for b in boxes_chunks], axis=0
                )

        return {
            "texts": texts_all,
            "scores": scores_all,
            "boxes": boxes_all,
            "polys": polys_all,
            "index_map": index_map,
        }
    
    def convert_lotASCII_to_chars(self, ascii_words, stop_at_null=True) -> str:
        """
        Convert list of 16-bit words (each contains 2 ASCII bytes) into a string.

        Example word: 21321 = 0x5349 -> bytes [0x49, 0x53] (little-endian) -> 'IS'
        """
        out_bytes = bytearray()

        for w in ascii_words:
            if w is None:
                continue

            w = int(w) & 0xFFFF  # ensure 16-bit

            # Most PLCs store as little-endian within the 16-bit word:
            # low byte first, then high byte.
            b0 = w & 0xFF
            b1 = (w >> 8) & 0xFF

            for b in (b0, b1):
                if b == 0:
                    if stop_at_null:
                        return out_bytes.decode("ascii", errors="ignore")
                    continue
                out_bytes.append(b)

        return out_bytes.decode("ascii", errors="ignore")
