import cv2
import sys
import yaml
import os
from enum import Enum
import time

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QStackedWidget, QLabel, QSlider, QMainWindow, QWidget, QCheckBox, QShortcut, QLineEdit
from PyQt5.uic import loadUi
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QCoreApplication
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from aikensa.opencv_imgprocessing.cannydetect import canny_edge_detection
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard, calculatecameramatrix
from aikensa.thread.calibration_thread import CalibrationThread, CalibrationConfig
from aikensa.thread.inspection_thread import InspectionThread, InspectionConfig

from aikensa.thread.time_thread import TimeMonitorThread
from aikensa.thread.modbus_thread import ModbusServerThread
from aikensa.thread.camera_thread import CameraThread


# List of UI files to be loaded
UI_FILES = [
    'aikensa/qtui/mainPage.ui',             # index 0
    'aikensa/qtui/empty.ui', # index 1
    'aikensa/qtui/empty.ui', # index 2
    'aikensa/qtui/empty.ui', # index 3
    'aikensa/qtui/empty.ui', # index 4
    'aikensa/qtui/P668307UA0A_COWL_TOP.ui', # index 5
    "aikensa/qtui/empty.ui", #empty 6
]


class AIKensa(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        server_ip_address = "192.168.1.129"  # Replace with actual PC IP or "0.0.0.0"
        server_port = 1502

        self.modbusThread = ModbusServerThread(host=server_ip_address, port=server_port)
        self.inspection_thread = InspectionThread(InspectionConfig(),  modbus_thread=self.modbusThread)  
        self.camera_thread = CameraThread(config_path="./aikensa/thread/TIS_config.yaml")
        self.camera_thread.new_frame.connect(self.inspection_thread.receive_frame)
        self.camera_thread.start()

        self.modbusThread.holdingUpdated.connect(self.inspection_thread.on_holding_update)
        self.modbusThread.start()

        # Starting up the threads
        self.calibration_thread = CalibrationThread(CalibrationConfig())

        #Passthrough the modbus thread to the inspection thread so it can write to the holding registers
 

        self.timeMonitorThread = TimeMonitorThread(check_interval=1)
        self.timeMonitorThread.time_signal.connect(self.timeUpdate)
        self.timeMonitorThread.start()

        self._setup_ui()

        self.initial_colors = {}#store initial colors of the labels

        self.widget_dir_map = {
            5: "P668307UA0A_COWL_TOP",
        }


    def timeUpdate(self, time):
        for label in self.timeLabel:
            label.setText(time)

    def trigger_kensa(self):
        self.Inspect_button.click()

    def trigger_rekensa(self):
        self.button_rekensa.click()

    def _setup_ui(self):

        self.inspection_thread.partNumber_signal.connect(self._set_partNumberUI)

        self.inspection_thread.part1Cam.connect(self._setPartFrame1)
        self.inspection_thread.part2Cam.connect(self._setPartFrame2)
        self.inspection_thread.part3Cam.connect(self._setPartFrame3)
        self.inspection_thread.part4Cam.connect(self._setPartFrame4)
        self.inspection_thread.part5Cam.connect(self._setPartFrame5)
        self.inspection_thread.part6Cam.connect(self._setPartFrame6)
        self.inspection_thread.part7Cam.connect(self._setPartFrame7)
        self.inspection_thread.part8Cam.connect(self._setPartFrame8)
        self.inspection_thread.part9Cam.connect(self._setPartFrame9)
        self.inspection_thread.part10Cam.connect(self._setPartFrame10)

        self.inspection_thread.current_numofPart_signal.connect(self._update_OKNG_label)
        self.inspection_thread.today_numofPart_signal.connect(self._update_todayOKNG_label)

        self.inspection_thread.P668307UA0A_InspectionStatus.connect(self._inspectionStatusText)

        self.stackedWidget = QStackedWidget()

        for ui in UI_FILES:
            widget = self._load_ui(ui)
            self.stackedWidget.addWidget(widget)

        self.stackedWidget.setCurrentIndex(0)

        main_widget = self.stackedWidget.widget(0)

        cameraCalibration1_widget = self.stackedWidget.widget(1)

        cameraCalibration_button = main_widget.findChild(QPushButton, "camcalibrationbutton")

        button_config = {
            "P668307UA0A_button": {"widget_index": 5, "inspection_param": 5},
        }

        for button_name, config in button_config.items():
            button = main_widget.findChild(QPushButton, button_name)
            
            if button:
                # Connect each signal with the necessary parameters
                button.clicked.connect(lambda _, idx=config["widget_index"]: self.stackedWidget.setCurrentIndex(idx))
                button.clicked.connect(lambda _, param=config["inspection_param"]: self._set_inspection_params(self.inspection_thread, 'widget', param))
                button.clicked.connect(lambda: self.inspection_thread.start() if not self.inspection_thread.isRunning() else None)
                button.clicked.connect(self.calibration_thread.stop)

        self.timeLabel = [self.stackedWidget.widget(i).findChild(QLabel, "timeLabel") for i in [0, 5]]

        self.inspection_widget_indices = [5]

        for i in self.inspection_widget_indices:
            self.Inspect_button = self.stackedWidget.widget(i).findChild(QPushButton, "InspectButton")
            if self.Inspect_button:
                self.Inspect_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, "doInspection", True))

        for i in self.inspection_widget_indices:
            self.connect_inspectionConfig_button(i, "kansei_plus", "kansei_plus", True)
            self.connect_inspectionConfig_button(i, "kansei_minus", "kansei_minus", True)
            self.connect_inspectionConfig_button(i, "furyou_plus", "furyou_plus", True)
            self.connect_inspectionConfig_button(i, "furyou_minus", "furyou_minus", True)
            self.connect_inspectionConfig_button(i, "kansei_plus_10", "kansei_plus_10", True)
            self.connect_inspectionConfig_button(i, "kansei_minus_10", "kansei_minus_10", True)
            self.connect_inspectionConfig_button(i, "furyou_plus_10", "furyou_plus_10", True)
            self.connect_inspectionConfig_button(i, "furyou_minus_10", "furyou_minus_10", True)
            #connect reset button
            self.connect_inspectionConfig_button(i, "counterReset", "counterReset", True)
            self.connect_line_edit_text_changed(widget_index=i, line_edit_name="kensain_name", inspection_param="kensainNumber")

        for i in range(self.stackedWidget.count()):
            widget = self.stackedWidget.widget(i)
            button_quit = widget.findChild(QPushButton, "quitbutton")
            button_main_menu = widget.findChild(QPushButton, "mainmenubutton")

            if button_quit:
                button_quit.clicked.connect(self._close_app)

            if button_main_menu:
                button_main_menu.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
                button_main_menu.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, 'widget', 0))
                button_main_menu.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, 'widget', 0))

        self.setCentralWidget(self.stackedWidget)
        self.showFullScreen()


    def connect_inspectionConfig_button(self, widget_index, button_name, cam_param, value):
        widget = self.stackedWidget.widget(widget_index)
        button = widget.findChild(QPushButton, button_name)
        if button:
            button.pressed.connect(lambda: self._set_inspection_params(self.inspection_thread, cam_param, value))
            # print(f"Button '{button_name}' connected to cam_param '{cam_param}' with value '{value}' in widget {widget_index}")

    def _close_app(self):
        # self.cam_thread.stop()
        self.calibration_thread.stop()
        self.inspection_thread.stop()
        self.modbusThread.stop()
        self.timeMonitorThread.stop()
        self.camera_thread.stop()
        
        time.sleep(1.0)
        QCoreApplication.instance().quit()

    def _load_ui(self, filename):
        widget = QMainWindow()
        loadUi(filename, widget)
        return widget


    def _set_button_color(self, pitch_data):
        colorOK = "green"
        colorNG = "red"

        label_names = ["P1color", "P2color", "P3color",
                       "P4color", "P5color", "Lsuncolor"]
        labels = [self.stackedWidget.widget(5).findChild(QLabel, name) for name in label_names]
        for i, pitch_value in enumerate(pitch_data):
            color = colorOK if pitch_value else colorNG
            labels[i].setStyleSheet(f"QLabel {{ background-color: {color}; }}")
        
    def _setPartFrame1(self, image):
        for i in [5]:
            widget = self.stackedWidget.widget(i)
            label1 = widget.findChild(QLabel, "FramePart1")
            label1.setPixmap(QPixmap.fromImage(image))

    def _setPartFrame2(self, image):
        for i in [5]:
            widget = self.stackedWidget.widget(i)
            label2 = widget.findChild(QLabel, "FramePart2")
            label2.setPixmap(QPixmap.fromImage(image))
    
    def _setPartFrame3(self, image):
        for i in [5]:
            widget = self.stackedWidget.widget(i)
            label3 = widget.findChild(QLabel, "FramePart3")
            label3.setPixmap(QPixmap.fromImage(image))

    def _setPartFrame4(self, image):
        for i in [5]:
            widget = self.stackedWidget.widget(i)
            label4 = widget.findChild(QLabel, "FramePart4")
            label4.setPixmap(QPixmap.fromImage(image))
    
    def _setPartFrame5(self, image):
        for i in [5]:
            widget = self.stackedWidget.widget(i)
            label5 = widget.findChild(QLabel, "FramePart5")
            label5.setPixmap(QPixmap.fromImage(image))

    def _setPartFrame6(self, image):
        for i in [5]:
            widget = self.stackedWidget.widget(i)
            label6 = widget.findChild(QLabel, "FramePart6")
            label6.setPixmap(QPixmap.fromImage(image))

    def _setPartFrame7(self, image):
        for i in [5]:
            widget = self.stackedWidget.widget(i)
            label7 = widget.findChild(QLabel, "FramePart7")
            label7.setPixmap(QPixmap.fromImage(image))

    def _setPartFrame8(self, image):
        for i in [5]:
            widget = self.stackedWidget.widget(i)
            label8 = widget.findChild(QLabel, "FramePart8")
            label8.setPixmap(QPixmap.fromImage(image))

    def _setPartFrame9(self, image):
        for i in [5]:
            widget = self.stackedWidget.widget(i)
            label9 = widget.findChild(QLabel, "FramePart9")
            label9.setPixmap(QPixmap.fromImage(image))

    def _setPartFrame10(self, image):
        for i in [5]:
            widget = self.stackedWidget.widget(i)
            label10 = widget.findChild(QLabel, "FramePart10")
            label10.setPixmap(QPixmap.fromImage(image))

    def _update_OKNG_label(self, numofPart):
        for widget_key, part_name in self.widget_dir_map.items():
            # Get OK and NG values using widget_key as index
            if 0 <= widget_key < len(numofPart):
                ok, ng = numofPart[widget_key]
                widget = self.stackedWidget.widget(widget_key)
                if widget:
                    current_kansei_label = widget.findChild(QLabel, "current_kansei")
                    current_furyou_label = widget.findChild(QLabel, "current_furyou")
                    if current_kansei_label:
                        current_kansei_label.setText(str(ok))
                    if current_furyou_label:
                        current_furyou_label.setText(str(ng))
            else:
                print(f"Widget key {widget_key} is out of bounds for numofPart")

    def _update_todayOKNG_label(self, numofPart):
        for widget_key, part_name in self.widget_dir_map.items():
            # Get OK and NG values using widget_key as index
            if 0 <= widget_key < len(numofPart):
                ok, ng = numofPart[widget_key]
                widget = self.stackedWidget.widget(widget_key)
                if widget:
                    current_kansei_label = widget.findChild(QLabel, "status_kansei")
                    current_furyou_label = widget.findChild(QLabel, "status_furyou")
                    if current_kansei_label:
                        current_kansei_label.setText(str(ok))
                    if current_furyou_label:
                        current_furyou_label.setText(str(ng))
            else:
                print(f"Widget key {widget_key} is out of bounds for todaynumofPart")

    def _inspectionStatusText(self, inspectionStatus):

        label_names = ["StatusP1", "StatusP2", "StatusP3", "StatusP4", "StatusP5", "StatusP6", "StatusP7", "StatusP8", "StatusP9", "StatusP10"]
        for i, status in enumerate(inspectionStatus):
            widget = self.stackedWidget.widget(5)
            label = widget.findChild(QLabel, label_names[i])
            if label:
                label.setText(status)
                if status == "製品未検出":
                    label.setStyleSheet("QLabel { background-color: orange; }")
                    label.setStyleSheet("QLabel { color: black; }")
                elif status == "セット OK":
                    label.setStyleSheet("QLabel { background-color: green; }")
                    label.setStyleSheet("QLabel { color: white; }")
                elif status == "セット NG":
                    label.setStyleSheet("QLabel { background-color: red; }")
                    label.setStyleSheet("QLabel { color: white; }")
                elif status == "防錆 OK":
                    label.setStyleSheet("QLabel { background-color: green; }")
                    label.setStyleSheet("QLabel { color: black; }")
                elif status == "防錆 NG":
                    label.setStyleSheet("QLabel { background-color: red; }")
                    label.setStyleSheet("QLabel { color: black; }")

    def _set_calib_params(self, thread, key, value):
        setattr(thread.calib_config, key, value)

    def _set_inspection_params(self, thread, key, value):
        setattr(thread.inspection_config, key, value)

    def connect_line_edit_text_changed(self, widget_index, line_edit_name, inspection_param):
        widget = self.stackedWidget.widget(widget_index)
        line_edit = widget.findChild(QLineEdit, line_edit_name)
        if line_edit:
            line_edit.textChanged.connect(lambda text: self._set_inspection_params(self.inspection_thread, inspection_param, text))

    def on_holding_updated(self, values: dict):
        """
        Slot called whenever the holding registers (0..9) change.
        `values` is a dict {address: new_value}.
        """
        print(f"Received holding registers update: {values}")

        #update the inspection_config.holding_registers with the new values
        self.inspection_thread.inspection_config.holding_registers = values

    def _set_partNumberUI(self, partNumber):
        """
        Update the UI with the part number.
        This method is connected to the partNumber_signal of the inspection_config.
        """
        widget = self.stackedWidget.widget(5)
        if partNumber == 0:
            label = "J30LH"
        elif partNumber == 1:
            label = "J30RH"
        elif partNumber == 2:
            label = "J59JLH"
        elif partNumber == 3:
            label = "J59JRH"
        else:
            label = "Unknown Part"
        partNumber_label = widget.findChild(QLabel, "partName")
        if partNumber_label:
            partNumber_label.setText(label)

def main():
    app = QApplication(sys.argv)
    aikensa = AIKensa()
    aikensa.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()