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
# from aikensa.opencv_imgprocessing.detectaruco import detectAruco
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard, calculatecameramatrix
from aikensa.thread.calibration_thread import CalibrationThread, CalibrationConfig
from aikensa.thread.inspection_thread import InspectionThread, InspectionConfig

from aikensa.thread.time_thread import TimeMonitorThread
from aikensa.thread.modbus_thread import ModbusServerThread


# List of UI files to be loaded
UI_FILES = [
    'aikensa/qtui/mainPage.ui',             # index 0
    'aikensa/qtui/calibration_cam.ui', # index 1
    'aikensa/qtui/empty.ui', # index 2
    'aikensa/qtui/empty.ui', # index 3
    'aikensa/qtui/empty.ui', # index 4
    'aikensa/qtui/AGC_LINE.ui', # index 5
    "aikensa/qtui/empty.ui", #empty 6
    "aikensa/qtui/dailyTenken.ui", #empty 7
]


class AIKensa(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        server_ip_address = "0.0.0.0"  # Replace with actual PC IP or "0.0.0.0"
        server_port = 502

        # Starting up the threads
        self.calibration_thread = CalibrationThread(CalibrationConfig())

        self.inspection_thread = InspectionThread(InspectionConfig())   

        self.timeMonitorThread = TimeMonitorThread(check_interval=1)
        self.timeMonitorThread.time_signal.connect(self.timeUpdate)
        self.timeMonitorThread.start()

        self.modbusThread = ModbusServerThread(host=server_ip_address, port=server_port)
        self.modbusThread.start()

        self._setup_ui()

        self.initial_colors = {}#store initial colors of the labels

        self.widget_dir_map = {
            8: "J30LH",
            9: "J30RH",
            10: "J59JLH",
            11: "J59JRH",
        }

        self.prevTriggerStates = 0
        self.TriggerWaitTime = 2.0
        self.currentTime = time.time()

    def timeUpdate(self, time):
        for label in self.timeLabel:
            label.setText(time)


    def trigger_kensa(self):
        self.Inspect_button.click()

    def trigger_rekensa(self):
        self.button_rekensa.click()

    def _setup_ui(self):

        self.inspection_thread.part1Cam.connect(self._setPartFrame1)
        self.inspection_thread.part2Cam.connect(self._setPartFrame2)
        self.inspection_thread.part3Cam.connect(self._setPartFrame3)
        self.inspection_thread.part4Cam.connect(self._setPartFrame4)
        self.inspection_thread.part5Cam.connect(self._setPartFrame5)

        self.inspection_thread.current_numofPart_signal.connect(self._update_OKNG_label)
        self.inspection_thread.today_numofPart_signal.connect(self._update_todayOKNG_label)

        self.inspection_thread.AGC_InspectionStatus.connect(self._inspectionStatusText)

        self.stackedWidget = QStackedWidget()

        for ui in UI_FILES:
            widget = self._load_ui(ui)
            self.stackedWidget.addWidget(widget)

        self.stackedWidget.setCurrentIndex(0)

        main_widget = self.stackedWidget.widget(0)

        cameraCalibration1_widget = self.stackedWidget.widget(1)

        cameraCalibration_button = main_widget.findChild(QPushButton, "camcalibrationbutton")
        # partInspection_P65820W030P_button = main_widget.findChild(QPushButton, "P65820W030Pbutton")

        button_config = {
            "AGClineButton": {"widget_index": 5, "inspection_param": 5},
            "dailytenkenButton": {"widget_index": 7, "inspection_param": 7},
        }

        for button_name, config in button_config.items():
            button = main_widget.findChild(QPushButton, button_name)
            
            if button:
                # Connect each signal with the necessary parameters
                button.clicked.connect(lambda _, idx=config["widget_index"]: self.stackedWidget.setCurrentIndex(idx))
                button.clicked.connect(lambda _, param=config["inspection_param"]: self._set_inspection_params(self.inspection_thread, 'widget', param))
                button.clicked.connect(lambda: self.inspection_thread.start() if not self.inspection_thread.isRunning() else None)
                button.clicked.connect(self.calibration_thread.stop)

        self.timeLabel = [self.stackedWidget.widget(i).findChild(QLabel, "timeLabel") for i in [0, 1, 5]]

        if cameraCalibration_button:
            cameraCalibration_button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))
            cameraCalibration_button.clicked.connect(lambda: self._set_calib_params(self.calibration_thread, 'widget', 1))
            cameraCalibration_button.clicked.connect(self.calibration_thread.start)

        for i in [1]:
            CalibrateSingleFrame = self.stackedWidget.widget(i).findChild(QPushButton, "calibSingleFrame")
            CalibrateSingleFrame.clicked.connect(lambda i=i: self._set_calib_params(self.calibration_thread, "calculateSingeFrameMatrix", True))

            CalibrateFinalCameraMatrix = self.stackedWidget.widget(i).findChild(QPushButton, "calibCam")
            CalibrateFinalCameraMatrix.clicked.connect(lambda i=i: self._set_calib_params(self.calibration_thread, "calculateCamMatrix", True))


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

    def _setCalibFrame(self, image):
        for i in [1, 2, 3, 4, 5]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "camFrame")
            label.setPixmap(QPixmap.fromImage(image))

    def _setMergeFrame1(self, image):
        widget = self.stackedWidget.widget(6)
        label = widget.findChild(QLabel, "camMerge1")
        label.setPixmap(QPixmap.fromImage(image))

    def _setMergeFrame2(self, image):
        widget = self.stackedWidget.widget(6)
        label = widget.findChild(QLabel, "camMerge2")
        label.setPixmap(QPixmap.fromImage(image))

    def _setMergeFrame3(self, image):
        widget = self.stackedWidget.widget(6)
        label = widget.findChild(QLabel, "camMerge3")
        label.setPixmap(QPixmap.fromImage(image))

    def _setMergeFrame4(self, image):
        widget = self.stackedWidget.widget(6)
        label = widget.findChild(QLabel, "camMerge4")
        label.setPixmap(QPixmap.fromImage(image))

    def _setMergeFrame5(self, image):
        widget = self.stackedWidget.widget(6)
        label = widget.findChild(QLabel, "camMerge5")
        label.setPixmap(QPixmap.fromImage(image))

    def _setMergeFrameAll(self, image):
        widget = self.stackedWidget.widget(6)
        label = widget.findChild(QLabel, "camMergeAll")
        label.setPixmap(QPixmap.fromImage(image))

    def _dailyTenkenFrame(self, image):
        widget = self.stackedWidget.widget(21)
        label1 = widget.findChild(QLabel, "dailytenkenFrame")
        label1.setPixmap(QPixmap.fromImage(image))
        
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
        label_names = ["StatusP1", "StatusP2", "StatusP3", "StatusP4", "StatusP5"]
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
                elif status == "テープ OK":
                    label.setStyleSheet("QLabel { background-color: green; }")
                    label.setStyleSheet("QLabel { color: black; }")
                elif status == "テープ NG":
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


def main():
    app = QApplication(sys.argv)
    aikensa = AIKensa()
    aikensa.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()