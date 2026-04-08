import cv2
import sys
import yaml
import os
from enum import Enum
import time

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QStackedWidget, QLabel, QSlider, QMainWindow, QWidget, QCheckBox, QShortcut, QLineEdit, QRadioButton, QComboBox
from PyQt5.uic import loadUi
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QCoreApplication
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from aikensa.opencv_imgprocessing.cannydetect import canny_edge_detection
# from aikensa.opencv_imgprocessing.detectaruco import detectAruco
from aikensa.opencv_imgprocessing.cameracalibrate import detectCharucoBoard, calculatecameramatrix
from aikensa.thread.calibration_thread import CalibrationThread, CalibrationConfig
from aikensa.thread.inspection_thread import InspectionThread, InspectionConfig
from aikensa.thread.tape_camera_thread import TapeCameraThread

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

        server_ip_address = "192.168.3.11"  # Replace with actual PC IP or "0.0.0.0"
        server_port = 502

        self.modbusThread = ModbusServerThread(host=server_ip_address, port=server_port)
        self.inspection_thread = InspectionThread(InspectionConfig(),  modbus_thread=self.modbusThread)  

        self.inspection_thread.SerialNumber_signal.connect(self._update_serial_number)
        self.inspection_thread.LotNumber_signal.connect(self._update_lot_number)
        self.inspection_thread.DailyPartsPerHour_signal.connect(self._update_jikan_atari)
        self.modbusThread.plcConnectionStatusChanged.connect(self._update_plc_status_label)
        self.modbusThread.start()
        self.inspection_thread.start()

        # Starting up the threads
        self.calibration_thread = CalibrationThread(CalibrationConfig())

        # Initialize tape camera thread (second camera, OpenCV, index 0, 1280x720)
        # Will auto-search for available camera if index 0 not available
        self.tape_camera_thread = TapeCameraThread(camera_index=0, width=1280, height=720)
        self.tape_camera_thread.frameSignal.connect(self._setTapeFrame)
        self.tape_camera_thread.start()

        #Passthrough the modbus thread to the inspection thread so it can write to the holding registers
        self.timeMonitorThread = TimeMonitorThread(check_interval=1)
        self.timeMonitorThread.time_signal.connect(self.timeUpdate)
        self.timeMonitorThread.start()

        self._setup_ui()
        self._update_jikan_atari(getattr(self.inspection_thread, "daily_parts_per_hour", 0.0))

        self.initial_colors = {}#store initial colors of the labels

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

        self.inspection_thread.partNumber_signal.connect(self._set_partNumberUI)

        self.inspection_thread.part1Cam.connect(self._setPartFrame1)
        self.inspection_thread.part2Cam.connect(self._setPartFrame2)
        self.inspection_thread.part3Cam.connect(self._setPartFrame3)
        self.inspection_thread.part4Cam.connect(self._setPartFrame4)
        self.inspection_thread.part5Cam.connect(self._setPartFrame5)
        self.inspection_thread.trayLeftCam.connect(self._setTrayEmitLeft)
        self.inspection_thread.trayRightCam.connect(self._setTrayEmitRight)

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

        # Go directly to widget index 5 with inspection param 5 (for debugging)
        self.stackedWidget.setCurrentIndex(5)
        self._set_inspection_params(self.inspection_thread, 'widget', 5)
        if not self.inspection_thread.isRunning():
            self.inspection_thread.start()
        self.calibration_thread.stop()

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
            self.Inspect_tape_button = self.stackedWidget.widget(i).findChild(QPushButton, "InspectTapeButton")

            if self.Inspect_button:
                self.Inspect_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, "doInspection", True))
            if self.Inspect_tape_button:
                self.Inspect_tape_button.clicked.connect(lambda: self._set_inspection_params(self.inspection_thread, "doTapeInspection", True))

            

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
            self.connect_QComboBox_changed(widget_index=i, combo_name="comboSelection", inspection_param="manual_part_selection")
            self.connect_QComboBox_changed(widget_index=i, combo_name="comboSelection_ONOFF", inspection_param="debug_mode_selection")
            self.connect_radio_button_toggle(widget_index=i, radio_name="SET_L", inspection_param="debug_bypass_set_left")
            self.connect_radio_button_toggle(widget_index=i, radio_name="SET_R", inspection_param="debug_bypass_set_right")
            self.connect_radio_button_toggle(widget_index=i, radio_name="TAPE_L", inspection_param="debug_bypass_tape_left")
            self.connect_radio_button_toggle(widget_index=i, radio_name="TAPE_C", inspection_param="debug_bypass_tape_center")
            self.connect_radio_button_toggle(widget_index=i, radio_name="TAPE_R", inspection_param="debug_bypass_tape_right")

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

        # Radio button for nichijoutenken
        self.connect_radio_to_button_visibility(widget_index=5, radio_name="nichijoutenken_radio", target_button_name="comboSelection", inspection_param="nichijoutenken_mode")
        self.connect_radio_to_button_visibility(widget_index=5, radio_name="debug_mode_radio", target_button_name="comboSelection_ONOFF", inspection_param="debug_mode")


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
        self.tape_camera_thread.stop()

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
            if label1:
                scaled_pixmap = QPixmap.fromImage(image).scaled(
                    label1.width(), label1.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                label1.setPixmap(scaled_pixmap)

    def _setPartFrame2(self, image):
        for i in [5]:
            widget = self.stackedWidget.widget(i)
            label2 = widget.findChild(QLabel, "FramePart2")
            if label2:
                scaled_pixmap = QPixmap.fromImage(image).scaled(
                    label2.width(), label2.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                label2.setPixmap(scaled_pixmap)

    def _setPartFrame3(self, image):
        for i in [5]:
            widget = self.stackedWidget.widget(i)
            label3 = widget.findChild(QLabel, "FramePart3")
            if label3:
                scaled_pixmap = QPixmap.fromImage(image).scaled(
                    label3.width(), label3.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                label3.setPixmap(scaled_pixmap)

    def _setPartFrame4(self, image):
        for i in [5]:
            widget = self.stackedWidget.widget(i)
            label4 = widget.findChild(QLabel, "FramePart4")
            if label4:
                scaled_pixmap = QPixmap.fromImage(image).scaled(
                    label4.width(), label4.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                label4.setPixmap(scaled_pixmap)

    def _setPartFrame5(self, image):
        for i in [5]:
            widget = self.stackedWidget.widget(i)
            label5 = widget.findChild(QLabel, "FramePart5")
            if label5:
                scaled_pixmap = QPixmap.fromImage(image).scaled(
                    label5.width(), label5.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                label5.setPixmap(scaled_pixmap)

    def _setTrayEmitLeft(self, image):
        for i in [5]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "tray_emit_L")
            if label:
                scaled_pixmap = QPixmap.fromImage(image).scaled(
                    label.width(), label.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                label.setPixmap(scaled_pixmap)

    def _setTrayEmitRight(self, image):
        for i in [5]:
            widget = self.stackedWidget.widget(i)
            label = widget.findChild(QLabel, "tray_emit_R")
            if label:
                scaled_pixmap = QPixmap.fromImage(image).scaled(
                    label.width(), label.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                label.setPixmap(scaled_pixmap)

    def _setTapeFrame(self, image):
        """Display tape camera stream on TapeFrame QLabel."""
        widget = self.stackedWidget.widget(5)
        if widget:
            label = widget.findChild(QLabel, "TapeFrame")
            if label:
                # Scale the image to fit the label
                scaled_pixmap = QPixmap.fromImage(image).scaled(
                    label.width(), label.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                label.setPixmap(scaled_pixmap)

    def _get_agc_widget(self):
        if not hasattr(self, "stackedWidget"):
            return None
        if self.stackedWidget.count() <= 5:
            return None
        return self.stackedWidget.widget(5)

    def _update_OKNG_label(self, numofPart):
        ok, ng = numofPart[0], numofPart[1]
        widget = self._get_agc_widget()
        if widget:
            current_kansei_label = widget.findChild(QLabel, "current_kansei")
            current_furyou_label = widget.findChild(QLabel, "current_furyou")
            if current_kansei_label:
                current_kansei_label.setText(str(ok))
            if current_furyou_label:
                current_furyou_label.setText(str(ng))

    def _update_todayOKNG_label(self, numofPart):
        ok, ng = numofPart[0], numofPart[1]
        widget = self._get_agc_widget()
        if widget:
            current_kansei_label = widget.findChild(QLabel, "status_kansei")
            current_furyou_label = widget.findChild(QLabel, "status_furyou")
            if current_kansei_label:
                current_kansei_label.setText(str(ok))
            if current_furyou_label:
                current_furyou_label.setText(str(ng))

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

    def connect_radio_button_toggle(self, widget_index, radio_name, inspection_param):
        widget = self.stackedWidget.widget(widget_index)
        radio = widget.findChild(QRadioButton, radio_name)
        if not radio:
            return

        radio.setAutoExclusive(False)
        initial_value = bool(getattr(self.inspection_thread.inspection_config, inspection_param, radio.isChecked()))
        radio.blockSignals(True)
        radio.setChecked(initial_value)
        radio.blockSignals(False)
        self._set_inspection_params(self.inspection_thread, inspection_param, initial_value)
        radio.toggled.connect(
            lambda checked,
                thread=self.inspection_thread,
                param=inspection_param: self._set_inspection_params(thread, param, checked)
        )

    def connect_line_edit_text_changed(self, widget_index, line_edit_name, inspection_param):
        widget = self.stackedWidget.widget(widget_index)
        line_edit = widget.findChild(QLineEdit, line_edit_name)
        if line_edit:
            line_edit.textChanged.connect(lambda text: self._set_inspection_params(self.inspection_thread, inspection_param, text))

    def connect_QComboBox_changed(self, widget_index, combo_name, inspection_param):
        widget = self.stackedWidget.widget(widget_index)
        combo_box: QComboBox = widget.findChild(QComboBox, combo_name)
        if not combo_box:
            return

        # index -> value map (make it a list or dict, not a set)
        id_map = [None, 1, 2, 3, 4]  # or {0: None, 1: 1, 2: 2, 3: 3, 4: 4}

        def on_index_changed(index: int):
            if 0 <= index < len(id_map):
                value = id_map[index]
            else:
                value = None
            self._set_inspection_params(self.inspection_thread, inspection_param, value)

        combo_box.currentIndexChanged[int].connect(on_index_changed)
        
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
        if partNumber == 1:
            label = "AGC J59J RH"
        elif partNumber == 2:
            label = "AGC J59J LH"
        elif partNumber == 3:
            label = "AG 910930 RH"
        elif partNumber == 4:
            label = "AG 910931 LH"
        else:
            label = "Unknown Part"
        partNumber_label = widget.findChild(QLabel, "partName")

        if partNumber != 0:
            if partNumber_label:
                partNumber_label.setText(label)


    def connect_radio_to_button_visibility(self,
                                       widget_index: int,
                                       radio_name: str,
                                       target_button_name: str,
                                       inspection_param: str | None = None):
        """
        - Shows/hides `target_button_name` based on `radio_name` checked state.
        - If `inspection_param` is provided, forwards the checked state to self._set_inspection_params(...)
        """
        widget = self.stackedWidget.widget(widget_index)
        radio  = widget.findChild(QRadioButton, radio_name)
        target = widget.findChild(QComboBox, target_button_name)

        if not radio or not target:
            # Silently ignore if either control isn't present on this widget
            return

        # Set initial visibility to match current radio state
        target.setVisible(radio.isChecked())

        # Update visibility on toggle
        radio.toggled.connect(lambda checked, btn=target: btn.setVisible(checked))

        # Optionally propagate the state to your inspection thread
        if inspection_param:
            radio.toggled.connect(
                lambda checked,
                    thread=self.inspection_thread,
                    param=inspection_param: self._set_inspection_params(thread, param, checked)
            )

    def _update_serial_number(self, serial_number):
        widget = self._get_agc_widget()
        if not widget:
            return
        serial_label = widget.findChild(QLabel, "status_SERIALNO")
        if serial_label:
            serial_label.setText(f"{serial_number}")

    def _update_lot_number(self, lot_number):
        widget = self._get_agc_widget()
        if not widget:
            return
        lot_label = widget.findChild(QLabel, "status_LOTNO")
        if lot_label:
            lot_label.setText(f"{lot_number}")

    def _update_jikan_atari(self, parts_per_hour):
        widget = self._get_agc_widget()
        if not widget:
            return

        label = widget.findChild(QLabel, "JIKAN_ATARI")
        if label:
            label.setText(f"{float(parts_per_hour):.1f}")

    def _update_plc_status_label(self, connected: bool):
        widget = self._get_agc_widget()
        if not widget:
            return

        label = widget.findChild(QLabel, "PLC_STATUS_label")
        if not label:
            return

        if connected:
            label.setText("CONNECTED")
            label.setStyleSheet("color: rgb(46, 204, 113);")
        else:
            label.setText("NOT CONNECTED")
            label.setStyleSheet("color: rgb(239, 41, 41);")




def main():
    app = QApplication(sys.argv)
    aikensa = AIKensa()
    aikensa.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()