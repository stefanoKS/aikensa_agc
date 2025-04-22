from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QStackedWidget, QLabel
import sys

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.stackedWidget = QStackedWidget(self)
        self.cam_thread = type('CamThread', (object,), {'cam_config': type('CamConfig', (object,), {'buttonH_HDResQT': False})()})()

        # Example widget and button setup
        self.widget1 = QWidget()
        self.button1 = QPushButton('HDResButton', self.widget1)
        self.label = QLabel('Current State: False', self.widget1)
        self.widget1.layout = QVBoxLayout(self.widget1)
        self.widget1.layout.addWidget(self.button1)
        self.widget1.layout.addWidget(self.label)

        self.stackedWidget.addWidget(self.widget1)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.stackedWidget)
        self.setLayout(main_layout)

        self.setWindowTitle('Color and Parameter Change Example')
        self.show()

        # Example usage of connect_camparam_colorChange
        self.connect_camparam_colorChange(0, "buttonH_HDResQT", "HDResButton")

    def _set_cam_params(self, cam_thread, cam_param, value):
        # Placeholder for the actual parameter setting logic
        print(f'Setting {cam_param} to {value}')

    def _toggle_param_and_update_label(self, param, button, label, color1_rgb, color2_rgb):
        # Toggle the parameter value
        current_value = getattr(self.cam_thread.cam_config, param, False)
        new_value = not current_value
        setattr(self.cam_thread.cam_config, param, new_value)
        self._set_cam_params(self.cam_thread, param, new_value)

        # Update the label color based on the new parameter value
        color = "green" if new_value else "red"
        label.setStyleSheet(f"QLabel {{ background-color: {color}; }}")
        label.setText(f'Current State: {new_value}')

        # Toggle the button color
        current_button_color = button.palette().button().color().name()
        new_button_color = color2_rgb if current_button_color == color1_rgb else color1_rgb
        button.setStyleSheet(f"background-color: {new_button_color}")

    def connect_camparam_colorChange(self, widget_index, cam_param, qtbutton, label):
        widget = self.stackedWidget.widget(widget_index)
        button = widget.findChild(QPushButton, qtbutton)

        if button:
            color1 = (211, 211, 211)  # RGB for light gray
            color2 = (173, 216, 230)  # RGB for light blue
            color1_rgb = f"rgb({color1[0]}, {color1[1]}, {color1[2]})"
            color2_rgb = f"rgb({color2[0]}, {color2[1]}, {color2[2]})"
            button.setStyleSheet(f"background-color: {color1_rgb}")

            button.pressed.connect(lambda: self._toggle_param_and_update_label(cam_param, button, label, color1_rgb, color2_rgb))

def main():
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
