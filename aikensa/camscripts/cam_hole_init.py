import cv2
import sys

def initialize_hole_camera(camNum): #Init 4k cam

    cap = cv2.VideoCapture(camNum, cv2.CAP_DSHOW) #for ubuntu. It's D_SHOW for windows

    # cap = cv2.VideoCapture(camNum, cv2.CAP_ARAVIS)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    cap.set(cv2.CAP_PROP_AUTO_WB, 0)

    # cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 134)
    # cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, 128)

    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

    # cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 1500)
    # cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)

    # cap.set(cv2.CAP_PROP_GAMMA, 50)
    # cap.set(cv2.CAP_PROP_GAIN, 100)

    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cap.set(cv2.CAP_PROP_FPS, 30) # Set the desired FPS

    return cap
