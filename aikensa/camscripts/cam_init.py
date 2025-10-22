import cv2
import sys

def initialize_camera(camNum): #Init 4k cam

    cap = cv2.VideoCapture(camNum, cv2.CAP_DSHOW) #for ubuntu. It's D_SHOW for windows
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3072)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2048)
    cap.set(cv2.CAP_PROP_FPS, 10) # Set the desired FPS

    return cap
