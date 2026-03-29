import cv2


def initialize_camera(camNum):
    cap = cv2.VideoCapture(camNum, cv2.CAP_DSHOW)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3072)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2048)
    cap.set(cv2.CAP_PROP_FPS, 10)

    return cap