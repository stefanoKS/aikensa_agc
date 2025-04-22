import cv2
import numpy as np
import yaml
import os

def detectAruco4_4(image):
    dict_type = cv2.aruco.DICT_4X4_250
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
    return image

def detectAruco5_5(image):
    dict_type = cv2.aruco.DICT_5X5_250
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
    return image

def detectAruco6_6(image):
    dict_type = cv2.aruco.DICT_5X5_1000
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
    return image