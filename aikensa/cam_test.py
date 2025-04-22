
import cv2
import os
from datetime import datetime
import numpy as np
import yaml
import time
import csv
import threading
from multiprocessing import Process, Queue

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

from camscripts.cam_init import initialize_camera
from opencv_imgprocessing.cameracalibrate import  warpTwoImages

from dataclasses import dataclass, field
from typing import List, Tuple


class CameraTest:

    def __init__(self):
        self.running = True

        self.widget_dir_map={
            3: "5755A491",
            4: "5755A492"
        }
        
        self.trigger = False


    def start(self):
        
        cap_cam1 = initialize_camera(0)
        print(f"Initiliazing Camera 1.... Located on {cap_cam1}")
        cap_cam2 = initialize_camera(2)
        print(f"Initiliazing Camera 2.... Located on {cap_cam2}")

        cameraMatrix1 = None
        distortionCoeff1 = None
        cameraMatrix2 = None
        distortionCoeff2 = None
        H = None

        #Read the yaml param once
        if os.path.exists("./cameracalibration/cam1calibration_param.yaml"):
            with open("./cameracalibration/cam1calibration_param.yaml") as file:
                cam1calibration_param = yaml.load(file, Loader=yaml.FullLoader)
                cameraMatrix1 = np.array(cam1calibration_param.get('camera_matrix'))
                distortionCoeff1 = np.array(cam1calibration_param.get('distortion_coefficients'))

        if os.path.exists("./cameracalibration/cam2calibration_param.yaml"):
            with open("./cameracalibration/cam2calibration_param.yaml") as file:
                cam2calibration_param = yaml.load(file, Loader=yaml.FullLoader)
                cameraMatrix2 = np.array(cam2calibration_param.get('camera_matrix'))
                distortionCoeff2 = np.array(cam2calibration_param.get('distortion_coefficients'))

        if os.path.exists("./cameracalibration/homography_param.yaml"):
            with open("./cameracalibration/homography_param.yaml") as file:
                homography_param = yaml.load(file, Loader=yaml.FullLoader)
                H = np.array(homography_param)

        # print (f"camMatrix: \n {cameraMatrix1}")
        # print (f"distCoeff: \n {distortionCoeff1}")
        # print (f"camMatrix: \n {cameraMatrix2}")
        # print (f"distCoeff: \n {distortionCoeff2}")
        # print (f"Homography: \n {H}")

        cap_cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 3072)
        cap_cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 2048)
        cap_cam2.set(cv2.CAP_PROP_FRAME_WIDTH, 3072)
        cap_cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 2048)
        # cap_cam1.set(cv2.CAP_PROP_FRAME_WIDTH, int(3840/4))
        # cap_cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, int(2160/4))
        # cap_cam2.set(cv2.CAP_PROP_FRAME_WIDTH, int(3840/4))
        # cap_cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, int(2160/4))
        map1=np.zeros((2048,3072),dtype=np.float32)
        map2=np.zeros((2048,3072),dtype=np.float32)

        while self.running:

            start_time = time.perf_counter()


            ret1, frame1 = cap_cam1.read()
            ret2, frame2 = cap_cam2.read()

            # frame1 = self.resizeImage(frame1, 3072//4, 2048//4)
            # frame2 = self.resizeImage(frame2, 3072//4, 2048//4)

            print(frame1.shape, frame2.shape)
            if frame1 is None:
                frame1 = np.zeros((2048, 3072, 3), dtype=np.uint8)
                # print("Frame 1 is None")
            if frame2 is None:
                frame2 = np.zeros((2048, 3072, 3), dtype=np.uint8) 
                # print("Frame 2 is None")
            
            # process frame1
            time_frame1_undistort_start = time.perf_counter()
            frame1 = self.undistortFrame(frame1, cameraMatrix1, distortionCoeff1,map1,map2)
            time_frame1_undistort_finish = time.perf_counter()
            #process frame2
            time_frame2_undistort_start = time.perf_counter()
            frame2 = self.undistortFrame(frame2, cameraMatrix1, distortionCoeff1,map1,map2)
            time_frame2_undistort = time.perf_counter()
            #merge frame1 and frame2
            time_combine_frames_start = time.perf_counter()
            combinedFrame_raw, combinedImage, croppedFrame1, croppedFrame2 = self.combineFrames(frame1, frame2, H)
            time_combine_frames_finish = time.perf_counter()

            # print(f"Undistort Frame 1: {time_frame1_undistort_finish - time_frame1_undistort_start:.3f} s")
            # print(f"Undistort Frame 2: {time_frame2_undistort - time_frame2_undistort_start:.3f} s")
            # print(f"Combine Frames: {time_combine_frames_finish - time_combine_frames_start:.3f} s")

            print(f"Total Time: {time_combine_frames_finish - start_time:.3f} s")
            print()



        cap_cam1.release()
        print("Camera 1 released.")
        cap_cam2.release()
        print("Camera 2 released.")
        

    def cap_frames(self, camIndex, outputQueue):
        cap = initialize_camera(camIndex)
        while self.running:
            ret, frame = cap.read()
            if ret:
                outputQueue.put(frame)
        cap.release()


    # def undistortFrame(self, frame,cameraMatrix, distortionCoeff):
    def undistortFrame(self, frame,cameraMatrix, distortionCoeff,map1=None,map2=None):
        elapsed=-time.time()
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # map1, map2 = cv2.initUndistortRectifyMap(cameraMatrix, distortionCoeff, None, cameraMatrix, (frame.shape[1], frame.shape[0]), cv2.CV_32FC1)
        # print(map1.shape, map2.shape)
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        # frame = cv2.undistort(frame, cameraMatrix, distortionCoeff, None, cameraMatrix)
        elapsed+=time.time()
        print(f'Undistort took {elapsed:.3f} seconds')
        return frame

    def combineFrames(self, frame1, frame2, H):
        # elapsed = -time.time()
        combinedFrame = warpTwoImages(frame2, frame1, H)
        
        # elapsed += time.time()
        # print(f'Warping took {elapsed:.3f} seconds')

        # elapsed = -time.time()
        combinedFrame, _ = self.planarize(combinedFrame)
        # elapsed += time.time()
        # print(f'Planarize took {elapsed:.3f} seconds')

        # elapsed = -time.time()
        combinedFrame_raw = combinedFrame.copy()
        # elapsed += time.time()
        # print(f'Copy combinedframe took {elapsed:.3f} seconds')

        # elapsed = -time.time()
        combinedFrame = self.resizeImage(combinedFrame, 1521, 363)
        # elapsed += time.time()
        # print(f'Copy resizeImage took {elapsed:.3f} seconds')

        croppedFrame1 = self.frameCrop(combinedFrame_raw, x=450, y=260, w=320, h=160, wout = 320, hout = 160)
        croppedFrame2 = self.frameCrop(combinedFrame_raw, x=3800, y=260, w=320, h=160, wout = 320, hout = 160)
        return combinedFrame_raw, combinedFrame, croppedFrame1, croppedFrame2

    
    
    def planarize(self, image):
        transform = None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        multiplier = 5.0
        IMAGE_HEIGHT = int(218 * multiplier)
        IMAGE_WIDTH = int(913 * multiplier)

        if os.path.exists("./param/warptransform.yaml"):
            with open('./param/warptransform.yaml', 'r') as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                transform = np.array(transform_list)
            image = cv2.warpPerspective(image, transform, (IMAGE_WIDTH, IMAGE_HEIGHT))



            return image, None

        else:
            corners, ids, rejected = detector.detectMarkers(gray)
            print (corners)
            print (ids)
            print (rejected)
            if corners and ids is not None:

                top_left_corner = None
                top_right_corner = None
                bottom_left_corner = None
                bottom_right_corner = None

                for i, corner in zip(ids, corners):
                    marker_id = i[0]
                    if marker_id == 0:
                        # Top left corner of marker 0
                        top_left_corner = corner[0][0]
                    elif marker_id == 1:
                        # Top right corner of marker 1
                        top_right_corner = corner[0][1]
                    elif marker_id == 2:
                        # Bottom left corner of marker 2
                        bottom_left_corner = corner[0][3]
                    elif marker_id == 3:
                        # Bottom right corner of marker 3
                        bottom_right_corner = corner[0][2]

                if top_left_corner is not None and top_right_corner is not None \
                and bottom_left_corner is not None and bottom_right_corner is not None:
                    # Concatenate the corners in the desired order
                    ordered_corners = np.array([
                        top_left_corner, top_right_corner,
                        bottom_left_corner, bottom_right_corner
                    ], dtype='float32')
    
                    transform = cv2.getPerspectiveTransform(ordered_corners, desired_plane)
                    image = cv2.warpPerspective(image, transform, (IMAGE_WIDTH, IMAGE_HEIGHT))

                return image, transform
            else:
                return image, None

    def stop(self):
        self.running = False
        time.sleep(0.5)

    def resizeImage(self, image, width=384, height=256):
        # Resize image using cv2.resize
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return resized_image
    
    def stop(self):
        self.running = False
        print(self.running)

    def convertQImage(self, image):
        # Convert resized cv2 image to QImage
        h, w, ch = image.shape
        bytesPerLine = ch * w
        processed_image = QImage(image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return processed_image

    def downSampling(self, image, width=384, height=256):
        # Resize image using cv2.resize
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        # Convert resized cv2 image to QImage
        h, w, ch = resized_image.shape
        bytesPerLine = ch * w
        processed_image = QImage(resized_image.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return processed_image

    def frameCrop(self,img=None, x=0, y=0, w=640, h=480, wout=640, hout=480):
        #crop and resize image to wout and hout
        img = img[y:y+h, x:x+w]
        img = cv2.resize(img, (wout, hout), interpolation=cv2.INTER_AREA)
        return img


if __name__ == "__main__":
    camera_test = CameraTest()
    camera_test.start()