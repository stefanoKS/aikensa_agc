import cv2
import numpy as np
import yaml
import os

from dataclasses import dataclass


dict_type = cv2.aruco.DICT_4X4_250
squares = (24, 16)
square_length = 0.025
marker_length = 0.020
aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
charboard = cv2.aruco.CharucoBoard(squares, square_length, marker_length, aruco_dict)
detector = cv2.aruco.CharucoDetector(charboard)

squares_large = (27, 11)
square_length_large = 0.030
marker_length_large = 0.025
aruco_dict_large = cv2.aruco.getPredefinedDictionary(dict_type)
charboard_large = cv2.aruco.CharucoBoard(squares_large, square_length_large, marker_length_large, aruco_dict_large)
detector_large = cv2.aruco.CharucoDetector(charboard_large)

allCharucoCorners = []
allCharucoIds = []
allObjectPoints = []
allImagePoints = []
imageSize = None
calibration_image = 0

allCharucoCorners_scaledImage = []
allCharucoIds_scaledImage = []
allObjectPoints_scaledImage = []
allImagePoints_scaledImage = []
imageSize_scaledImage = None
calibration_image_scaledImage = 0




@dataclass
class FontConfig:
    font_face: int = 0
    font_scale: float = 0.5
    font_thickness: int = 1
    font_color: tuple = (255, 255, 255)
    font_position: tuple = (0, 0)

def detectCharucoBoard(image):
    global allCharucoCorners, allCharucoIds, allObjectPoints, allImagePoints, imageSize, calibration_image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    imageSize = (width, height)

    charucoCorners, charucoIds, markerCorners, markersIds = detector.detectBoard(gray)   
    
    if charucoCorners is not None and charucoIds is not None:
        # print("Charuco board detected.")
        allCharucoCorners.append(charucoCorners)
        allCharucoIds.append(charucoIds)
        currentObjectPoints, currentImagePoints = charboard.matchImagePoints(charucoCorners, charucoIds)
        allObjectPoints.append(currentObjectPoints)
        allImagePoints.append(currentImagePoints)

        #add calibration_image to the top of the image
        

    #Lets draw the markers
    image = cv2.aruco.drawDetectedMarkers(image, markerCorners, markersIds)

    # print (allCharucoIds)

    return image, charucoCorners, charucoIds

#make the exact same function but different naming for the global variables

def detectCharucoBoard_scaledImage(image):
    global allCharucoCorners_scaledImage, allCharucoIds_scaledImage, allObjectPoints_scaledImage, allImagePoints_scaledImage, imageSize_scaledImage, calibration_image_scaledImage

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    imageSize_scaledImage = (width, height)

    charucoCorners, charucoIds, markerCorners, markersIds = detector.detectBoard(gray)   
    
    if charucoCorners is not None and charucoIds is not None:
        # print("Charuco board detected.")
        allCharucoCorners_scaledImage.append(charucoCorners)
        allCharucoIds_scaledImage.append(charucoIds)
        currentObjectPoints, currentImagePoints = charboard.matchImagePoints(charucoCorners, charucoIds)
        allObjectPoints_scaledImage.append(currentObjectPoints)
        allImagePoints_scaledImage.append(currentImagePoints)

    image = cv2.aruco.drawDetectedMarkers(image, markerCorners, markersIds)

    return image, charucoCorners, charucoIds

def detectCharucoBoardLarge(image):
    global allCharucoCorners, allCharucoIds, allObjectPoints, allImagePoints, imageSize, calibration_image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    imageSize = (width, height)

    charucoCorners, charucoIds, markerCorners, markersIds = detector_large.detectBoard(gray)   
    
    if charucoCorners is not None and charucoIds is not None:
        # print("Charuco board detected.")
        allCharucoCorners.append(charucoCorners)
        allCharucoIds.append(charucoIds)
        currentObjectPoints, currentImagePoints = charboard_large.matchImagePoints(charucoCorners, charucoIds)
        allObjectPoints.append(currentObjectPoints)
        allImagePoints.append(currentImagePoints)

        #add calibration_image to the top of the image
        

    #Lets draw the markers
    image = cv2.aruco.drawDetectedMarkers(image, markerCorners, markersIds)

    #print (allCharucoIds)
    allCharucoCorners = []
    allCharucoIds = []
    allObjectPoints = []
    allImagePoints = []
    imageSize = None
    calibration_image = 0

    return image, charucoCorners, charucoIds
    
def detectCharucoBoard_aprilTag(image):
    dict_type = cv2.aruco.DICT_APRILTAG_36H10
    squares = (83, 16)
    square_length = 0.030
    marker_length = 0.025
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    charboard = cv2.aruco.CharucoBoard(squares, square_length, marker_length, aruco_dict)
    detector = cv2.aruco.CharucoDetector(charboard)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    charucoCorners, charucoIds, markerCorners, markersIds = detector.detectBoard(gray)   
    # image = cv2.aruco.drawDetectedMarkers(image, markerCorners, markersIds)

    return image, charucoCorners, charucoIds
    

def detectCharucoBoard_6_6(image):
    dict_type = cv2.aruco.DICT_6X6_1000
    squares = (67, 13)
    square_length = 0.030
    marker_length = 0.025
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    charboard = cv2.aruco.CharucoBoard(squares, square_length, marker_length, aruco_dict)
    detector = cv2.aruco.CharucoDetector(charboard)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    imageSize = (width, height)

    charucoCorners, charucoIds, markerCorners, markersIds = detector.detectBoard(gray)   
    # image = cv2.aruco.drawDetectedMarkers(image, markerCorners, markersIds)

    return image, charucoCorners, charucoIds
    
def calculatecameramatrix():
    global allCharucoCorners, allCharucoIds, allObjectPoints, allImagePoints, imageSize, calibration_image

    if not allObjectPoints or not allImagePoints:
        print("Insufficient data for calibration.")
        return None

    calibration_flags = 0  # You can adjust flags here if necessary

    # The cameraMatrix and distCoeffs will be initialized internally by the calibrateCamera function
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        allObjectPoints, allImagePoints, imageSize, None, None, flags=calibration_flags
    )

    if ret:
        calibration_data = {
            'camera_matrix': cameraMatrix.tolist(),
            'distortion_coefficients': distCoeffs.tolist(),
            'rotation_vectors': [r.tolist() for r in rvecs],
            'translation_vectors': [t.tolist() for t in tvecs]
        }
        # print(calibration_data)

        # Clear the lists and variables after successful calibration
        allCharucoCorners = []
        allCharucoIds = []
        allObjectPoints = []
        allImagePoints = []
        imageSize = None
        calibration_image = 0

        return calibration_data
    else:
        print("Calibration failed.")
        return None


def calculatecameramatrix_scaledImage():
    global allCharucoCorners_scaledImage, allCharucoIds_scaledImage, allObjectPoints_scaledImage, allImagePoints_scaledImage, imageSize_scaledImage, calibration_image_scaledImage

    if not allObjectPoints_scaledImage or not allImagePoints_scaledImage:
        print("Insufficient data for calibration.")
        return None

    calibration_flags = 0  # You can adjust flags here if necessary

    # The cameraMatrix and distCoeffs will be initialized internally by the calibrateCamera function
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        allObjectPoints_scaledImage, allImagePoints_scaledImage, imageSize_scaledImage, None, None, flags=calibration_flags
    )

    if ret:
        calibration_data = {
            'camera_matrix': cameraMatrix.tolist(),
            'distortion_coefficients': distCoeffs.tolist(),
            'rotation_vectors': [r.tolist() for r in rvecs],
            'translation_vectors': [t.tolist() for t in tvecs]
        }
        # print(calibration_data)

        # Clear the lists and variables after successful calibration
        allCharucoCorners_scaledImage = []
        allCharucoIds_scaledImage = []
        allObjectPoints_scaledImage = []
        allImagePoints_scaledImage = []
        imageSize_scaledImage = None
        calibration_image_scaledImage = 0

        return calibration_data
    else:
        print("Calibration failed.")
        return None



def calculateHomography(img1, img2):
    _, charucoCorners1, charucoIds1 = detectCharucoBoard_6_6(img1)
    _, charucoCorners2, charucoIds2 = detectCharucoBoard_6_6(img2)
    print("calculating homography")

    if charucoCorners1 is not None and charucoCorners2 is not None:

        # Flatten the ID arrays for easier comparison
        charucoIds1 = charucoIds1.flatten()
        charucoIds2 = charucoIds2.flatten()
        
        # Find common IDs in both sets
        commonIds = np.intersect1d(charucoIds1, charucoIds2)
        
        # Filter corners based on common IDs
        corners1 = []
        corners2 = []

        for commonId in commonIds:
            idIndex1 = np.where(charucoIds1 == commonId)[0][0]
            idIndex2 = np.where(charucoIds2 == commonId)[0][0]
            corners1.append(charucoCorners1[idIndex1])
            corners2.append(charucoCorners2[idIndex2])
        
        if corners1 and corners2:
            # Convert lists to numpy arrays and reshape for findHomography
            corners1 = np.array(corners1).reshape(-1, 2)
            corners2 = np.array(corners2).reshape(-1, 2)

            # Calculate the homography matrix
            M, mask = cv2.findHomography(corners2, corners1, cv2.RANSAC, 5.0)
            # print(M)
        

        results = warpTwoImages(img1, img2, M)

        return results, M
    
def calculateHomography_template(img1, img2):
    #init corner and id as empty
    charucoCorners1 = None
    charucoCorners2 = None
    charucoIds1 = None
    charucoIds2 = None

    _, charucoCorners1, charucoIds1 = detectCharucoBoard_aprilTag(img1)
    _, charucoCorners2, charucoIds2 = detectCharucoBoard_aprilTag(img2)
    print("Calculating homography...\n")
    # #print image size
    # print(f"Image 1 size: {img1.shape}")
    # print(f"Image 2 size: {img2.shape}")

    # print (f"charucoCorners1: {charucoCorners1}")
    # print (f"charucoCorners2: {charucoCorners2}")
    # print (f"charucoIds1: {charucoIds1}")
    # print (f"charucoIds2: {charucoIds2}")

    if charucoCorners1 is not None and charucoCorners2 is not None:

        # Flatten the ID arrays for easier comparison
        charucoIds1 = charucoIds1.flatten()
        charucoIds2 = charucoIds2.flatten()
        
        # Find common IDs in both sets
        commonIds = np.intersect1d(charucoIds1, charucoIds2)
        
        # Filter corners based on common IDs
        corners1 = []
        corners2 = []

        for commonId in commonIds:
            idIndex1 = np.where(charucoIds1 == commonId)[0][0]
            idIndex2 = np.where(charucoIds2 == commonId)[0][0]
            corners1.append(charucoCorners1[idIndex1])
            corners2.append(charucoCorners2[idIndex2])
        
        if corners1 and corners2:
            # Convert lists to numpy arrays and reshape for findHomography
            corners1 = np.array(corners1).reshape(-1, 2)
            corners2 = np.array(corners2).reshape(-1, 2)

            # Calculate the homography matrix
            M, mask = cv2.findHomography(corners2, corners1, cv2.RANSAC, 5.0)
            # print(M)
        

        # results = warpTwoImages(img1, img2, M)
        results = None


        return results, M


def warpTwoImages(img1, img2, H):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)

    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)

    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
    # print(xmin, ymin, xmax, ymax)

    result = cv2.warpPerspective(img1, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img2

    return result

def warpTwoImages_template(img1, img2, H): #image1 is template
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Warp img2 to the perspective of img1
    img2_warped = cv2.warpPerspective(img2, H, (w1, h1))

    # Create a mask for img2 to handle alpha blending
    mask2 = np.ones_like(img2_warped, dtype=np.uint8)

    # Warp the mask to the perspective of img1
    mask2_warped = cv2.warpPerspective(mask2, H, (w1, h1))

    # Inserting img2 into img1
    result = img1.copy()
    np.putmask(result, mask2_warped, img2_warped)

    return result