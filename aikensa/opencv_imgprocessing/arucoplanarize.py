import cv2
import numpy as np
import os
import yaml

#detect 4 aruco marker corner and make them into a planar rectangular with certain aspect ratio
#layout is id 0 for topleft, 1 for topright, 2 for bottomleft, 3 for bottomright

multiplier = 5.0
ORIGINAL_IMAGE_HEIGHT = int(218 * multiplier)
ORIGINAL_IMAGE_WIDTH = int(913 * multiplier)

dict_type = cv2.aruco.DICT_5X5_250
aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


def adjust_warp_transform(warp_transform, scale_factor):
    warp_transform[0, 2] /= scale_factor  # Adjust tx
    warp_transform[1, 2] /= scale_factor  # Adjust ty
    return warp_transform


def planarize(image, scale_factor=1.0):
    transform = None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    IMAGE_HEIGHT = int(ORIGINAL_IMAGE_HEIGHT / scale_factor)
    IMAGE_WIDTH = int(ORIGINAL_IMAGE_WIDTH / scale_factor)

    desired_plane = np.array([[0, 0], [ORIGINAL_IMAGE_WIDTH, 0], [0, ORIGINAL_IMAGE_HEIGHT], [ORIGINAL_IMAGE_WIDTH, ORIGINAL_IMAGE_HEIGHT]], dtype='float32')

    if os.path.exists("./aikensa/param/warptransform.yaml") and os.path.exists("./aikensa/param/warptransform_lowres.yaml"):
        if scale_factor == 1.0:
            with open('./aikensa/param/warptransform.yaml', 'r') as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                transform = np.array(transform_list)
                if scale_factor != 1.0:
                    transform = adjust_warp_transform(transform, scale_factor)
            image = cv2.warpPerspective(image, transform, (IMAGE_WIDTH, IMAGE_HEIGHT))
            return image, None
        if scale_factor != 1.0:
            with open('./aikensa/param/warptransform_lowres.yaml', 'r') as file:
                transform_list = yaml.load(file, Loader=yaml.FullLoader)
                transform = np.array(transform_list)
                if scale_factor != 1.0:
                    transform = transform#modify logic to adjust scale factor -> lowres being "fooled" to have scale of 1.0
            image = cv2.warpPerspective(image, transform, (int(IMAGE_WIDTH), int(IMAGE_HEIGHT)))
            #resize image to original size
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            return image, None


    else:
        corners, ids, rejected = detector.detectMarkers(gray)
        # print (corners)
        # print (ids)
        # print (rejected)
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

                if scale_factor != 1.0:
                    desired_plane = np.array([[0, 0], [ORIGINAL_IMAGE_WIDTH//scale_factor, 0], [0, ORIGINAL_IMAGE_HEIGHT//scale_factor], [ORIGINAL_IMAGE_WIDTH//scale_factor, ORIGINAL_IMAGE_HEIGHT//scale_factor]], dtype='float32')
                
                transform = cv2.getPerspectiveTransform(ordered_corners, desired_plane)
                image = cv2.warpPerspective(image, transform, (IMAGE_WIDTH, IMAGE_HEIGHT))

            return image, transform
        else:
            return image, None

def planarize_image(image, target_width, target_height, top_offset, bottom_offset):
    transform = None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    desired_plane = np.array([[0, 0], [target_width, 0], [0, target_height], [target_width, target_height]], dtype='float32')
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
                # Made a mistake here -> when taping, accidentally rotated it by 180 degrees originally [0][1] -> [0][3]
                top_right_corner = corner[0][3]
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

            desired_plane = np.array([[0, top_offset], [target_width, top_offset], [0, target_height-bottom_offset], [target_width, target_height-bottom_offset]], dtype='float32')
            # modified_desired_plane = np.array([[0, 0], [target_width, 0], [0, target_height], [target_width, target_height]], dtype='float32')
            
            transform = cv2.getPerspectiveTransform(ordered_corners, desired_plane)
            image = cv2.warpPerspective(image, transform, (target_width, target_height))
            #Squeezed image height by 1.26 comes from 340mm/270mm (distance of visible height/distance of arucho marker height)
            image = cv2.resize(image, (target_width, int(target_height/1.26)))

        return image, transform
    
    else:
        return image, None


def planarize_image_temp(image, target_width, target_height, top_offset, bottom_offset):
    transform = None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    desired_plane = np.array([[0, 0], [target_width, 0], [0, target_height], [target_width, target_height]], dtype='float32')
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
                # Made a mistake here -> when taping, accidentally rotated it by 180 degrees originally [0][1] -> [0][3]
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

            desired_plane = np.array([[0, top_offset], [target_width, top_offset], [0, target_height-bottom_offset], [target_width, target_height-bottom_offset]], dtype='float32')
            # modified_desired_plane = np.array([[0, 0], [target_width, 0], [0, target_height], [target_width, target_height]], dtype='float32')
            
            transform = cv2.getPerspectiveTransform(ordered_corners, desired_plane)
            image = cv2.warpPerspective(image, transform, (target_width, target_height))
            #Squeezed image height by 1.26 comes from 340mm/270mm (distance of visible height/distance of arucho marker height)
            image = cv2.resize(image, (target_width, int(target_height/1.26)))

        return image, transform
    
    else:
        return image, None

