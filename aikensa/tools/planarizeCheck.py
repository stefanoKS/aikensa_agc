import cv2
import numpy as np
import sys
import os

# Detect 4 ArUco marker corners and make them into a planar rectangular with a certain aspect ratio
# Layout is id 0 for top left, 1 for top right, 2 for bottom left, 3 for bottom right

multiplier = 5.0
ORIGINAL_IMAGE_HEIGHT = int(218 * multiplier)
ORIGINAL_IMAGE_WIDTH = int(913 * multiplier)

dict_type = cv2.aruco.DICT_5X5_250
aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
desired_plane = np.array([[0, 0], [ORIGINAL_IMAGE_WIDTH, 0], [0, ORIGINAL_IMAGE_HEIGHT], [ORIGINAL_IMAGE_WIDTH, ORIGINAL_IMAGE_HEIGHT]], dtype='float32')

def planarize(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    IMAGE_HEIGHT = ORIGINAL_IMAGE_HEIGHT
    IMAGE_WIDTH = ORIGINAL_IMAGE_WIDTH

    corners, ids, rejected = detector.detectMarkers(gray)
    if corners and ids is not None:
        top_left_corner = None
        top_right_corner = None
        bottom_left_corner = None
        bottom_right_corner = None

        for i, corner in zip(ids, corners):
            marker_id = i[0]
            if marker_id == 0:
                top_left_corner = corner[0][0]
            elif marker_id == 1:
                top_right_corner = corner[0][1]
            elif marker_id == 2:
                bottom_left_corner = corner[0][3]
            elif marker_id == 3:
                bottom_right_corner = corner[0][2]

        if top_left_corner is not None and top_right_corner is not None and bottom_left_corner is not None and bottom_right_corner is not None:
            ordered_corners = np.array([top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner], dtype='float32')
            transform = cv2.getPerspectiveTransform(ordered_corners, desired_plane)
            image = cv2.warpPerspective(image, transform, (IMAGE_WIDTH, IMAGE_HEIGHT))

            return image, transform
    return image, None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py image.jpeg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(f"File {image_path} does not exist.")
        sys.exit(1)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not open or find the image {image_path}.")
        sys.exit(1)

    processed_image, transform = planarize(image)

    if transform is not None:
        print("Transform Matrix:")
        print(transform)
        output_path = os.path.splitext(image_path)[0] + "_processed.jpg"
        cv2.imwrite(output_path, processed_image)
        print(f"Processed image saved as {output_path}")
    else:
        print("Transform Matrix not found.")
