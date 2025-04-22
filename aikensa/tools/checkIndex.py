import cv2

def check_camera_indexes(up_to_index=10):
    for i in range(up_to_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera found at index {i}")
            cap.release()
        else:
            print(f"No camera found at index {i}")

# Adjust the range if you suspect more cameras could be connected
check_camera_indexes(10)
