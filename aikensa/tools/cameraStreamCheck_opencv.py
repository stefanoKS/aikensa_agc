import cv2

def show_camera_stream(camera_index=0):
    # Open the camera stream
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Unable to open camera with index {camera_index}")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to read frame from camera")
            break

        # Display the resulting frame
        cv2.imshow(f'Camera {camera_index}', frame)

        # Press 'q' to exit the loop and close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Show the camera stream for camera index 0
    show_camera_stream(1)
