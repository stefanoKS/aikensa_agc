import sys
import cv2
import numpy as np
sys.path.append("../python-common")

import TIS

# This sample shows, how to get an image and convert it to OpenCV
# needed packages:
# pyhton-opencv
# pyhton-gst-1.0
# tiscamera

Tis = TIS.TIS()
#Tis.openDevice("10710286", 640, 480, "30/1", TIS.SinkFormats.BGRA,True)
# the camera with serial number 10710286 uses a 640x480 video format at 30 fps and the image is converted to
# RGBx, which is similar to RGB32.

# The next line is for selecting a device, video format and frame rate.
if not Tis.select_device():
    quit(0)

# Just in case trigger mode is enabled, disable it.
try:
    Tis.set_property("TriggerMode","Off")
except Exception as error:
    print(error)


Tis.start_pipeline()  # Start the pipeline so the camera streams

kernel = np.ones((5, 5), np.uint8)  # Create a Kernel for OpenCV erode function

if Tis.snap_image(1):  # Snap an image with one second timeout
    image = Tis.get_image()  # Get the image. It is a numpy array
    image = cv2.erode(image, kernel, iterations=5)  # Example OpenCV image processing
    cv2.imwrite('test_image.png', image)  # Save the processed image


# Stop the pipeline and clean up
Tis.stop_pipeline()
print('Program ends')
