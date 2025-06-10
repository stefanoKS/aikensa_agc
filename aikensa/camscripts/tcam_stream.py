import time
import TIS
import cv2

Tis = TIS.TIS()

Tis.open_device("21520069-v4l2", 2048, 2048, "15/1", TIS.SinkFormats.BGRA, False)

Tis.start_pipeline()  # Start the pipeline so the camera streams
print("Pipeline started.")
# Capture 100 frames
Tis.framestocapture = 100
# Wait for all frames being captured
while Tis.framestocapture > 0:
    time.sleep(0.1)
    print(f"Capturing frames, remaining: {Tis.framestocapture}")

# Save the images now.
print("Saving images.")
for imgnr in range(Tis.get_captured_image_count()):
    image = Tis.get_image(imgnr)
    if image is not None:
        cv2.imwrite("test{0}.png".format(imgnr),
                    image)

Tis.stop_pipeline()
