# https://robotpy.readthedocs.io/en/stable/vision/code.html
# Import the camera server
from cscore import CameraServer

# Import OpenCV and NumPy
import cv2
import numpy as np

def main():
    cs = CameraServer.getInstance()
    cs.enableLogging()

    # Capture from the first USB Camera on the system
    camera = cs.startAutomaticCapture()
    camera.setResolution(1280, 1280)

    # Get a CvSink. This will capture images from the camera
    cvSink = cs.getVideo()

    # (optional) Setup a CvSource. This will send images back to the Dashboard
    outputStream = cs.putVideo("My Camera", 1280, 1280)

    # Allocating new images is very expensive, always try to preallocate
    img = np.zeros(shape=(1280, 1280, 3), dtype=np.uint8)

    while True:
        # Tell the CvSink to grab a frame from the camera and put it
        # in the source image.  If there is an error notify the output.
        time, img = cvSink.grabFrame(img)

        cv2.putText(img, "hello", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, 2, cv2.LINE_AA)
        
        if time == 0:
            # Send the output the error.
            outputStream.notifyError(cvSink.getError())
            # skip the rest of the current iteration
            continue

        #
        # Insert your image processing logic here!
        #

        # (optional) send some image back to the dashboard
        outputStream.putFrame(img)

main()