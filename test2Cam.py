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

    # (optional) Setup a CvSource. This will send images back to the Dashboard. 
    outputStream = cs.putVideo("My Camera", 1280, 1280)

    # Allocating new images is very expensive, always try to preallocate
    img = np.zeros(shape=(1280, 1280, 3), dtype=np.uint8)

    while True:
        start_time = mytime.time()
        
        # Tell the CvSink to grab a frame from the camera and put it
        # in the source image.  If there is an error notify the output.
        mytime, img = cvSink.grabFrame(img)
        output_img = np.copy(img)

        cv2.putText(img, "hello", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, 2, cv2.LINE_AA)
        
        if mytime == 0:
            # Send the output the error.
            outputStream.notifyError(cvSink.getError())
            # skip the rest of the current iteration
            continue

        #
        # Insert your image processing logic here!
        #
        # hsv_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
        # binary_img = cv2.inRange(hsv_img, (min_hue, min_sat, min_val), (max_hue, max_sat, max_val))

        processing_time = time.time() - start_time
        fps = 1 / processing_time
        print(f"fps: {fps}")
        cv2.putText(output_img, str(round(fps, 1)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        # (optional) send some image back to the dashboard
        outputStream.putFrame(output_img)

main()