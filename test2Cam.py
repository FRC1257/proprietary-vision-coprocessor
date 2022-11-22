# https://robotpy.readthedocs.io/en/stable/vision/code.html
# Import the camera server
from cscore import CameraServer

# Import OpenCV and NumPy
import cv2
import numpy as np

# constants
MIN_HUE = 60
MIN_SAT = 70
MIN_VAL = 70
MAX_HUE = 85
MAX_SAT = 255
MAX_VAL = 255

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
        input_img = np.copy(img)

        cv2.putText(img, "hello", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, 2, cv2.LINE_AA)
        
        if mytime == 0:
            # Send the output the error.
            outputStream.notifyError(cvSink.getError())
            # skip the rest of the current iteration
            continue

        #
        # Insert your image processing logic here!
        #
        hsv_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)

        _, contour_list, _ = cv2.findContours(output_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        if len(contour_list) > 0:
            largest = contour_list[0]
            for contour in contour_list:
                if cv2.contourArea(contour) > cv2.contourArea(largest):
                    largest = contour

        rect = cv2.minAreaRect(largest)
        center, _, _ = rect
        center_x, center_y = center

        corners = cv2.convexHull(contour)
        corners = cv2.approxPolyDP(corners, 0.1 * cv2.arcLength(contour), True)
        
        # binary_img = cv2.inRange(hsv_img, (MIN_HUE, MIN_SAT, MIN_VAL), (MAX_HUE, MAX_SAT, MAX_SAT))
        # just for testing, the binary_img will be outputted
        output_img = cv2.inRange(hsv_img, (MIN_HUE, MIN_SAT, MIN_VAL), (MAX_HUE, MAX_SAT, MAX_SAT))

        processing_time = time.time() - start_time
        fps = 1 / processing_time
        print(f"fps: {fps}")
        cv2.putText(output_img, str(round(fps, 1)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        # (optional) send some image back to the dashboard
        outputStream.putFrame(output_img)

main()