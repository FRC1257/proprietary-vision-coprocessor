# https://robotpy.readthedocs.io/en/stable/vision/code.html
# Import the camera server
from cscore import CameraServer
import cv2
from networktables import NetworkTables
import numpy as np
import time

# Constants
WIDTH = 1280
HEIGHT = 1280
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
    camera.setResolution(WIDTH, HEIGHT)

    # Get a CvSink. This will capture images from the camera
    cvSink = cs.getVideo()

    # Setup a CvSource. This will send images back to the Dashboard. 
    outputStream = cs.putVideo("Processed", WIDTH, HEIGHT)

    # Initialize network table
    NetworkTables.initialize()
    vision_nt = NetworkTables.getTable("Vision")
    # Wait for init
    time.sleep(0.5)
    # Test code
    vision_nt.putNumberArray("lower-h", [float(MIN_HUE)])
    vision_nt.putNumberArray("lower-s", [float(MIN_SAT)])
    vision_nt.putNumberArray("lower-v", [float(MIN_VAL)])

    # Allocating new images is very expensive, always try to preallocate
    img = np.zeros(shape=(WIDTH, HEIGHT, 3), dtype=np.uint8)

    while True:
        start_time = time.time()
        
        # Tell the CvSink to grab a frame from the camera and put it
        # in the source image.  If there is an error notify the output.
        mytime, img = cvSink.grabFrame(img)
        input_img = np.copy(img)

        if mytime == 0:
            # Send the output the error.
            outputStream.notifyError(cvSink.getError())
            # Skip the rest of the current iteration
            continue

        # --- Processing logic begin ---
        hsv_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
        
        # Here output_img outputs a binary image
        output_img = cv2.inRange(hsv_img, (MIN_HUE, MIN_SAT, MIN_VAL), (MAX_HUE, MAX_SAT, MAX_SAT))

        """
        # comment out this docstring and remove the above output_img when testing runPipeline()

        largestContour, output_img, x_list, y_list = runPipeline(input_img, lower, upper)

        # maybe console log what largestContour, x_list, and y_list output
        print(f"largestContour: {largestContour}")
        print(f"x_list: {x_list}")
        print(f"y_list: {y_list}")
        """

        # --- Processing logic end ---

        # Display FPS
        processing_time = time.time() - start_time
        fps = 1 / processing_time
        print(f"fps: {fps}")
        vision_nt.putNumberArray("tx", fps)
        cv2.putText(output_img, str(round(fps, 1)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        # Sends processed image to separate output stream
        # Shows up as a separate camera on web dashboard
        outputStream.putFrame(output_img)

"""
The following functions are directly copied from basePi.py and probably won't work at first.
The only thing I changed is removing the width and height params and just used the consts
because before width and height were being passed as params twice without being modified
which was redundant. I also made comments about some of the vars are there but are unused.
"""
# runPipeline() is called every frame by Limelight's backend.
def runPipeline(image, lower, upper):
    # convert the input image to the HSV color space
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # convert the hsv to a binary image by removing any pixels
    # that do not fall within the following HSV Min/Max values
    img_threshold = cv2.inRange(img_hsv, lower, upper)

    # find contours in the new binary image
    largestContour, image, x_list, y_list = findContours(image, img_threshold)

    #return the largest contour for the LL crosshair, the modified image, and custom robot data
    
    # change the image type
    # sometimes we want to see the part of the image that is being processed
    """mode isn't defined"""
    mode = 0
    if mode == 1:
        return largestContour, img_threshold, x_list, y_list
    return largestContour, image, x_list, y_list

def findContours(image, img_threshold):
    contours, _ = cv2.findContours(img_threshold,
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largestContour = np.array([[]])

    # initialize an empty array of values to send back to the robot
    """llpython var unused"""
    llpython = [0,0,0,0,0,0,0,0]

    # if contours have been detected, draw them
    if len(contours) > 0:
        cv2.drawContours(image, contours, -1, 255, 2)
        # record the largest contour
        largestContour = max(contours, key=cv2.contourArea)

        # get the unrotated bounding box that surrounds the contour
        x,y,w,h = cv2.boundingRect(largestContour)

        # draw the unrotated bounding box
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)

    x_list, y_list = [], []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:4]:
        # record some custom data to send back to the robot
        rect = cv2.minAreaRect(contour)
        """size and angle vars unused"""
        center, size, angle = rect
        center = tuple(
            [int(dim) for dim in center]
        )
        x_list.append((center[0] - WIDTH / 2) / (WIDTH / 2))
        y_list.append(-(center[1] - HEIGHT / 2) / (HEIGHT / 2))

    return largestContour, image, x_list, y_list

main()
