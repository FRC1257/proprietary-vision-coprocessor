from cscore import CameraServer
from networktables import NetworkTables

import cv2
import json
import numpy as np
import time

lower = np.array((60, 70, 70))
upper = np.array((85, 255, 255))

mode = 0
# mode 0: normal
# mode 1: threshold

networkInitialized = False

def findContours(image, img_threshold, width, height):
    contours, _ = cv2.findContours(img_threshold,
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largestContour = np.array([[]])

    # initialize an empty array of values to send back to the robot
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
        center, size, angle = rect
        center = tuple(
            [int(dim) for dim in center]
        )
        x_list.append((center[0] - width / 2) / (width / 2))
        y_list.append(-(center[1] - height / 2) / (height / 2))

    return largestContour, image, x_list, y_list


# https://robotpy.readthedocs.io/projects/pynetworktables/en/stable/examples.html#pynetworktables-examples
def valueChanged(table, key, value, isNew):
    print("valueChanged: key: '%s'; value: %s; isNew: %s" % (key, value, isNew))
    global lower, upper
    if key == "lower_h":
        lower[0] = value
    elif key == "lower_s":
        lower[1] = value
    elif key == "lower_v":
        lower[2] = value
    elif key == "upper_h":
        upper[0] = value
    elif key == "upper_s":
        upper[1] = value
    elif key == "upper_v":
        upper[2] = value
    
    if key == "mode":
        global mode
        mode = value

# runPipeline() is called every frame by Limelight's backend.
def runPipeline(image, lower, upper, width, height):
    # convert the input image to the HSV color space
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # convert the hsv to a binary image by removing any pixels
    # that do not fall within the following HSV Min/Max values
    img_threshold = cv2.inRange(img_hsv, lower, upper)

    # find contours in the new binary image
    largestContour, image, x_list, y_list = findContours(image, img_threshold, width, height)

    #return the largest contour for the LL crosshair, the modified image, and custom robot data
    
    # change the image type
    # sometimes we want to see the part of the image that is being processed
    if mode == 1:
        return largestContour, img_threshold, x_list, y_list
    return largestContour, image, x_list, y_list

def main():
    with open("/boot/frc.json") as f:
        config = json.load(f)
    camera = config["cameras"][0]

    width = camera["width"]
    height = camera["height"]

    CameraServer.startAutomaticCapture()

    input_stream = CameraServer.getVideo()
    output_stream = CameraServer.putVideo("Processed", width, height)

    # Table for vision output information
    try:
        vision_nt = NetworkTables.getTable("Vision")
        vision_nt.addEntryListener(valueChanged)
        global networkInitialized
        networkInitialized = True
    except Exception as e:
        print("NetworkTables not initialized")
        print(e)

    # Allocating new images is very expensive, always try to preallocate
    img = np.zeros(shape=(240, 320, 3), dtype=np.uint8)

    # Wait for NetworkTables to start
    time.sleep(0.5)

    while True:
        start_time = time.time()

        frame_time, input_img = input_stream.grabFrame(img)
        output_img = np.copy(input_img)

        # process the image
        height, width = img.shape[:2]
        largestContour, output_img, x_list, y_list = runPipeline(output_img, lower, upper, width, height)

        # send the data to the robot
        if networkInitialized:
            vision_nt.putNumberArray("target_x", x_list)
            vision_nt.putNumberArray("target_y", y_list)

        # calculate the processing time and fps
        processing_time = time.time() - start_time
        fps = 1 / processing_time

        # write it on the image
        cv2.putText(
            output_img,
            str(round(fps, 1)),
            (0, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
        )
        # display the image
        output_stream.putFrame(output_img)

main()