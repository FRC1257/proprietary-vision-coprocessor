#!/usr/bin/env python3

# Copyright (c) FIRST and other WPILib contributors.
# Open Source Software; you can modify and/or share it under the terms of
# the WPILib BSD license file in the root directory of this project.

print("Start")
print("Importing")
import json, numpy as np, cv2
import time
import sys

from cscore import CameraServer, VideoSource, UsbCamera, MjpegServer
from networktables import NetworkTablesInstance

print("Done")

lower = np.array((60, 70, 70))
upper = np.array((85, 255, 255))
mode = 1

#   JSON format:
#   {
#       "team": <team number>,
#       "ntmode": <"client" or "server", "client" if unspecified>
#       "cameras": [
#           {
#               "name": <camera name>
#               "path": <path, e.g. "/dev/video0">
#               "pixel format": <"MJPEG", "YUYV", etc>   // optional
#               "width": <video mode width>              // optional
#               "height": <video mode height>            // optional
#               "fps": <video mode fps>                  // optional
#               "brightness": <percentage brightness>    // optional
#               "white balance": <"auto", "hold", value> // optional
#               "exposure": <"auto", "hold", value>      // optional
#               "properties": [                          // optional
#                   {
#                       "name": <property name>
#                       "value": <property value>
#                   }
#               ],
#               "stream": {                              // optional
#                   "properties": [
#                       {
#                           "name": <stream property name>
#                           "value": <stream property value>
#                       }
#                   ]
#               }
#           }
#       ]
#       "switched cameras": [
#           {
#               "name": <virtual camera name>
#               "key": <network table key used for selection>
#               // if NT value is a string, it's treated as a name
#               // if NT value is a double, it's treated as an integer index
#           }
#       ]
#   }

configFile = "/boot/frc.json"

class CameraConfig: pass

team = 1257
server = False
cameraConfigs = []
# switchedCameraConfigs = []
cameras = []

def parseError(str):
    """Report parse error."""
    print("config error in '" + configFile + "': " + str, file=sys.stderr)

def readCameraConfig(config):
    """Read single camera configuration."""
    cam = CameraConfig()

    # name
    try:
        cam.name = config["name"]
    except KeyError:
        parseError("could not read camera name")
        return False

    # path
    try:
        cam.path = config["path"]
    except KeyError:
        parseError("camera '{}': could not read path".format(cam.name))
        return False

    # stream properties
    cam.streamConfig = config.get("stream")

    cam.config = config

    cameraConfigs.append(cam)
    return True

""" def readSwitchedCameraConfig(config):
    # Read single switched camera configuration.
    cam = CameraConfig()

    # name
    try:
        cam.name = config["name"]
    except KeyError:
        parseError("could not read switched camera name")
        return False

    # path
    try:
        cam.key = config["key"]
    except KeyError:
        parseError("switched camera '{}': could not read key".format(cam.name))
        return False

    switchedCameraConfigs.append(cam)
    return True """

def readConfig():
    """Read configuration file."""
    global team
    global server

    # parse file
    try:
        with open(configFile, "rt", encoding="utf-8") as f:
            j = json.load(f)
    except OSError as err:
        print("could not open '{}': {}".format(configFile, err), file=sys.stderr)
        return False

    # top level must be an object
    if not isinstance(j, dict):
        parseError("must be JSON object")
        return False

    # team number
    try:
        team = j["team"]
    except KeyError:
        parseError("could not read team number")
        return False

    # ntmode (optional)
    if "ntmode" in j:
        str = j["ntmode"]
        if str.lower() == "client":
            server = False
        elif str.lower() == "server":
            server = True
        else:
            parseError("could not understand ntmode value '{}'".format(str))

    # cameras
    try:
        cameras = j["cameras"]
    except KeyError:
        parseError("could not read cameras")
        return False
    for camera in cameras:
        if not readCameraConfig(camera):
            return False

    # switched cameras
    """ if "switched cameras" in j:
        for camera in j["switched cameras"]:
            if not readSwitchedCameraConfig(camera):
                return False """

    return True

output_streams = []

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
    if key == "lower-h":
        lower[0] = value
    elif key == "lower-s":
        lower[1] = value
    elif key == "lower-v":
        lower[2] = value
    elif key == "upper-h":
        upper[0] = value
    elif key == "upper-s":
        upper[1] = value
    elif key == "upper-v":
        upper[2] = value
    
    if key == "mode":
        global mode
        mode = int(value)

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

def startCamera(config):
    """Start running the camera."""
    print("Starting camera '{}' on {}".format(config.name, config.path))
    inst = CameraServer.getInstance()
    camera = UsbCamera(config.name, config.path)
    server = inst.startAutomaticCapture(camera=camera, return_server=True)

    camera.setConfigJson(json.dumps(config.config))
    camera.setConnectionStrategy(VideoSource.ConnectionStrategy.kKeepOpen)

    # input_stream = inst.getVideo()
    # print(dir(config))
    # output_stream = inst.putVideo('Processed ' + config.name, 640, 640)
    # output_streams.append(output_stream)
    output_streams.append(inst)

    if config.streamConfig is not None:
        server.setConfigJson(json.dumps(config.streamConfig))

    return camera

""" def startSwitchedCamera(config):
    # Start running the switched camera.
    print("Starting switched camera '{}' on {}".format(config.name, config.key))
    server = CameraServer.getInstance().addSwitchedCamera(config.name)

    def listener(fromobj, key, value, isNew):
        if isinstance(value, float):
            i = int(value)
            if i >= 0 and i < len(cameras):
              server.setSource(cameras[i])
        elif isinstance(value, str):
            for i in range(len(cameraConfigs)):
                if value == cameraConfigs[i].name:
                    server.setSource(cameras[i])
                    break
    
    print(config.key)
    NetworkTablesInstance.getDefault().getEntry(config.key).addListener(
        listener,
        NetworkTablesInstance.NotifyFlags.IMMEDIATE |
        NetworkTablesInstance.NotifyFlags.NEW |
        NetworkTablesInstance.NotifyFlags.UPDATE
    )

    return server """

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        configFile = sys.argv[1]

    # read configuration
    if not readConfig():
        sys.exit(1)

    # start NetworkTables
    ntinst = NetworkTablesInstance.getDefault()
    if server:
        print("Setting up NetworkTables server")
        ntinst.startServer()
    else:
        print("Setting up NetworkTables client for team {}".format(team))
        ntinst.startClientTeam(team)
        ntinst.startDSClient()

    # start cameras
    for config in cameraConfigs:
        cameras.append(startCamera(config))

    # start switched cameras
    """ for config in switchedCameraConfigs:
        startSwitchedCamera(config) """
    input_stream = output_streams[0].getVideo()
    try:
        vision_nt = ntinst.getTable("Vision")
        vision_nt.addEntryListener(valueChanged)
    except Exception as e:
        print("NetworkTables not initialized")
        print(e)

    output_stream = output_streams[0].putVideo("Processed", 1280, 1280)

    # loop forever
    
    # loop to start
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
