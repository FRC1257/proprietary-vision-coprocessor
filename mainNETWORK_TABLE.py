#!/usr/bin/env python3

# Copyright (c) FIRST and other WPILib contributors.
# Open Source Software; you can modify and/or share it under the terms of
# the WPILib BSD license file in the root directory of this project.

print("Start")
print("Importing")
import json, numpy as np, cv2
import os
import stat
import time
import sys

from cscore import CameraServer, VideoSource, UsbCamera, MjpegServer
from networktables import NetworkTablesInstance

print("Done")

# Numbers to be changed by network table
lower = np.array((0, 0, 200))
upper = np.array((255, 255, 255))
mode = 0
kernelSize = 3
iterations = 10
minArea = 1000

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

def readConfig():
    """Read configuration file."""
    global team
    global server

    data = {
  "cameras": [
    {
      "brightness": 46,
      "exposure": 4,
      "fps": 30,
      "height": 1280,
      "name": "Tape Detection",
      "path": "/dev/video0",
      "pixel format": "mjpeg",
      "properties": [
        { "name": "connect_verbose", "value": 1 },
        { "name": "contrast", "value": 50 },
        { "name": "saturation", "value": 45 },
        { "name": "power_line_frequency", "value": 1 },
        { "name": "sharpness", "value": 22 },
        { "name": "backlight_compensation", "value": 0 },
        { "name": "pan_absolute", "value": 14400 },
        { "name": "tilt_absolute", "value": -36000 },
        { "name": "zoom_absolute", "value": 0 }
      ],
      "stream": { "properties": [] },
      "white balance": "auto",
      "width": 720
    },
    {
      "brightness": 74,
      "exposure": "auto",
      "fps": 30,
      "height": 1280,
      "name": "Driver",
      "path": "/dev/video2",
      "pixel format": "mjpeg",
      "properties": [
        { "name": "connect_verbose", "value": 1 },
        { "name": "contrast", "value": 50 },
        { "name": "saturation", "value": 45 },
        { "name": "power_line_frequency", "value": 1 },
        { "name": "sharpness", "value": 22 },
        { "name": "backlight_compensation", "value": 0 },
        { "name": "pan_absolute", "value": 0 },
        { "name": "tilt_absolute", "value": 0 },
        { "name": "zoom_absolute", "value": 0 }
      ],
      "stream": { "properties": [] },
      "white balance": "auto",
      "width": 720
    }
  ],
  "ntmode": "client",
  "switched cameras": [],
  "team": 1257
}


    # parse file
    try:
        with open(configFile, "rt", encoding="utf-8") as f:
            j = json.load(f)
            print("Config Json file", j)
    except OSError as err:
        print("could not open '{}': {}".format(configFile, err), file=sys.stderr)
        j = data

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
            # return False
            pass

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
    confidence = 0
    area = 0

    # initialize an empty array of values to send back to the robot

    # if contours have been detected, draw them
    if len(contours) > 0:
        cv2.drawContours(image, contours, -1, 255, 2)
        # record the largest contour
        largestContour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largestContour)

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
    
    if area > minArea:
        confidence = 1

    return largestContour, image, x_list, y_list, area, confidence


# https://robotpy.readthedocs.io/projects/pynetworktables/en/stable/examples.html#pynetworktables-examples
def valueChanged(table, key, value, isNew):
    print("valueChanged: key: '%s'; value: %s; isNew: %s" % (key, value, isNew))
    global lower, upper, iterations, minArea, kernelSize
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

    if key == "iterations":
        iterations = int(value)
    
    if key == "minArea":
        minArea = int(value)
    
    if key == "kernel":
        kernelSize = int(value)
    
    if key == "mode":
        global mode
        mode = int(value)

# runPipeline() is called every frame by camera's backend.
def runPipeline(image, lower, upper, width, height):
    # convert the input image to the HSV color space
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # convert the hsv to a binary image by removing any pixels
    # that do not fall within the following HSV Min/Max values
    mask = cv2.inRange(img_hsv, lower, upper)
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    img_threshold = cv2.erode(mask, kernel, iterations=iterations)

        
    # find contours in the new binary image
    largestContour, image, x_list, y_list, area, confidence = findContours(image, img_threshold, width, height)

    #return the largest contour for the LL crosshair, the modified image, and custom robot data
    
    # change the image type
    # sometimes we want to see the part of the image that is being processed
    if mode == 1:
        return largestContour, img_threshold, x_list, y_list, area, confidence
    return largestContour, image, x_list, y_list, area, confidence

def startCamera(config):
    """Start running the camera."""
    print("Starting camera '{}' on {}".format(config.name, config.path))
    inst = CameraServer.getInstance()
    camera = UsbCamera(config.name, config.path)
    server = inst.startAutomaticCapture(camera=camera, return_server=True)

    camera.setConfigJson(json.dumps(config.config))
    print("Current Configuration", json.dumps(config.config))
    camera.setConnectionStrategy(VideoSource.ConnectionStrategy.kKeepOpen)

    # input_stream = inst.getVideo()
    # print(dir(config))
    # output_stream = inst.putVideo('Processed ' + config.name, 640, 640)
    # output_streams.append(output_stream)
    output_streams.append(inst)

    if config.streamConfig is not None:
        server.setConfigJson(json.dumps(config.streamConfig))

    return camera

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        configFile = sys.argv[1]

    # read configuration
    if not readConfig():
        print("could not read cameras config: " + configFile)
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
    print("Starting cameras")
    for config in cameraConfigs:
        cameras.append(startCamera(config))

    # start switched cameras
    """ for config in switchedCameraConfigs:
        startSwitchedCamera(config) """
    print("Get specific camera")
    input_stream = output_streams[0].getVideo()
    # drive_in_stream = output_streams[1].getVideo()

    print("Starting NetworkTables listener")
    try:
        vision_nt = ntinst.getTable("Vision")
        vision_nt.addEntryListener(valueChanged)
    except Exception as e:
        print("NetworkTables not initialized")
        print(e)

    print("Starting output streams", len(output_streams))
    output_stream = output_streams[0].putVideo("Processed", 1280, 1280)
    # drive_out_stream = output_streams[1].putVideo("Drive", 1280, 1280)

    # loop forever
    
    # loop to start
    # Allocating new images is very expensive, always try to preallocate
    print("Starting loop")
    img = np.zeros(shape=(1280, 720, 3), dtype=np.uint8)

    # Wait for NetworkTables to start
    time.sleep(0.5)

    t = 0

    while True:
        start_time = time.time()

        frame_time, input_img = input_stream.grabFrame(img)
        # _, drive_in = drive_in_stream.grabFrame(img)
        output_img = np.copy(input_img)
        # drive_img = np.copy(drive_in)

        # process the image
        height, width = img.shape[:2]
        largestContour, output_img, x_list, y_list, area, confidence = runPipeline(output_img, lower, upper, width, height)

        # send the data to the robot
        vision_nt.putNumberArray("tx", x_list)
        vision_nt.putNumberArray("ty", y_list)
        vision_nt.putNumber("ta", area)
        vision_nt.putNumber("tv", confidence)

        # calculate the processing time and fps
        processing_time = time.time() - start_time
        fps = 1 / processing_time
        """ if t % 50 == 0:
            print("FPS", round(fps))
            changeMode(); """

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
        # drive_out_stream.putFrame(drive_img)
        t += 1
