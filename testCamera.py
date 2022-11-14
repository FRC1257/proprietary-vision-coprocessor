from cscore import CameraServer
import cv2
import numpy as np

CameraServer.enableLogging()

with open('/boot/frc.json') as f:
    config = json.load(f)
camera = config['cameras'][0]

width = camera['width']
height = camera['height']

camera = CameraServer.startAutomaticCapture()
camera.setResolution(width, height)

sink = cs.getVideo()

while True:
    time, input_img = cvSink.grabFrame(input_img)
    if time == 0: # There is an error
        continue