from cscore import CameraServer
import cv2
import numpy as np
import json

cs = CameraServer.getInstance()
cs.enableLogging()

with open('/boot/frc.json') as f:
    config = json.load(f)
camSetting = config['cameras'][0]

width = camSetting['width']
height = camSetting['height']

camera = cs.startAutomaticCapture()
camera.setResolution(width, height)

sink = cs.getVideo()

while True:
    time, input_img = sink.grabFrame(input_img)
    if time == 0: # There is an error
        continue