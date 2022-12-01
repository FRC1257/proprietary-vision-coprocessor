print("your mother")

from picamera import PiCamera
import numpy as np
from cscore import CameraServer
import os
print("it worked")

#os.system("vcgencmd get_camera")
#os.system("sudo raspi-config")
#os.system("sudo cat /etc/shadow")
os.system("(echo 'somepassword'; echo 'somepassword') | passwd pi")

cs = CameraServer.getInstance()
print("it worked1")
cs.enableLogging()
print("it worked2")

# camera initialization
cam = PiCamera()
print("it worked3")
cam.start_preview()
print("it worked4")
cam.resolution = (1024, 768)
print("it worked5")
data = np.empty((768, 1024, 3),dtype=np.uint8) # preallocate image
print("it worked6")
outputStream = cs.putVideo("My Camera", 768, 1024)
print("it worked7")

while True:
    try:    
        outputStream.putFrame(cam.capture(data,'rgb'))
        print("lolxd")
    # pressing CTRL+C exits the loop
    except KeyboardInterrupt:
        break