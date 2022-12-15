import mainNETWORK_TABLE
import cv2

video = cv2.VideoCapture("WIN_20221212_16_49_22_Pro.mp4")

while video.isOpened():
    ret, frame = video.read()
    if ret:
        cv2.imshow("Window", frame)
        cv2.waitKey(0)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break
    else:
        break

