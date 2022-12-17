import mainNETWORK_TABLE as test
import cv2, numpy as np

lower = np.array((11, 92, 124))
upper = np.array((41, 255, 255))

video = cv2.VideoCapture("WIN_20221212_16_49_22_Pro.mp4")

while video.isOpened():
    ret, frame = video.read()
    if ret:
        cv2.imshow("Window", frame)
        
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # convert the hsv to a binary image by removing any pixels
        # that do not fall within the following HSV Min/Max values
        mask = cv2.inRange(img_hsv, lower, upper)
        new_image = cv2.bitwise_and(frame, frame, mask=mask)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 80, 100, L2gradient=True)
        cv2.imshow("edge", edge)

        # closer to 1, the better circle it looks for
        circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT_ALT,dp=1.5,minDist=20,
                                    param1=300,param2=0.75,minRadius=50,maxRadius=300)
        print(f"circle: {circles}")
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
        cv2.imshow("edge2", frame)
        largestContour, image, x_list, y_list, area, confidence, angle = test.findBalls(frame, mask, 1, 1)
        cv2.imshow("Circle", image)
        cv2.imshow("Mask", mask)
        cv2.imshow("New", new_image)
        cv2.waitKey(0)
        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break
    else:
        break

