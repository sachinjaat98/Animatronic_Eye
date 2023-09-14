from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
from serial import Serial

##########################################################
ard = Serial('COM4',9600)
#ard.open()

######################################################
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())
font = cv2.FONT_HERSHEY_SIMPLEX
#######################################################################
# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
greenLower = (70, 100, 50)
greenUpper = (100, 255, 255)
# list of tracked points
pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
	cap = VideoStream(src=0).start()
# otherwise, grab a reference to the video file
else:
	cap = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

################################################################
def mapObjectPosition(x, y):
    print("[INFO] Object Center coordinates at X0 = {0} and Y0 =  {1}".format(x, y))


def servomotor(x, y):  # Arduino function
    if x > 360:
        ard.write('L'.encode())
        time.sleep(0.01)
    elif x < 230:
        ard.write('R'.encode())
        time.sleep(0.01)
    else:
        ard.write('S'.encode())
        time.sleep(0.01)
    if y > 320:
        ard.write('D'.encode())
        time.sleep(0.01)
    elif y < 150:
        ard.write('U'.encode())
        time.sleep(0.01)
    else:
        ard.write('S'.encode())
        time.sleep(0.01)

################################
servoPosition = 90
servoPosition1 = 90
servoOrientation = 0

####################################################
while True:
    try:
        frame = cap.read()
        # handle the frame from VideoCapture or VideoStream
        frame = frame[1] if args.get("video", False) else frame

        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if frame is None:
            break

        frame = imutils.resize(frame, width=600)
        frame = cv2.flip(frame,1)
        #frame = imutils.rotate(frame,angle = 360)
        blurred = cv2.GaussianBlur(frame, (5,5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small blobs left in the mask
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2);
        mask = cv2.dilate(mask, None, iterations=2);
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);
        cnts = imutils.grab_contours(cnts);
        center = None

        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)


        else:
            ard.write('F'.encode())
            # time.sleep(0.01)
            # print("f")
            if (servoOrientation == 0):
                if (servoPosition >= 90):
                    servoOrientation = 1
                else:
                    servoOrientation = -1
            if (servoOrientation == 1):
                ard.write('L'.encode())
                time.sleep(0.01)
                servoPosition += 1
                if (servoPosition > 140):
                    servoPosition = 140
                    ard.write('U'.encode())
                    time.sleep(0.01)
                    servoPosition1 += 1
                    if (servoPosition1 > 80):
                        servoPosition1 = 80
                        servoOrientation = -1
            else:
                ard.write('R'.encode())
                time.sleep(0.01)
                servoPosition -= 1
                if (servoPosition < 70):
                    servoPosition = 70
                    ard.write('D'.encode())
                    time.sleep(0.01)
                    servoPosition1 -= 1
                    if (servoPosition1 < 60):
                        servoPosition = 60
                        servoOrientation = 1
        ###########################################################################
        # update the points queue
        pts.appendleft(center)

        # loop over the set of tracked points
        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue
            # otherwise, compute the thickness of the line and draw the connecting lines
            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)


        ########################################################
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        #if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

    except:
        pass

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	cap.stop()
# otherwise, release the camera
else:
	cap.release()
# close all windows
cv2.destroyAllWindows()