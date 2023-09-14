
from __future__ import print_function
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os
import RPi.GPIO as GPIO

# define Servos GPIOs
baseServo = 23  #lower servo
tiltServo = 17  # upper servo

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)


# position servos
def positionServo (servo, angle):

    GPIO.setup(servo, GPIO.OUT)
    pwm = GPIO.PWM(servo, 50)  # SERVO AND frequency
    pwm.start(6)
    dutyCycle = angle / 18. + 2.
    pwm.ChangeDutyCycle(dutyCycle)
    time.sleep(0.1)
    pwm.stop()



def servo_position(x,y,center):
    global baseAngle
    global  tiltAngle

    if x < center-30:
        baseAngle += 3
        if baseAngle > 140:
            baseAngle = 140
        positionServo(baseServo, baseAngle)

    elif x > center +30 :
        baseAngle -= 3
        if baseAngle < 40:
            baseAngle = 40
        positionServo(baseServo, baseAngle)
    if y < center-30:
        tiltAngle += 3
        if tiltAngle > 140:
            tiltAngle = 140
        positionServo(tiltServo, tiltAngle)

    elif y > center +30 :
        tiltAngle -= 3
        if tiltAngle < 40:
            tiltAngle = 40
        positionServo(tiltServo, tiltAngle)


# initialize the video stream and allow the camera sensor to warmup
print("[INFO] waiting for camera to warmup...")
cap = VideoStream(0).start()
time.sleep(2.0)

# define the lower and upper boundaries of the object
# to be tracked in the HSV color space
colorLower = (24, 100, 100)
colorUpper = (44, 255, 255)

# Start with LED off
#GPIO.output(redLed, GPIO.LOW)
#ledOn = False

# Initialize angle servos at 90-90 position
global baseServoAngle
baseAngle = 90
global tiltAngle
tiltAngle =90

# positioning Pan/Tilt servos at initial position
positionServo(baseServo, baseAngle)
positionServo(tiltServo, tiltAngle)

classNames = []
classesFile = 'coco.names'

#initilizing the trained model  files path
with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

print(classNames)
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# loop over the frames from the video stream
while True:
    frame = cap.read()
    frame2 = frame.copy()
    frame = imutils.resize(frame, width=500)
    frame = imutils.rotate(frame, angle=180)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, colorLower, colorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)


    classIds, confs, bbox = net.detect(frame, confThreshold=0.5)
    # print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(frame2, box, color=(0, 255, 0), thickness=2)
            cv2.putText(frame2, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 0, 0), 2)
            cv2.putText(frame2, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 2)


    frame_center_x = frame.shape[1]/2
    frame_center_y = frame.shape[0]/2

    # find contours in the mask and initialize the current
    # (x, y) center of the object
    cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)


            servo_position(int(x),int(y),frame_center_x,frame_center_y)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    cv2.imshow("recognition",frame2)

    # if [ESC] key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == 13:
        break


# do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff \n")
positionServo(baseServo, 90)
positionServo(tiltServo, 90)
GPIO.cleanup()
cv2.destroyAllWindows()
cap.stop()