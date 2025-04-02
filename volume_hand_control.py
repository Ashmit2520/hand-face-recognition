#Author: Rhushil Vasavada
#Volume Control With Hand Gestures
#Description: This program manipulates a computer's volume bar (macOS) 
#based on a user's hand gestures using OpenCV library

#import libraries and configure webcam
import cv2
import time
import numpy as np
import hand_tracking_module as htm
import math
import osascript

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

#create a new hand detector to be used to recognize user's hand
detector = htm.handDetector(detectionCon=0.7)

while True:
    #read webcam frame and run hand detector on the frame
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        #if we have found hand landmarks, we need to isolate the specific 
        #landmarks we are interested in, namely: the pad of the pointer finger
        #and the pad of the thumb. The distance between these two landmarks
        #will essentially be the controller of the volume
        
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        #calculate distance between said landmarks
        length = math.hypot(x2-x1,y2-y1)
        if length<40:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
        lengthModified = length/255
        lengthModified = lengthModified*100

        if lengthModified>100:
            lengthModified=100
        if lengthModified<10:
            lengthModified=0

        volBar = np.interp(length, [20, 240], [400, 150])

        #depending on the distance of the landmarks, we want to create a screen "volume bar"
        #to represent how much the volume is changing; depending on the distance, we have 
        #different colors and percentages of the rectangle are filled 
        if volBar<275 and volBar>150:
            cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 4)
            cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
        elif volBar>=275 and volBar<380:
            cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 0, 255), cv2.FILLED)
            cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 4)
        elif volBar==150:
            cv2.rectangle(img, (50, int(volBar)), (85, 400), (238, 130, 238), cv2.FILLED)
            cv2.rectangle(img, (50, 150), (85, 400), (238, 130, 238), 4)
        elif volBar>=380:
            cv2.rectangle(img, (50, int(volBar)), (85, 400), (130, 0, 75), cv2.FILLED)
            cv2.rectangle(img, (50, 150), (85, 400), (130, 0, 75), 4)

        #standardized the length between landmarks to a small, whole number 
        lengthModified = round(lengthModified/10)*10
        vol = "set volume output volume " + str(lengthModified)
        #depending on this number, change the macOS volume
        osascript.osascript(vol)

    #output frames per second as well as image with detections displayed
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    img = cv2.flip(img, 1)

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
