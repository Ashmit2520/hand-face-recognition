# Author: Rhushil Vasavada
# Finger Counting
# Description: This program utilizes hand landmarks and linear calculations
# to count the number of fingers a user's hand is showing to the webcam.

# import libraries and configure webcam
import cv2
import time
import mediapipe as mp
import hand_tracking_module as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

# create new hand detector via MediaPipe
detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

# optional: detect face landmarks (only if you want extra features such as head-controlled counting)
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(255, 0,0))

while True:
    # find the landmarks on the hand via the frame being processed
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=True)

    # for each finger, (lmList), we want to check if it is "up" or "down" based on the position of the 
    # finger's pad and the finger's middle crease. If the pad is above the middle crease, the finger is
    # "up" and it is counted. If the pad is below the middle crease, the finger is "down" and it is not
    # counted. We will look at the y-positions (except for the thumb, for which we will check the x-position of 
    # the said landmarks)
    if len(lmList) !=0:
        fingers = []
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # this is the number of finger's that have been calculated as being "up"
        totalFingers = fingers.count(1)

        # orient image and display the number of fingers calculated
        img = cv2.flip(img, 1)
        cv2.putText(img, "Number: "+str(totalFingers), (15, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    else:
        img = cv2.flip(img, 1)

    # output frames per second as well as image with detections displayed
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.imshow("Image", img)
    cv2.waitKey(1)
