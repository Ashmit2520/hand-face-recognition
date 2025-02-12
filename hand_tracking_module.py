#Author: Rhushil Vasavada
#Hand-tracking MediaPipe Module
#Description: This program models a custom hand detection class to detect 21 hand landmarks as a building block for
#future programs that use this to solve more complex tasks, such as finger counting.

#import libraries
import mediapipe as mp
import cv2
import time
import random

#define class to be used to create a hand detector
class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        #define variables to keep track of custom parameters such as max hands allowed to be detected and detection 
        #confidence (how confident the model is labeling a specific hand landmark)
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon,
                                        self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        #use MediaPipe's built-in library library to detect the landmarks in a given frame image
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []
        #find the positions of landmarks on a given detected hand
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                lmList.append([id, cx, cy])
                randomR = random.randint(0, 255)
                randomG = random.randint(0, 255)
                randomB = random.randint(0, 255)
                if draw:
                    cv2.circle(img, (cx, cy), 10, (245, 135, 66), cv2.FILLED)
        return lmList


def main():
    #configure webcam and run this hand detection class on live camera footage
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        #display output with landmarks drawn over screen
        img = cv2.flip(img,1)
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
