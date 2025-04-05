# Author: Rhushil Vasavada
# Facial Mesh Reconstruction
# Description: This program uses MediaPipe to recognize 468 landmarks 
# on a human face given through webcam feedback

# import libraries and configure webcam
import mediapipe as mp
import cv2
import time

cap = cv2.VideoCapture(0)

pTime = 0

# configure MediaPipe models, namely face mesh
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=5)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2, color=(255, 255,255))

while True:
    # read frame and run MediaPipe face mesh model on it
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    
    if results.multi_face_landmarks:
        # draw each landmark that has been detected
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
        
        for id, lm in enumerate(faceLms.landmark):
            ih, iw, ic = img.shape
            x,y = int(lm.x*iw), int(lm.y*ih)
            if id == 10:
                cv2.circle(img, (x,y), 20, (0, 10, 240), 20)
                cv2.circle(img, (x,y), 5, (255, 255, 255), 5)

    # reflect the output frame to ensure we are seeing the user's face in correct orientation
    img = cv2.flip(img, 1)

    # output frames per second as well as image with detections displayed
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)

    cv2.waitKey(1)
