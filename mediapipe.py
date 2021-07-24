import time
import mediapipe as mp
import time
import cv2

mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils


pTime=0
cTime=0

cap=cv2.VideoCapture(0)
while True:
    ret,img=cap.read()
    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #img_gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    result=hands.process(img_rgb)
    #result=hands.process(img_gray)
    
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
    
    
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    
    cv2.putText(img,str(int(fps)),(18,78),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.imshow('RESULT',img)
    
    key=cv2.waitKey(1)    
    if key==27:
        break
    #print(result.multi_hand_landmarks)
    
    
    
