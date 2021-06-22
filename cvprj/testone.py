import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import winsound
import wave


cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
count = 0

folderPath = "asdf"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
arr=np.zeros(15)
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}',cv2.IMREAD_COLOR)
    image = cv2.resize(image,dsize = (480,640))
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlayList.append(image)
    results = hands.process(imgRGB)
#print(results.multi_hand_landmarks)
    a=np.zeros((6,2))
    b=np.zeros(15)
    for handLms in results.multi_hand_landmarks:
        for id, lm in enumerate(handLms.landmark):
            if (id==0):
                a[0][0]=lm.x
                a[0][1]=lm.y
            if (id==4):
                a[1][0]=lm.x
                a[1][1]=lm.y
            if (id==8):
                a[2][0]=lm.x
                a[2][1]=lm.y
            if (id==12):
                a[3][0]=lm.x
                a[3][1]=lm.y        
            if (id==16):
                a[4][0]=lm.x
                a[4][1]=lm.y    
            if (id==20):
                a[5][0]=lm.x
                a[5][1]=lm.y

    

    #cv2.imshow("Image", image)
    b[0]=math.hypot(a[0][0]-a[1][0],a[0][1]-a[1][1])
    b[1]=math.hypot(a[0][0]-a[2][0],a[0][1]-a[2][1])
    b[2]=math.hypot(a[0][0]-a[3][0],a[0][1]-a[3][1])
    b[3]=math.hypot(a[0][0]-a[4][0],a[0][1]-a[4][1])
    b[4]=math.hypot(a[0][0]-a[5][0],a[0][1]-a[5][1])
    b[5]=math.hypot(a[1][0]-a[2][0],a[1][1]-a[2][1])
    b[6]=math.hypot(a[1][0]-a[3][0],a[1][1]-a[3][1])
    b[7]=math.hypot(a[1][0]-a[4][0],a[1][1]-a[4][1])
    b[8]=math.hypot(a[1][0]-a[5][0],a[1][1]-a[5][1])
    b[9]=math.hypot(a[2][0]-a[3][0],a[2][1]-a[3][1])
    b[10]=math.hypot(a[2][0]-a[4][0],a[2][1]-a[4][1])
    b[11]=math.hypot(a[2][0]-a[5][0],a[2][1]-a[5][1])
    b[12]=math.hypot(a[3][0]-a[4][0],a[3][1]-a[4][1])
    b[13]=math.hypot(a[3][0]-a[5][0],a[3][1]-a[5][1])
    b[14]=math.hypot(a[4][0]-a[5][0],a[4][1]-a[5][1])
    arr=np.vstack((arr,b))


print(arr)
print(arr.shape)
#print(len(overlayList))
pTime = 0


cv2.waitKey(0)
if cap.isOpened(): 
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  
  
    #width  = cap.get(3)  
    #height = cap.get(4)  

    print('width, height:', width, height)
prev_p = 0
c_p = 0    
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    a=np.zeros((6,2))
    b=np.zeros(15)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                if (id==0):
                    a[0][0]=lm.x
                    a[0][1]=lm.y
                if (id==4):
                    a[1][0]=lm.x
                    a[1][1]=lm.y
                if (id==8):
                    a[2][0]=lm.x
                    a[2][1]=lm.y
                if (id==12):
                    a[3][0]=lm.x
                    a[3][1]=lm.y        
                if (id==16):
                    a[4][0]=lm.x
                    a[4][1]=lm.y    
                if (id==20):
                    a[5][0]=lm.x
                    a[5][1]=lm.y
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) 
        b[0]=math.hypot(a[0][0]-a[1][0],a[0][1]-a[1][1])
        b[1]=math.hypot(a[0][0]-a[2][0],a[0][1]-a[2][1])
        b[2]=math.hypot(a[0][0]-a[3][0],a[0][1]-a[3][1])
        b[3]=math.hypot(a[0][0]-a[4][0],a[0][1]-a[4][1])
        b[4]=math.hypot(a[0][0]-a[5][0],a[0][1]-a[5][1])
        b[5]=math.hypot(a[1][0]-a[2][0],a[1][1]-a[2][1])
        b[6]=math.hypot(a[1][0]-a[3][0],a[1][1]-a[3][1])
        b[7]=math.hypot(a[1][0]-a[4][0],a[1][1]-a[4][1])
        b[8]=math.hypot(a[1][0]-a[5][0],a[1][1]-a[5][1])
        b[9]=math.hypot(a[2][0]-a[3][0],a[2][1]-a[3][1])
        b[10]=math.hypot(a[2][0]-a[4][0],a[2][1]-a[4][1])
        b[11]=math.hypot(a[2][0]-a[5][0],a[2][1]-a[5][1])
        b[12]=math.hypot(a[3][0]-a[4][0],a[3][1]-a[4][1])
        b[13]=math.hypot(a[3][0]-a[5][0],a[3][1]-a[5][1])
        b[14]=math.hypot(a[4][0]-a[5][0],a[4][1]-a[5][1])
        min = 65535 
        min_num = 0
        for i in range(arr.shape[0]-1):
            temp = 0
            for u in range(15):
                temp = temp + (arr[i+1][u]-b[u])**2
            if(temp<min):
                min = temp
                min_num=i
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime        
        print(min_num,fps)       
        """c_p = min_num
        if(prev_p==0):
            if(c_p==1):
                winsound.PlaySound('p1.wav',winsound.SND_FILENAME)
                prev_p=1
            elif(c_p==2):
                winsound.PlaySound('p2.wav',winsound.SND_FILENAME)    
                prev_p=2
        else:
            if(c_p==0):
                prev_p=0"""                          
    cv2.imshow("asdf",img)
    cv2.waitKey(1)
