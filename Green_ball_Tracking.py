import cv2
import numpy as np

kernel = np.ones((5,5),np.uint8)

# Take input from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    #Guassian blur to reduce noise
    frame = cv2.GaussianBlur(frame,(5,5),0)

    #bgr to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #split hsv
    h, s, v = cv2.split(hsv)

    #HSV values for upper and lower green
    greenLower = np.array([29, 86, 6])
    greenUpper = np.array([64, 255, 255])

    # Apply thresholding
    hthresh = cv2.inRange(np.array(h),np.array([29]),np.array([64]))
    sthresh = cv2.inRange(np.array(s),np.array([86]),np.array([255]))
    vthresh = cv2.inRange(np.array(v),np.array([6]),np.array([255]))

    # AND h s and v
    tracking = cv2.bitwise_and(hthresh,cv2.bitwise_and(sthresh,vthresh))

    #Gussian blur again
    dilation = cv2.dilate(tracking,kernel,iterations = 1)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    res = cv2.GaussianBlur(closing,(5,5),0)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(res,cv2.HOUGH_GRADIENT,2,120,param1=120,param2=50,minRadius=10,maxRadius=0)                   
    
    #Draw Circles
    if circles is not None:
            for i in circles[0,:]:
                           # If the ball is far, draw it in green
                           if int(round(i[2])) < 30:
                               cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),5)
                               cv2.circle(frame,(i[0],i[1]),2,(0,255,0),10)
                           # else draw it in red
                           elif int(round(i[2])) > 35:
                               cv2.circle(frame,(i[0],i[1]),i[2],(0,0,255),5)
                               cv2.circle(frame,(i[0],i[1]),2,(0,0,255),10)

            #circles = np.round(circles[0, :]).astype("int")
            #X = circles

            #print the coordinates of the center
            print('x=,y=',i[0],i[1])
 
    #Show the result in frames
    cv2.imshow('HueComp',hthresh)
    cv2.imshow('SatComp',sthresh)
    cv2.imshow('ValComp',vthresh)
    cv2.imshow('res',res)
    cv2.imshow('tracking',frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()
                      
