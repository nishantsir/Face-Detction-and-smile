# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 13:00:32 2021

@author: user
"""

import numpy as np
import cv2

face_dect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_dect = cv2.CascadeClassifier('haarcascade_smile.xml')
cascade2 = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret,frame = cap.read()
    if ret == True :
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cascade = face_dect.detectMultiScale(frame,1.1,5)
        for(x,y,w,h)in cascade:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            
            smile = smile_dect.detectMultiScale(frame,1.98,20)
            for(x,y,w,h) in smile:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
            
            eye = cascade2.detectMultiScale(frame,1.1,10)
            for(x,y,w,h) in eye:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) == ord('q'): 
           break;
cap.release()
cv2.destroyAllWindows()