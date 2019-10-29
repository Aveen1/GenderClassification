#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#myproject
import tkinter as tk
from tkinter import *
import cv2
import numpy as np
import time
import cv2
import numpy as np
import time

window=Toplevel()
window.title("Gender Classification")
window.configure(background="black")

#def click():
    #print('hey')
    
#my photo
#photo1=PhotoImage(file="b.gif")
#Label (window, image=photo1, bg="black") .grid(row=0, column=0, sticky=W)

#creating labels
Label (window, text="Welcome to Age and Gender Video Classification", bg="black", fg="white", font="none 32 bold") .grid(row=2, column=3, sticky=W) 
Label (window, text="Click here for webcam", bg="black", fg="white", font="none 28 bold") .grid(row=3, column=0, sticky=W) 

#creating text entry box
#textentry=Entry(window, width=20, bg="white")
#textentry.grid(row=5, column=0, sticky=W)


cap = cv2.VideoCapture(0)

cap.set(3, 480) #set width of video frames
cap.set(4, 640) #set height of video frames

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

#list for storing model mean values, age and gender
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']

#Function to load the caffe model
def load_caffe_models():

    age_net = cv2.dnn.readNetFromCaffe(
        '/Users/Aveen Faheem/Desktop/Anaconda_work/deploy_age.prototxt', 
        '/Users/Aveen Faheem/Desktop/Anaconda_work/age_net.caffemodel')

    gender_net = cv2.dnn.readNetFromCaffe(
        '/Users/Aveen Faheem/Desktop/Anaconda_work/deploy_gender.prototxt', 
        '/Users/Aveen Faheem/Desktop/Anaconda_work/gender_net.caffemodel')

    return(age_net, gender_net)


def video_detector():
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:

        ret, image = cap.read() #read the captured frame
        
        #Face detection with Haar cascades, pre-built model
        face_cascade = cv2.CascadeClassifier('/Users/Aveen Faheem/Desktop/Anaconda_work/haarcascade_frontalface_alt.xml')
        
        #converting image to greyscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        #Image,scalefactor,minNeighbours
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        #counting number of faces
        if(len(faces)>0):
            print("Found {} faces".format(str(len(faces))))
            
        #looping through the faces and creatng rectangles
        for (x, y, w, h )in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)

            # Geting Face 
            face_img = image[y:y+h, h:h+w].copy()
            #blob is doing image pre-processing
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            #Predicting Gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            print("Gender : " + gender)

            #Predicting Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            print("Age Range: " + age)
            
            #writing age and gender on top of the box
            overlay_text = "%s %s" % (gender, age)
            cv2.putText(image, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        #displaying output
        cv2.imshow('frame', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#Function for Face detection, Age detection, and Gender detection
if __name__ == "__main__":
    age_net, gender_net = load_caffe_models()

    
    #video_detector(age_net, gender_net)


#creating a button
Button (window, text="Open webcam", width=20, command=video_detector) .grid(row=6, column=0, sticky=W)


window.mainloop()

            





