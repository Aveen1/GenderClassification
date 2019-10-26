#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import time


# In[2]:


cap = cv2.VideoCapture(0)

cap.set(3, 480) #set width of video frames
cap.set(4, 640) #set height of video frames

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

#list for storing model mean values, age and gender
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']


# In[4]:


#Function to load the caffe model
def load_caffe_models():

    age_net = cv2.dnn.readNetFromCaffe(
        '/Users/Aveen Faheem/Desktop/Anaconda_work/deploy_age.prototxt', 
        '/Users/Aveen Faheem/Desktop/Anaconda_work/age_net.caffemodel')

    gender_net = cv2.dnn.readNetFromCaffe(
        '/Users/Aveen Faheem/Desktop/Anaconda_work/deploy_gender.prototxt', 
        '/Users/Aveen Faheem/Desktop/Anaconda_work/gender_net.caffemodel')

    return(age_net, gender_net)


# In[ ]:


def video_detector(age_net, gender_net):
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

    video_detector(age_net, gender_net)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




