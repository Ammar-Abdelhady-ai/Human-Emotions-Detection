from sklearn.neighbors import KNeighborsClassifier

import cv2 as cv
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch


def speak(str1):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)
    

cap = cv.VideoCapture(0)
facedetection = cv.CascadeClassifier(r"data\haarcascade_frontalface_default.xml")

with open(r"data\names.pkl", "rb") as f:
    LABELS = pickle.load(f)
    
with open(r"data\faces_data.pkl", "rb") as f:
    FACES = pickle.load(f)
    
KNNC = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
KNNC.fit(FACES, LABELS)

COL_NAMES = ["NAME", "TIME"]

while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = facedetection.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resize_img = cv.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        
        prediction = KNNC.predict(resize_img)
        
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        
        exist = os.path.isfile(f"Attendance\\Attendance_{date}.csv")
        
        cv.putText(frame, f"{prediction[0]}", (x, y-15), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 50, 50), 2)
        
        attendance = [str(prediction[0]), str(timestamp)]
    
    cv.imshow("Frame : ", frame)
    
    k = cv.waitKey(1)
    
    if k==ord("o"):
        speak("Attendance Taken ..")
        time.sleep(5)
        if exist:
            with open(f"Attendance\\Attendance_{date}.csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open(f"Attendance\\Attendance_{date}.csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvfile.close()
    
    if k==ord("q"):
        break
    
cap.release()
cv.destroyAllWindows()