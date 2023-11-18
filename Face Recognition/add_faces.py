import cv2 as cv
import pickle
import numpy as np
import os


cap = cv.VideoCapture(0)
facedetection = cv.CascadeClassifier(r"data\haarcascade_frontalface_default.xml")
faces_data = []

i = 0

name = input("Please inter your Name : ")

while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = facedetection.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resize_img = cv.resize(crop_img, (50, 50))
        
        if len(faces_data) <= 100 and i%10 == 0:
            faces_data.append(resize_img)
        i = i + 1
        
        cv.putText(frame, f"face len : {len(faces_data)}", (50, 50), cv.FONT_HERSHEY_COMPLEX, 0.5, (50, 255, 50), 1)
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 50, 50), 2)
    
    cv.imshow("Frame : ", frame)
    
    k = cv.waitKey(1)
    
    if k==ord("q") or len(faces_data) == 100:
        break
    
cap.release()
cv.destroyAllWindows()

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)



if "names.pkl" not in os.listdir("data\\"):
    names = [name]*100
    with open(r"data\names.pkl", "wb") as f:
        pickle.dump(names, f)

else:
    with open(r"data\names.pkl", "rb") as f:
        names = pickle.load(f)
    names = names + [name]*100
    
    with open(r"data\names.pkl", "wb") as f:
        pickle.dump(names, f)
        

if "faces_data.pkl" not in os.listdir("data\\"):
    with open(r"data\faces_data.pkl", "wb") as f:
        pickle.dump(faces_data, f)

else:
    with open(r"data\faces_data.pkl", "rb") as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    
    with open(r"data\names.pkl", "wb") as f:
        pickle.dump(faces, f)