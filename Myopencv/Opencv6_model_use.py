import cv2 as cv
import numpy as np


people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
haar_cascade = cv.CascadeClassifier("haar_face.xml")

#features= np.load("features.npy")
#labels=np.load("labels.npy")
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("faced_train.yml")


img=cv.imread(r'../Myopencv/Faces/val/ben_afflek/2.jpg')
cv.imshow("person",img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("person",gray)


faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 7)
for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'{label} with a confidence of {confidence}')
    cv.putText(img, str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
cv.imshow("Detected",img)

cv.waitKey(10000)
cv.destroyAllWindows
