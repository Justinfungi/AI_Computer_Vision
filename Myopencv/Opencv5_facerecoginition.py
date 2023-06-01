# Face Detection with Haar Cascades
import cv2 as cv
import numpy as np

img = cv.imread("Photos/group 2.jpg")
#img = cv.resize(img, (500,700), interpolation=cv.INTER_AREA) #INTER_AREA/LINER/CUBIC
#cv.imshow("Resize",img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# really sensitive
#just for fun not accurate
haar_cascade = cv.CascadeClassifier("haar_face.xml")

#Recognize session
# larger minNeighbors can reduce sensitivity
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 7)
print(f'{len(faces_rect)} faces are found')

#draw session
for (x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

#show
cv.imshow("detected Faces", img)






















# Video capture

# Face Detection with Haar Cascades
import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0)
# if argument is 0, turn on camera
haar_cascade = cv.CascadeClassifier("haar_face.xml")
while True:
    istrue, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 7)
    print(f'{len(faces_rect)} faces are found')
    for (x,y,w,h) in faces_rect:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    cv.imshow("Video",frame)

    if cv.waitKey(20) & 0xFF == ord("d"):
        break
capture.release()
cv.destroyAllWindows









cv.waitKey(10000)
cv.destroyAllWindows
