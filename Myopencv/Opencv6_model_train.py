# Face Recognition with OpenCV's built-in recognizer
import cv2 as cv
import numpy as np
import os


'''
p=[]
for i in os.listdir(
    r'/home/fung0311/Documents/DM_MAC_AI/AI/Algorithm/Computer_Vision/Myopencv/Faces/train'):
    p.append(i)

'''

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
haar_cascade = cv.CascadeClassifier("haar_face.xml")
DIR = r'../Myopencv/Faces/train'
features=[]
labels=[]

def create_train():
    # set the path for a specific person
    #label the person
    for person in people:
        path = os.path.join(DIR, person)
        label= people.index(person)
        #print(label)



        # Training with photos
        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            img_array = cv.imread(img_path)
            gray= cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            # identify faces
            faces_rect=faces_rect = haar_cascade.detectMultiScale(
                                        gray, scaleFactor = 1.1, minNeighbors = 4)


            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)


create_train()
print("Training done ------------------------------------")

'''
print(f'{len(features)} features')
print(f'{len(labels)} labels')
'''

features = np.array(features, dtype="object")
labels = np.array(labels)
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the face_recognizer on the features and labels

face_recognizer.train(features,labels)

face_recognizer.save("faced_train.yml")
np.save("features.npy",features)
np.save("label.npy",labels)









cv.waitKey(10000)
cv.destroyAllWindows
