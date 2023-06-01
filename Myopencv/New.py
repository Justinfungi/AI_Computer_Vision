# Annotation

mport cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread("Photos/mei.jpg")
img = cv.resize(img, (500,700), interpolation=cv.INTER_AREA) #INTER_AREA/LINER/CUBIC
cv.imshow("Resize",img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
