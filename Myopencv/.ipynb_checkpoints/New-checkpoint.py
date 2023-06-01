# Color Space
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("Photos/mei.jpg")
img = cv.resize(img, (500,700), interpolation=cv.INTER_AREA) #INTER_AREA/LINER/CUBIC
cv.imshow("Resize",img)

# plt.imshow(img)
# plt.show()

# BGR to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
cv.waitKey(0)
