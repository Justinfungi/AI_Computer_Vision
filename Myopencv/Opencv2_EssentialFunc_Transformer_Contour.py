import cv2 as cv
import numpy as np


img = cv.imread("Photos/park.jpg")
cv.imshow("park", img)

#Convert to gray rescale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray",gray)

# Edge Cascade
canny = cv.Canny(img, 15, 175)
cv.imshow("Canny Edges1", canny)
canny = cv.Canny(img, 125, 15)
cv.imshow("Canny Edges2", canny)
canny = cv.Canny(img, 125, 175)
cv.imshow("Canny Edges3", canny)
canny = cv.Canny(blur, 125, 175) # reduce the amount of edges
cv.imshow("Canny Edges4", canny)

# Diating the Images
dilated =cv.dilate(canny,(7,7), iterations=1)
cv.imshow("Dilated Images", dilated)

# Eroding
eroded= cv.erode(dilated, (3,3), iterations=1)
cv.imshow("Eroded Images", eroded)



'''-----------------# Blurring------------------------'''
# use to reduce noise


# Average blur
average = cv.blur(img,(3,3))
cv.imshow("Avergae blur", average)

# GaussianBlur   weight on surrounding pixels
guass = cv.GaussianBlur(img, (3,3), 0)
cv.imshow("GaussianBlur", guass)

# Midian blur     More effective
median = cv.medianBlur(img, 3)
cv.imshow("Median", median)

# Bilateral    get a retained edge
bilateral = cv.bilateralFilter(img, 3, 15, 15)
cv.imshow("Bilateral",bilateral)















# Image Transformer


img = cv.imread("Photos/park.jpg")
cv.imshow("park", img)

#Translation

def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

'''
x
+ = right
- = left

y
- = Up
+ = Down
'''
translated = translate(img, 100, 100)
cv.imshow("translated", translated)

# Rotated

def rotate (img, angle, rotPoint=None):
    (height,width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions= (width, height)

    return cv.warpAffine(img, rotMat, dimensions)

rotated = rotate(img, -45)
cv.imshow('Rotated', rotated)
rotated= rotate(img, 45)
cv.imshow("Rotate",rotated)



# Flipping
flip= cv.flip(img, -1)
cv.imshow("flip",flip)
# 1 horizontally
# 0 vertically
#-1 180 degree

# Resize
resized = cv.resize(img, (500,500), interpolation=cv.INTER_AREA) #INTER_AREA/LINER/CUBIC
cv.imshow("Resize",resized)

#Cropping
cropped = img[50:200, 100:200]
cv.imshow("Cropped",cropped)




























cv.waitKey(10000)
cv.destroyAllWindows
