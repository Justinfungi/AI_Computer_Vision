# Contour Detection
'''
Shape analysis
'''

#img = cv.imread("Photos/mei.jpeg")
img = cv.imread("Photos/mei.jpg")
img = cv.resize(img, (500,700), interpolation=cv.INTER_AREA) #INTER_AREA/LINER/CUBIC
cv.imshow("Resize",img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# By blur-canny-find contour
blur =cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT) # if blur, the contours reduce significantly
cv.imshow("Blur", blur)
canny = cv.Canny(blur,125, 175)
cv.imshow("Canny edge", canny)
contours, hierarchies = cv. findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) # EXTERNEL LIST
print(f'{len(contours)} contour found')


#by threshold find contour
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY) #threshold is 125, max is 256
cv.imshow("Thresh",thresh)
contours1, hierarchies1 = cv. findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) # EXTERNEL LIST
print(f'{len(contours1)} contour found')


#draw contour
blank=np.zeros(img.shape,dtype="uint8") #create a blank for drawing contour
cv.drawContours(blank, contours, -1, (0,0,255), 1)
cv.imshow("Contours Drawn", blank)
blank1=np.zeros(img.shape,dtype="uint8")
cv.drawContours(blank1, contours1, -1, (0,0,255), 1)
cv.imshow("Contours Drawn1", blank1)

# a combined contour drawing
blank3=np.zeros(img.shape,dtype="uint8")
cv.drawContours(blank3, contours, -1, (0,0,255), 1)
cv.drawContours(blank3, contours1, -1, (0,0,255), 1)
cv.imshow("Contours Drawn2", blank3)





# Thresholded
# dst = cv.adaptiveThreshold( src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst] )
img = cv.imread("Photos/mei.jpg")
img = cv.resize(img, (500,700), interpolation=cv.INTER_AREA) #INTER_AREA/LINER/CUBIC
cv.imshow("Resize",img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Simple Thresholding
threshold, thresh = cv.threshold(gray, 150,255, cv.THRESH_BINARY)
cv.imshow("Simple Thresholding", thresh)

threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV )
cv.imshow('Simple Thresholded Inverse', thresh_inv)

# Adaptive Thresholding
# find an optimal value for threshold
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 9)
cv.imshow('Adaptive Thresholding', adaptive_thresh)

'''
Application: Sheet Music Reader
Suppose you wanted to build an application that could read (decode) sheet music. This is similar to
Optical Character Recognigition (OCR) for text documents where the goal is to recognize text characters.
In either application, one of the first steps in the processing pipeline is to isolate
the important information in the image of a document (separating it from the background).
This task can be accomplished with thresholding
techniques. Let's take a look at an example.
'''






# Edge detection
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread("Photos/mei.jpg")
img = cv.resize(img, (500,700), interpolation=cv.INTER_AREA) #INTER_AREA/LINER/CUBIC
cv.imshow("Resize",img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


#Laplacian
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow("Laplacian",lap)

#Sobel
sobelx = cv.Sobel(gray, cv.CV_64F,1 ,0)
sobely = cv.Sobel(gray, cv.CV_64F,0 ,1)
combined_sobel = cv.bitwise_or(sobelx, sobely)

cv.imshow("X", sobelx)
cv.imshow("Y", sobely)
cv.imshow('Combined Sobel', combined_sobel)

#Canny
canny = cv.Canny(gray, 150, 175)
cv.imshow('Canny', canny)










cv.waitKey(10000)
cv.destroyAllWindows
