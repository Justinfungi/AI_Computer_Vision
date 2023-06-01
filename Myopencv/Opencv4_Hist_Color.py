# Histogram Computation
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread("Photos/mei.jpg")
img = cv.resize(img, (500,700), interpolation=cv.INTER_AREA) #INTER_AREA/LINER/CUBIC
cv.imshow("Resize",img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


#Grayscale Histogram
gray_hist= cv.calcHist([gray],[0],None,[256],[0,256])

plt.figure("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of pixels")
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()


# Colour Histogram
blank = np.zeros(img.shape[:2], dtype='uint8')
mask = cv.circle(blank, (img.shape[1]//2,img.shape[0]//10*4), 100, 255, -1)
masked = cv.bitwise_and(img,img,mask=mask)
cv.imshow('Mask', masked)
plt.figure()
plt.title('Colour Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
colors = ('b', 'g', 'r')
for i,col in enumerate(colors):
    hist = cv.calcHist([img], [i], mask, [256], [0,256])
    plt.plot(hist, color=col)
    plt.xlim([0,256])

plt.show()







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

# BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)

# BGR to L*a*b
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', lab)

# BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RGB', rgb)
#BGR is the opencv using coloring system
#RGB is matplotlib using

# HSV to BGR
lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
cv.imshow('LAB --> BGR', lab_bgr)




'''
Opencv is BGR
matplotlib Mediapipe is RGB
'''




# Color Channel
import cv2 as cv
import numpy as np
img = cv.imread("Photos/mei.jpg")
img = cv.resize(img, (500,700), interpolation=cv.INTER_AREA) #INTER_AREA/LINER/CUBIC
cv.imshow("Resize",img)



b, g, r = cv.split(img)
cv.imshow("Blue",b)
cv.imshow("Green",g)
cv.imshow("Red",r)
print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)



blank=np.zeros(img.shape[:2], dtype="uint8")
blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])
cv.imshow("Blue",blue)
cv.imshow("Green",green)
cv.imshow("Red",red)



merged = cv.merge([b,g,r])
cv.imshow("merge Photos", merged)



































cv.waitKey(10000)
cv.destroyAllWindows
