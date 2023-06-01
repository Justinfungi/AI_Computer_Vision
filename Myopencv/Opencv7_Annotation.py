'''---------------------------------#Drawing Shapes, Putting text---------------------------------------------------------'''
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


blank = np.zeros((400,400,3),dtype="uint8")
cv.imshow("blank", blank)

#paint the img a certain color
blank[:]=0,255,0 #RGB value
cv.imshow("Green", blank)

#with shapes
blank[200:300,300:350]=0,0,255
cv.imshow("Shape",blank)

#draw border
cv.rectangle(blank,(0,0),(500,250),(255,0,0),thickness=2) #(0,0)(250,250) is point coordination
cv.imshow("rectangle",blank)

#fill the rectangle
cv.rectangle(
            blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(255,0,0),thickness=cv.FILLED
            ) #(0,0)(250,250) is point coordination
cv.imshow("rectangle",blank)


#Draw a circle
#blank, center of the circle, radius, color, thickness
#img = cv.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
cv.circle(
        blank,(blank.shape[1]//2,blank.shape[0]//2),40,(122,122,122),thickness=-1
        )
cv.imshow("circle",blank)


# Draw a line
#blank, pt1,pt2, color, thickness, linetyoe
#img = cv.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
cv.line(
        blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(1,122,1),thickness=3
        )
cv.imshow("Line",blank)



'''------------------------------------------------write text------------------------------------------------'''
img = cv.imread("Photos/mei.jpeg")
imgText = img.copy()
text = "Apollo 11 Saturn V Launch, July 16, 1969"
fontScale = 2.3
fontFace = cv.FONT_HERSHEY_PLAIN
fontColor = (100, 100, 100)
fontThickness = 2

cv.putText(img, text, (200, 700), fontFace, fontScale, fontColor, fontThickness, cv.LINE_AA);

# Display the img
plt.imshow(imgText[:,:,::-1])
cv.imshow("Text",img)
cv.waitKey(10000)
cv.destroyAllWindows
