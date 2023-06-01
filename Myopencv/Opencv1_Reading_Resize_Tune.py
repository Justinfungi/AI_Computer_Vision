import cv2 as cv
import numpy as np




'''  --------------------Read img and vid------------------------------------'''


img=cv.imread("Photos/cat_large.jpg")
cv.imshow("Cat", img)
cv.waitKey(0)


import cv2 as cv
capture = cv.VideoCapture("dog.mp4")
# if argument is 0, turn on camera
while True:
    istrue, frame = capture.read()
    cv.imshow("Video",frame)

    if cv.waitKey(20) & 0xFF == ord("d"):
        break
capture.release()
cv.destroyAllWindows




import cv2
import sys

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

while cv2.waitKey(1) != 27: # Escape
    has_frame, frame = source.read()
    if not has_frame:
        break
    cv2.imshow("Camera Preview", frame)

source.release()
cv2.destroyWindow(win_name)

























''' ----------------------Resizing and Rescaling------------------------------'''


def rescaleFrame(frame, scale=0.75):
    #Images, Videos and Live Video
    width=int(frame.shape[1]*scale)  # frame.shape[1] is exactly the width
    height=int(frame.shape[0]*scale)
    dimensions =(width, height)
    return cv.resize(frame, dimensions,interpolation=cv.INTER_AREA)

def changeRes(width,height):
    # OnLive video
    capture.set(3,width)
    capture.set(4,height)

#capture = cv.VideoCapture("Videos/dog.mp4")
capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    frame_resized= rescaleFrame(frame, scale=0.25)
    cv.imshow("Video",frame)
    cv.imshow("Video_resized",frame_resized)

    if cv.waitKey(20) & 0xFF == ord("d"):
        break
capture.release()
cv.destroyAllWindows
cv.waitKey(0)













''' ----------------------Brightness / Contrasting / Handling Overflow using np.clip------------------------------'''

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread("Photos/mei.jpg")
img = cv.resize(img, (500,700), interpolation=cv.INTER_AREA) #INTER_AREA/LINER/CUBIC
cv.imshow("Resize",img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)



'''
                Addition or Brightness
The first operation we discuss is simple addition of images.
This results in increasing or decreasing the brightness of the
image since we are eventually increasing or decreasing the intensity
values of each pixel by the same amount. So, this will result in a
global increase/decrease in brightness.
'''



matrix = np.ones(img.shape, dtype = "uint8") * 50  #effective matrix
print(matrix)

img_brighter = cv.add(img, matrix)
img_darker   = cv.subtract(img, matrix)

# Show the images
plt.figure(figsize=[18,5])
plt.subplot(131); plt.imshow(img_darker);  plt.title("Darker");
plt.subplot(132); plt.imshow(img);         plt.title("Original");
plt.subplot(133); plt.imshow(img_brighter);plt.title("Brighter");





'''
         Multiplication or Contrast
Just like addition can result in brightness change,
multiplication can be used to improve the contrast of the image.
Contrast is the difference in the intensity values of the pixels
of an image. Multiplying the intensity values with a constant
can make the difference larger or smaller ( if multiplying factor is < 1 ).
'''


matrix1 = np.ones(img_rgb.shape) * .8
matrix2 = np.ones(img_rgb.shape) * 1.2

img_rgb_darker   = np.uint8(cv2.multiply(np.float64(img_rgb), matrix1))
img_rgb_brighter = np.uint8(cv2.multiply(np.float64(img_rgb), matrix2))

# Show the images
plt.figure(figsize=[18,5])
plt.subplot(131); plt.imshow(img_rgb_darker);  plt.title("Lower Contrast");
plt.subplot(132); plt.imshow(img_rgb);         plt.title("Original");
plt.subplot(133); plt.imshow(img_rgb_brighter);plt.title("Higher Contrast");







'''
Can you see the weird colors in some areas of the
image after multiplication?
The issue is that after multiplying,
the values which are already high,
are becoming greater than 255.
Thus, the overflow issue. How do we overcome this?
Handling Overflow using np.clip
'''

matrix1 = np.ones(img_rgb.shape) * .8
matrix2 = np.ones(img_rgb.shape) * 1.2

img_rgb_lower   = np.uint8(cv2.multiply(np.float64(img_rgb), matrix1))
img_rgb_higher  = np.uint8(np.clip(cv2.multiply(np.float64(img_rgb), matrix2),0,255))

# Show the images
plt.figure(figsize=[18,5])
plt.subplot(131); plt.imshow(img_rgb_lower);  plt.title("Lower Contrast");
plt.subplot(132); plt.imshow(img_rgb);         plt.title("Original");
plt.subplot(133); plt.imshow(img_rgb_higher);plt.title("Higher Contrast");




















cv.waitKey(10000)
cv.destroyAllWindows





















































































































cv.waitKey(10000)
cv.destroyAllWindows
