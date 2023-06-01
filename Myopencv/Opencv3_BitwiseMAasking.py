import cv2 as cv
import numpy as np

blank = np.zeros((400,400), dtype="uint8")
#cv.imshow("blank",blank)
rectangle = cv.rectangle(blank.copy(),(30,30),(370,370),255, -1)
circle = cv.circle(blank.copy(),(200,200),200,255, -1)

cv.imshow("Rec", rectangle)
cv.imshow("Cir", circle)


# bitwise AND --> intersecting regions
bitwise_and = cv.bitwise_and(rectangle, circle)
cv.imshow('Bitwise AND', bitwise_and)

# bitwise OR --> non-intersecting and intersecting regions
bitwise_or = cv.bitwise_or(rectangle, circle)
cv.imshow('Bitwise OR', bitwise_or)

# bitwise XOR --> non-intersecting regions
bitwise_xor = cv.bitwise_xor(rectangle, circle)
cv.imshow('Bitwise XOR', bitwise_xor)

# bitwise NOT
bitwise_not = cv.bitwise_not(circle)
cv.imshow('Circle NOT', bitwise_not)










#Masking forcus on the part of the photos

import cv2 as cv
import numpy as np

img = cv.imread("Photos/mei.jpg")
img = cv.resize(img, (500,700), interpolation=cv.INTER_AREA) #INTER_AREA/LINER/CUBIC
cv.imshow("Resize",img)

blank = np.zeros(img.shape[:2], dtype= "uint8")
cv.imshow("Blank",blank)

mask = cv.circle(blank, (img.shape[1]//2,img.shape[0]//2), 100, 255, -1)
cv.imshow("Mask", mask)

masked = cv.bitwise_and(img, img, mask = mask)
cv.imshow("Masked", masked)





'''
Application: Logo Manipulation
In this section we will show you how to fill in
the white lettering of the Coca-Cola logo below
with a background image.
'''

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
#%matplotlib inline #only for jupyter notebook
from IPython.display import Image

'''#import logo'''
img_bgr = cv.imread("./Photos/coca-cola-logo.png")
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
plt.imshow(img_bgr)
logo_w = img_rgb.shape[0]
logo_h = img_rgb.shape[1]

'''#import background'''
# Read in image of color cheackerboad background
img_background_bgr = cv.imread("./Photos/checkerboard_color.png")
img_background_rgb = cv.cvtColor(img_background_bgr, cv.COLOR_BGR2RGB)
# Set desired width (logo_w) and maintain image aspect ratio
aspect_ratio = logo_w / img_background_rgb.shape[1]
dim = (logo_w, int(img_background_rgb.shape[0] * aspect_ratio))
# Resize background image to sae size as logo image
img_background_rgb = cv.resize(img_background_rgb, dim, interpolation=cv.INTER_AREA)
plt.imshow(img_background_rgb)
print(img_background_rgb.shape)


'''#create grayscale'''
img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
# Apply global thresholding to creat a binary mask of the logo
retval, img_mask = cv.threshold(img_gray,127,255,cv.THRESH_BINARY)
plt.imshow(img_mask,cmap="gray")
print(img_mask.shape)
print(retval)

'''#revert grayscale'''
# Create an inverse mask
img_mask_inv = cv.bitwise_not(img_mask)
plt.imshow(img_mask_inv,cmap="gray")


# Create colorful background "behind" the logo lettering
img_background_mask = cv.bitwise_and(img_background_rgb, img_background_rgb, mask=img_mask)
plt.imshow(img_background_mask)








plt.show()
cv.waitKey(10000)
cv.destroyAllWindows
