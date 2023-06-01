'''
Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields (2017)

Highlights
Train elsewhere, perform inference in an OpenCV application

Supports Caffe, Tensorflow, Torch and Darknet.

Supported layers : AbsVal, AveragePooling, BatchNormalization,
Concatenation, Convolution (including dilated convolution),
Crop, Deconvolution, DetectionOutput (SSD-specific layer), Dropout,
Eltwise (+, *, max), Flatten, FullyConnected, LRN, LSTM, MaxPooling,
MaxUnpooling, MVN, NormalizeBBox (SSD-specific layer), Padding, Permute,
Power, PReLU (including ChannelPReLU with channel-specific slopes),
PriorBox (SSD-specific layer), ReLU, RNN, Scale, Shift, Sigmoid, Slice,
Softmax, Split, TanH

Use Caffe model trained on the Multi-Person Image Dataset (MPI)
to demonstrate human pose estimation for a single person.
'''
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib

protoFile = "./pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "./model/pose_iter_160000.caffemodel"

nPoints = 15
POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

im = cv2.imread("./Photos/mei.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
inWidth = im.shape[1]
inHeight = im.shape[0]


netInputSize = (368, 368)
inpBlob = cv2.dnn.blobFromImage(im, 1.0 / 255, netInputSize, (0, 0, 0), swapRB=True, crop=False)
net.setInput(inpBlob)

# Forward Pass
output = net.forward()

# Display probability maps
plt.figure(figsize=(20,5))
for i in range(nPoints):
    probMap = output[0, i, :, :]
    displayMap = cv2.resize(probMap, (inWidth, inHeight), cv2.INTER_LINEAR)
    plt.subplot(2, 8, i+1); plt.axis('off'); plt.imshow(displayMap, cmap='jet')
# X and Y Scale
scaleX = inWidth / output.shape[3]
scaleY = inHeight / output.shape[2]

# Empty list to store the detected keypoints
points = []

# Treshold
threshold = 0.1

for i in range(nPoints):
    # Obtain probability map
    probMap = output[0, i, :, :]

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    # Scale the point to fit on the original image
    x = scaleX * point[0]
    y = scaleY * point[1]

    if prob > threshold :
        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
    else :
        points.append(None)
imPoints = im.copy()
imSkeleton = im.copy()
# Draw points
for i, p in enumerate(points):
    cv2.circle(imPoints, p, 8, (255, 255,0), thickness=-1, lineType=cv2.FILLED)
    cv2.putText(imPoints, "{}".format(i), p, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, lineType=cv2.LINE_AA)


# Draw skeleton
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        cv2.line(imSkeleton, points[partA], points[partB], (255, 255,0), 2)
        cv2.circle(imSkeleton, points[partA], 8, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)

plt.figure(figsize=(50,50))
plt.subplot(121); plt.axis('off'); plt.imshow(imPoints);
plt.subplot(122); plt.axis('off'); plt.imshow(imSkeleton);
plt.show()
cv.waitKey(20000)
cv.destroyAllWindows
