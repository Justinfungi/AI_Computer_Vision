import cv2

# Read image and convert to grayscale
img = frame
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply LSD algorithm to detect lines
lsd = cv2.createLineSegmentDetector(0)
dlines = lsd.detect(gray)[0]

# Convert the output to a Numpy array for usage with drawSegments function
dlines = np.array(dlines)

# Reshape the array to fit the expected format by drawSegments function
dlines = dlines.reshape((dlines.shape[0], 1, 4))

# Draw detected lines on the original image
drawn_img = lsd.drawSegments(img, dlines)

# Display the result
fig = plt.figure(figsize=(20, 40))
plt.imshow(drawn_img)
plt.show()

