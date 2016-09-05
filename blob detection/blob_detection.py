import cv2
import numpy as np
 
# Read image
orig = cv2.imread("horses.jpeg",cv2.IMREAD_UNCHANGED)
im = cv2.imread("horses.jpeg", cv2.IMREAD_GRAYSCALE)

# Setup SimpleBlobDetector parameters
params = cv2.SimpleBlobDetector_Params()
 
# Thresholds
params.minThreshold = 10
params.maxThreshold = 240
 
# Filter by Area
params.filterByArea = True
params.minArea = 1200

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1
 
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.4
 
# Set up the detector with parameters
detector = cv2.SimpleBlobDetector(params)
 
# Detect blobs
keypoints = detector.detect(im)

# Draw detected blobs
im_blobs = cv2.drawKeypoints(orig, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Blobs Detected", im_blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()