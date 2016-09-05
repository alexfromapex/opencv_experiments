import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    ret,frame = cam.read()
    ret2,frame2 = cam.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Remove noise
    frame = cv2.medianBlur(frame,11)
    frame2 = cv2.medianBlur(frame2,11)

    # Movement detection via visbile light change
    diff_frame = cv2.subtract(gray,gray2)

    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
     
    # Thresholds
    params.minThreshold = 1
    params.maxThreshold = 255
     
    # Filter by Area
    params.filterByArea = True
    params.minArea = 50

    # Distance between blobs
    params.minDistBetweenBlobs = 130

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5

    # Set up the detector with parameters
    detector = cv2.SimpleBlobDetector(params)
     
    # Detect blobs
    keypoints = detector.detect(diff_frame)

    # Draw detected blobs
    im_blobs = cv2.drawKeypoints(diff_frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('Detection',im_blobs)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.waitKey(0)
cam.release()
cv2.destroyAllWindows()
