import cv2
import numpy as np
capture = cv2.VideoCapture("ISS_space_walk.mp4")
for x in range(1500):
	capture.read()

while True:
	ret, img = capture.read()

	imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(imgray,40,255,0)

	contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(img, contours, -1, (0,255,0), 3)

	# Show keypoints
	cv2.imshow("OpenCV2 - NASA Space Walk", img)

	if 0xFF & cv2.waitKey(5) == 27:
		break

capture.release()
cv2.destroyAllWindows()