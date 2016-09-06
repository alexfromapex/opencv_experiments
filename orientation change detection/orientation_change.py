import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    ret,frame = cam.read()q
    ret2,frame2 = cam.read()
    height,width,channels = frame.shape
    print "Height: "+str(height)+", Width: "+str(width)+", Channels: "+str(channels)
    frame3 = cv2.subtract(frame,frame2)
    cv2.imshow('Orienation Change Detection',frame3)

    print(frame3[0,0])

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.waitKey(0)
cam.release()
cv2.destroyAllWindows()
