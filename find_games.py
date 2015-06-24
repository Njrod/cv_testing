# -*- coding: utf-8 -*-

# import the necessary packages
import numpy as np
import cv2
import time

camera = cv2.VideoCapture(0)
time.sleep(0.25)

while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    (grabbed, frame) = camera.read()
    text = "Unoccupied"

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        break

    # load the games image
    image = frame

    # find the red color game in the image
    upper = np.array([65, 65, 255])
    lower = np.array([0, 0, 200])
    mask = cv2.inRange(image, lower, upper)

    # find contours in the masked image ##and keep the largest one
    (_, cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw a green bounding box surrounding the red game
    for c in cnts:
        # approximate the contour
        print(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)

    cv2.imshow("Image", image)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
