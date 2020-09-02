import argparse
import numpy as np
import cv2

img = cv2.imread('field.jpg')
"""
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Masks out any green color
green_mask = cv2.inRange(hsv, (25, 100, 100), (70, 255, 255))
player_mask = cv2.bitwise_not(green_mask)


res_bgr = cv2.cvtColor(green, cv2.COLOR_HSV2BGR)
res_gray = cv2.cvtColor(res_bgr, cv2.COLOR_BGR2GRAY)

# Defining a kernel to do morphological operation in threshold
# image to get better output.
kernel = np.ones((13, 13), np.uint8)
thresh = cv2.threshold(
    res_gray, 1, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)"""

img = cv2.imread('field2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 75, 255, apertureSize=3)
minLineLength = 50
maxLineGap = 20

lines = cv2.HoughLines(edges, 0.5, np.pi/180, 100)

for line in lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

"""
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
"""

cv2.imshow('houghlines5.jpg', img)

#cv2.imshow('image', img)
#cv2.imshow('green', thresh)
#cv2.imshow('green2', green)
#cv2.imshow('green1', res_bgr)
#cv2.imshow('green2', res_gray)

cv2.waitKey(0)
