# Proof of Concept 1
#
# Finds matches in the image
# Calculates depth via disparity using first match found

import cv2
import numpy as np


# load images
left_img = cv2.imread("assets/left.JPEG")
right_img = cv2.imread("assets/right.JPEG")

# convert to grayscale
left_gs = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
right_gs = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

# use ORB to detect keypoints
orb = cv2.ORB_create()
left_k, left_d = orb.detectAndCompute(left_gs, None)
right_k, right_d = orb.detectAndCompute(right_gs, None)

# match features using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(left_d, right_d)

# sort matches by hamming distance
matches = sorted(matches, key=lambda x: x.distance)

# get first matched keypoints
pt1 = left_k[matches[0].queryIdx].pt
pt2 = right_k[matches[0].trainIdx].pt

# calculate disparity in x
disparity = abs(pt1[0] - pt2[0])

# vars
baseline = 30.48 # cm
focal_len = 15294
# focal length = 26mm. pixel size = 1.7um

# calculate disparity
if disparity > 0:
    depth = (focal_len * baseline) / disparity
    print(f"Distance to box: {depth:.2f} cm")
else:
    print("Disparity is too small to calculate depth")

# visualize matches
matched_img = cv2.drawMatches(left_img, left_k, right_img, right_k, matches[:10], None)
cv2.imwrite("output/matches.JPEG", matched_img)


# output
#cv2.imwrite("output/left_gray.JPEG", left_gs)
#cv2.imwrite("output/right_gray.JPEG", right_gs)
