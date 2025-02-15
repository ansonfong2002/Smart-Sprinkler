# Proof of Concept 1
#
# Performs preprocessing, isolates largest green and non-green contours by area
# Identifies boundary by intersecting contour polygon with centerline vertical
# Crops images to +/- 100 pixels from the boundary intersect, searches for matches
# Calculates depth using matched anchor point

import cv2
import numpy as np
import math
import heapq

# get biggest contour by area
def get_max_contour(contours):
    if contours:
        return max(contours, key=cv2.contourArea)
    return None

# get intersection
def get_contour_intersection(contour, inv_contour, height, center_x):
    last_on = 0
    for y in range(height - 1, -1, -1):
        # on green contour?
        on_contour = cv2.pointPolygonTest(contour, (center_x, y), False)
        if on_contour > 0:
            last_on = y
            continue

        # if transition to inverted contour is within 3 pixels
        # limitation: if there are "islands" in the contour map, the intersection point may be skewed in the y direction
        on_inv_contour = cv2.pointPolygonTest(inv_contour, (center_x, y), False)
        if on_inv_contour == 1 and y - last_on < 3: return (center_x, y)

    return None

# crop image about grass border midline + convert to grayscale
def crop(image, border, range=100):
    midpoint = border[1]
    bottom = max(0, midpoint - range)
    top = min(image.shape[0], midpoint + range)
    return cv2.cvtColor(image[bottom:top, :], cv2.COLOR_BGR2GRAY)

# process each image
def process_image(name):
    # load image
    img = cv2.imread(f"assets/s2/{name}.JPEG")
    height, width = img.shape[:2]
    center_x = width // 2

    # gaussian blur and HSV color segments
    img_blur = cv2.GaussianBlur(img, (21, 21), 0)
    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    # create masks to detect green regions
    green_lower_bound = np.array([35, 40, 40])
    green_upper_bound = np.array([85, 255, 255])
    img_mask = cv2.inRange(img_hsv, green_lower_bound, green_upper_bound)
    img_mask_inv = cv2.bitwise_not(img_mask)

    # find max contours
    contours, _ = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inv_contours, _ = cv2.findContours(img_mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = get_max_contour(contours)
    max_inv_contour = get_max_contour(inv_contours)

    # find contour intersection and visualize
    border = get_contour_intersection(max_contour, max_inv_contour, height, center_x)
    if border:
        cv2.circle(img, border, 30, (0, 255, 0), 3)
        img_cropped = crop(img, border)

    # save images
    cv2.imwrite(f"output/{name}/1_blur.JPEG", img_blur)
    cv2.imwrite(f"output/{name}/2_hsv.JPEG", img_hsv)
    cv2.imwrite(f"output/{name}/3_mask.JPEG", img_mask)
    cv2.imwrite(f"output/{name}/4_mask_inv.JPEG", img_mask_inv)
    cv2.imwrite(f"output/{name}/5_marked.JPEG", img)
    cv2.imwrite(f"output/{name}/6_cropped.JPEG", img_cropped)

    return img_cropped, border

# get the average disparity among the best matches
def get_disparities(matches, k, border, x_mid):
    heap = []
    for match in matches:
        left_pt = k[0][match.queryIdx].pt
        right_pt = k[1][match.trainIdx].pt

        # check relative position to anchor x
        if ((left_pt[0] - x_mid) * (right_pt[0] - x_mid) < 0): continue

        # avg y distance
        dist_y = (abs(left_pt[1] - 100) + abs(right_pt[1] - 100)) // 2

        # calculate disparity
        left_x = border[0][0] - left_pt[0]
        right_x = border[1][0] - right_pt[0]
        disparity = abs(left_x - right_x)

        # push y, disparity pair
        heapq.heappush(heap, (dist_y, disparity))
    
    return heap

# find the best anchor between two images and return depth using disparity calculation
def calc_depth(left, right, left_border, right_border):
    # use ORB to detect keypoints
    orb = cv2.ORB_create()
    left_k, left_d = orb.detectAndCompute(left, None)
    right_k, right_d = orb.detectAndCompute(right, None)
    if not left_d.any() or not right_d.any(): return None

    # match feature using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(left_d, right_d)
    if len(matches) > 10:
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:10]
    else: return None

    # get average disparity to best anchors
    disparities = get_disparities(matches, (left_k, right_k), (left_border, right_border), left.shape[1])
    for item in disparities:
        print(item)
        
    return

    # vars
    baseline = 30.48
    focal_len = 15294
    # focal length = 26mm. pixel size = 1.7um

    # calculate depth
    if avg_disparity > 1:
        depth = (focal_len * baseline) / avg_disparity
        print(f"Distance to boundary: {depth:.2f} cm")
    else:
        print("Disparity is too small to calculate depth")

    # save imgs
    matches_img = cv2.drawMatches(left, left_k, right, right_k, matches[:10], None)
    #anchor_left = cv2.circle(left, tuple(math.floor(i) for i in closest[0]), 30, (0, 255, 0), 3)
    #anchor_right = cv2.circle(right, tuple(math.floor(i) for i in closest[1]), 30, (0, 255, 0), 3)
    cv2.imwrite("output/all_matches.JPEG", matches_img)
    #cv2.imwrite("output/left/7_anchor.JPEG", anchor_left)
    #cv2.imwrite("output/right/7_anchor.JPEG", anchor_right)

    return depth

# task flow
def process_imgs():
    left_cropped, left_border = process_image("left")
    right_cropped, right_border = process_image("right")

    print(f"Left border: {left_border}")
    print(f"Right border: {right_border}")
    
    if left_border and right_border:
        depth = calc_depth(left_cropped, right_cropped, left_border, right_border)

if __name__ == "__main__":
    process_imgs()