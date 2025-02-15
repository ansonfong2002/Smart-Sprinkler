# Proof of Concept 3
#
# Performs preprocessing, identifies boundary by finding shift from longest green section to non green along centerline vertical

import cv2
import numpy as np
import math

def find_boundary(img, height, center_x):
    # centerline pixels
    centerline = img[:, center_x]

    # determine largest unmasked (green) stretch
    max_stretch = 0
    max_start = -1
    current_start = -1
    current_stretch = 0
    for y in range(height - 1, -1, -1):
        # unmasked pixel: get stretch length
        if centerline[y] == 255:
            if current_start == -1: current_start = y
            current_stretch += 1

        # masked pixel: log transition if largest
        else:
            if current_stretch > max_stretch:
                max_stretch = current_stretch
                max_start = current_start
            current_start = -1
            current_stretch = 0

    # only unmasked or masked
    if current_stretch > max_stretch or max_start == -1: return -1

    # find transition to masked (non-green) stretch over threshold
    threshold = 15
    current_stretch = 0
    for y in range(max_start - 1, -1, -1):
        if centerline[y] == 0:
            current_stretch += 1
            if current_stretch >= threshold:
                return y + current_stretch
        
        else: current_stretch = 0

    return -1

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

    # find border
    border = (center_x, find_boundary(img_mask, height, center_x))
    cv2.circle(img, border, 30, (0, 255, 0), 3)
    cv2.circle(img_mask, border, 30, (0, 255, 0), 3)

    # save images
    cv2.imwrite(f"output/{name}/1_mask.JPEG", img_mask)
    cv2.imwrite(f"output/{name}/2_marked.JPEG", img)

    return img, border

# task flow
def process_imgs():
    left, left_border = process_image("left")
    right, right_border = process_image("right")

    print(f"Left border: {left_border}")
    print(f"Right border: {right_border}")

if __name__ == "__main__":
    process_imgs()