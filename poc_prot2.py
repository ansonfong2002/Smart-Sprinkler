# Proof of Concept 2
#
# Performs preprocessing, isolates largest green and non-green contours by area
# Identifies boundary by intersecting contour polygon with centerline vertical
# Overlays images on top of one another

import cv2
import numpy as np

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

def process_img(name):
    # load image
    img = cv2.imread(f"assets/{name}.JPEG")
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
    if border: cv2.circle(img, border, 30, (0, 255, 0), 3)

    # save images
    cv2.imwrite(f"output/{name}/1_blur.JPEG", img_blur)
    cv2.imwrite(f"output/{name}/2_hsv.JPEG", img_hsv)
    cv2.imwrite(f"output/{name}/3_mask.JPEG", img_mask)
    cv2.imwrite(f"output/{name}/4_mask_inv.JPEG", img_mask_inv)
    cv2.imwrite(f"output/{name}/5_marked.JPEG", img)

    return img, border

# MAIN TASK
def process_imgs():
    left_img, left_border = process_img("left")
    right_img, right_border = process_img("right")

    print(f"Left: {left_border}")
    print(f"Right: {right_border}")

    overlay_img = cv2.addWeighted(left_img, 0.5, right_img, 0.5, 0)
    cv2.imwrite("output/overlay.JPEG", overlay_img)

if __name__ == "__main__":
    process_imgs()