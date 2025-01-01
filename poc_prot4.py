# Proof of Concept 4
#
# Performs preprocessing, identifies boundary by finding shift from longest green section to non green along centerline vertical

import cv2
import numpy as np
import math

# returns where the longest unmasked strip starts
def longest(line, height):
    max_stretch = 0
    max_start = -1
    current_start = -1
    current_stretch = 0
    for y in range(height - 1, -1, -1):
        # unmasked pixel: get stretch length
        if line[y] == 255:
            if current_start == -1: current_start = y
            current_stretch += 1

        # masked pixel: log transition if largest
        else:
            if current_stretch > max_stretch:
                max_stretch = current_stretch
                max_start = current_start
            current_start = -1
            current_stretch = 0

    if current_stretch > max_stretch or max_start == -1: return -1
    return max_start

# returns the list of points that make up the boundary
def find_boundary(mask, height, width):
    # 20 pixel transition threshold
    threshold = 30

    # 20 verticals
    n = 50
    step = width // n
    verticals = [i * step for i in range(n)]

    # record all valid transition points
    transitions = []
    for x in verticals:
        line = []
        stretch = 0
        start = longest(mask[:, x], height)
        if start != -1: 
            for y in range(start, -1, -1):
                if mask[y, x] == 0:
                    stretch += 1
                    if stretch == threshold:
                        line.append(y + threshold - 1)

                else: stretch = 0
        if not line:
            del x
            n -= 1
            continue
        transitions.append(line)

    # dp table & backtrack table to find flattest line
    dp = [[float('inf')] * len(transitions[i]) for i in range(n)]
    parent = [[-1] * len(transitions[i]) for i in range(n)]

    # initialize first line
    for j in range(len(transitions[0])): dp[0][j] = 0

    # fill dp table
    for i in range(1, n):
        for j, current_y in enumerate(transitions[i]):
            for k, previous_y in enumerate(transitions[i - 1]):
                if previous_y != - 1 and current_y != -1:
                    cost = abs(current_y - previous_y)
                    if dp[i][j] > dp[i - 1][k] + cost:
                        dp[i][j] = dp[i - 1][k] + cost
                        parent[i][j] = k
    
    # find min cost in last column
    min_cost = float('inf')
    last = -1
    for j in range(len(transitions[-1])):
        if dp[-1][j] < min_cost:
            min_cost = dp[-1][j]
            last = j

    # backtrack to find the flattest path
    path = []
    for i in range(n - 1, -1, -1):
        if last == -1: break
        path.append((verticals[i], transitions[i][last]))
        last = parent[i][last]

    return path

def label_border(border, img):
    for point in border: cv2.circle(img, point, 30, (0, 255, 0), 3)
    return img

# process each image
def process_image(name):
    # load image
    img = cv2.imread(f"assets/v2/{name}.JPEG")
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
    border = find_boundary(img_mask, height, width)
    if not border: return None

    # label images
    img_marked = label_border(border, img.copy())

    # canny edge detector
    # img_edges = cv2.Canny(img_crop, threshold1=100, threshold2=400)

    # save images
    cv2.imwrite(f"output/{name}/1_mask.JPEG", img_mask)
    cv2.imwrite(f"output/{name}/2_marked.JPEG", img_marked)

    return img, border

def crop(img, border, height):
    max_y = 0
    min_y = float('inf')
    for point in border:
        if point[1] > max_y: max_y = point[1]
        if point[1] < min_y: min_y = point[1]

    min_y = max(0, min_y - 100)
    max_y = min(height, max_y + 100)

    return cv2.cvtColor(img[min_y:max_y, :], cv2.COLOR_BGR2GRAY)

# task flow
def process_imgs():
    left, left_border = process_image("left")
    right, right_border = process_image("right")

if __name__ == "__main__":
    process_imgs()


# IDEA 1
# use dp solution to get an approximate border
# crop left and right images around the border
# canny edge detect on the cropped images
# either use ORB and match/anchor or train a model to find disparity

# IDEA 2
# train a model to classify grass
# walk up images to find border using model
# ====