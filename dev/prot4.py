# Proof of Concept 4
#
# Performs preprocessing, identifies boundary by finding shift from longest green section to non green along centerline vertical

import cv2
import numpy as np
import math

buffer = 50

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

# crops
def crop(img, border, height):
    avg = 0
    for point in border: avg += point[1]
    avg = int(avg / len(border))

    min_y = max(0, avg - buffer)
    max_y = min(height, avg + buffer)

    return cv2.cvtColor(img[min_y:max_y, :], cv2.COLOR_BGR2GRAY) 

# process each image
def process_image(name):
    # load image
    img = cv2.imread(f"assets/s3/{name}.JPEG")
    height, width = img.shape[:2]
    center_x = width // 2

    # HSV color segments
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # create masks to detect green regions
    green_lower_bound = np.array([35, 40, 40])
    green_upper_bound = np.array([85, 255, 255])
    img_mask = cv2.inRange(img_hsv, green_lower_bound, green_upper_bound)

    # find border
    border = find_boundary(img_mask, height, width)
    if not border: return None

    # label images
    img_marked = label_border(border, img.copy())

    # crop image around border
    img_crop = crop(img, border, height)

    # save images
    cv2.imwrite(f"output/{name}/1_mask.JPEG", img_mask)
    cv2.imwrite(f"output/{name}/2_marked.JPEG", img_marked)
    cv2.imwrite(f"output/{name}/3_crop.JPEG", img_crop)

    return img_crop, border

# remove outliers from disparity lists
def remove_outliers(x_distances, y_distances):
    q1 = np.percentile(x_distances, 25)
    q3 = np.percentile(x_distances, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    filtered = [(x, y) for x, y in zip(x_distances, y_distances) if lower <= x <= upper]
    return zip(*filtered)

# perform a block match between left and right images
def calc_depth(left, right):
    # SIFT feature detect
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(left, None)
    kp2, des2 = sift.detectAndCompute(right, None)

    # brute force matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x:x.distance)[:10]

    # get x disparities and distance to midline
    x_distances = []
    y_distances = []
    for m in matches:
        left_kp = kp1[m.queryIdx]
        right_kp = kp2[m.trainIdx]
        disparity = left_kp.pt[0] - right_kp.pt[0]
        if disparity > 0: 
            x_distances.append(disparity)
            y_distances.append(0.5 * (abs(right_kp.pt[1] - buffer) + abs(left_kp.pt[1] - buffer)))

    # calculate weighted average: closest to midline gets 100% influence, others get 30% scaled by y_distance
    x_distances, y_distances = remove_outliers(x_distances, y_distances)
    min_y = min(y_distances)
    if min_y == 0: weights = [1 if y == min_y else 0.3 * 1 / y for y in y_distances]
    else:
        weights = [min_y / y if y == min_y else 0.3 * min_y / y for y in y_distances]
        weighted_sum = sum(x * weight for x, weight in zip(x_distances, weights))
    weighted_avg = weighted_sum / sum(weights)

    # vars
    baseline = 30.48
    focal_len = 15294
    # focal length = 26mm. pixel size = 1.7um

    # calculate depth
    if weighted_avg > 1:
        depth = (focal_len * baseline) / weighted_avg
        print(f"Distance to boundary: {depth:.2f} cm")
    else:
        print("Disparity is too small to calculate depth.")

    # save image
    img_matches = cv2.drawMatches(left, kp1, right, kp2, matches, None, flags=2)
    cv2.imwrite(f"output/match.JPEG", img_matches)

    return depth

# task flow
def process_imgs():
    # get edge detection around detected border
    left, left_border = process_image("left")
    right, right_border = process_image("right")

    # calculate disparity and depth
    calc_depth(left, right)

if __name__ == "__main__":
    process_imgs()