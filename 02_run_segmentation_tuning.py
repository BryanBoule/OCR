# USAGE
# python 02_run_segmentation.py --filename deskewed_constat_1.jpg
# filename must be in ./data/output/ folder

import numpy as np
import cv2
import json
import os
from OCR_helpers import load_image, display_image
from constants import OUTPUT_PATH

# import argparse
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--filename", required=True,
#                 help="path to input image")
# args = vars(ap.parse_args())
#
# FILENAME = args["filename"]

FILENAME = 'deskewed_constat_1.jpg'

# for yellow
# lower = np.array([110, 50, 50], dtype="uint8")
# upper = np.array([130, 255, 255], dtype="uint8")

# for blue
# lower = np.array([170, 0, 0], dtype="uint8")
# upper = np.array([255, 170, 170], dtype="uint8")

def get_crop_points(start_point, end_point):
    margin_from_top = start_point[1]
    margin_from_left = start_point[0]
    height = end_point[1] - start_point[1]
    width = end_point[0] - start_point[0]
    return margin_from_top, margin_from_left, height, width


if __name__ == '__main__':

    img = load_image(os.path.join(OUTPUT_PATH, FILENAME))

    # get blue line
    # create a mask based on blue color (B, R, G)
    lower = np.array([170, 0, 0], dtype="uint8")
    upper = np.array([255, 170, 170], dtype="uint8")
    mask = cv2.inRange(img, lower, upper)
    img_filtered = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('img_filtered', img_filtered)
    cv2.waitKey(0)

    # convert as gray
    gray = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)

    # blur to reduce noise
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    cv2.imshow('blur_gray', blur_gray)
    cv2.waitKey(0)

    # use canny filter for edge detection
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    cv2.imshow('edges', edges)
    cv2.waitKey(0)

    # Line detection
    rho = 3  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid
    # cell)
    min_line_length = 200  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line
    # segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)


    # filter on horizontal lines (gap between y coordinates of the two
    # points small)
    line_list = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(y1 - y2) < 0.01*img.shape[0]:
                line_list.append(line)

    new_line_list = []
    # --------- filter area ----------
    # inconvenient: lack of genericity
    for line in line_list:
        for x1, y1, x2, y2 in line:
            new_line_list.append(line)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    cv2.imshow('line_image', line_image)
    cv2.waitKey(0)

    # select line with minimum y
    print(new_line_list)

    resu = new_line_list[0]
    for line in new_line_list:
        print(line)
        # y is mesured from top to bottom (min value on top)
        min_y = line_image.shape[0]
        for _, y1, _, _ in line:
            if y1 < min_y:
                resu = line

    x1, y1, x2, y2 = resu[0]

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.rectangle() method
    # Draw a rectangle with blue line borders of thickness of 2 px
    boxing_height = int(0.07 * img.shape[0])
    image = cv2.rectangle(img,
                          (x1, y1),
                          (x2, y2 + boxing_height),
                          (255, 0, 0),
                          thickness)
    cv2.imshow('img', img)
    cv2.waitKey(0)

    margin_from_top, margin_from_left, height, width \
        = get_crop_points((x1, y1), (x2, y2 + boxing_height))

    json_data = {
        'margin_from_top': str(margin_from_top),
        'margin_from_left': str(margin_from_left),
        'height': str(height),
        'width': str(width)
    }

    with open('to_crop.json', 'w') as outfile:
        json.dump(json_data, outfile)
