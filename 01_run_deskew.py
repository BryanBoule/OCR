# USAGE
# python 01_run_deskew.py --filename constat_1.jpg
# filename must be in ./data/input/ folder

import math
import cv2
import os
import numpy as np
from constants import PATH, OUTPUT_PATH
from deskew import determine_skew

# import argparse
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--filename", required=True,
#                 help="path to input image")
# args = vars(ap.parse_args())
#
# FILENAME = args["filename"]

FILENAME = './constat_1.jpg'

def cv2_rotate(image, angle, background):
    old_width, old_height = image.shape[:2]
    # degres to radian
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(
        np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(
        np.cos(angle_radian) * old_height)
    # get coordinates of the center
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    # each pixel will get a coordinate (x,y) and the 2d rotation matrix will
    # https://docs.opencv.org/3.4/da/d54
    # /group__imgproc__transform.html#gafbbc470ce83812914a70abfb604f4326
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat,
                          (int(round(height)), int(round(width))),
                          borderValue=background)


def deskew_image():
    image = cv2.imread(os.path.join(PATH, FILENAME))
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    rotated = cv2_rotate(image, angle, (0, 0, 0))
    cv2.imwrite(os.path.join(OUTPUT_PATH, f'deskewed_' +
                             FILENAME), rotated)


if __name__ == '__main__':
    deskew_image()
