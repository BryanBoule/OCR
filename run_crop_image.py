import cv2
from helpers import constants
import pytesseract as tess
import numpy as np
from tqdm import tqdm
import os

FILENAME = 'final_rotation_scikit_output_driving_license_resized.png'

# TODO : make a proper cropping algorithm
if __name__ == '__main__':
    img = cv2.imread(constants.OUTPUT_PATH + FILENAME)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, im = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(os.path.join(constants.OUTPUT_PATH, 'final_rotation_' +
                             FILENAME), im)
    print(cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
