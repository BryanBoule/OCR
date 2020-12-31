from OCR_helpers import load_image, display_image
import cv2
from imutils import contours
import numpy as np
import imutils
import cv2

if __name__ == '__main__':
    img = load_image('../data/input/constat_1.jpg')
    img_ref = load_image('../data/input/preneur_assurance_ref.JPG')
    display_image(img_ref)

    gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)[1]
    display_image(thresh)

    # locate contours
    # find contours in the OCR-A image (i.e,. the outlines of the digits)
    # sort them from left to right, and initialize a dictionary to map
    # digit name to the ROI
    refCnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    refCnts = imutils.grab_contours(refCnts)
    #print(refCnts)
    refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
    print(thresh.shape)
    print(img.shape)


    # initialize a rectangular (wider than it is tall) and square
    # structuring kernel
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 700))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
