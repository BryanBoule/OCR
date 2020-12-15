import cv2
from helpers import image_processing_helper, constants
import pytesseract as tess
import numpy as np
from tqdm import tqdm
import os

tess.pytesseract.tesseract_cmd = \
    r'C:\Users\PC\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

if __name__ == '__main__':

    img = cv2.imread(os.path.join(constants.PATH,
                                  "driving_license_resized.png"))

    list_len = []
    for i in tqdm(range(-40,40,2)):
        rotated = image_processing_helper.rotate_image(img, i)
        text = tess.image_to_string(rotated)
        list_len.append(len(text))

    print(list_len)
    print(np.argmax(list_len))

    angle_fix = -40 + 2*(np.argmax(list_len))
    final_rotation = image_processing_helper.rotate_image(img, angle_fix)
    final_text = tess.image_to_string(final_rotation)
    print(final_text)

    cv2.imshow('Fixed inclination', final_rotation)
    cv2.waitKey(0)

    cv2.imshow('Fixed inclination', image_processing_helper.rotate_image(
        img, angle_fix-2))
    cv2.waitKey(0)

    cv2.imshow('Fixed inclination', image_processing_helper.rotate_image(
        img, angle_fix-4))
    cv2.waitKey(0)
