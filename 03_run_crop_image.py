import cv2
import os
from constants import OUTPUT_PATH

FILENAME = 'cv2_output_constat_1.jpg'
# output of 02_run_segmentation.py.py
MARGIN_FROM_TOP = 216
MARGIN_FROM_LEFT = 26
HEIGHT = 80
WIDTH = 224


def crop_image(image, y, x, h, w):
    crop_img = image[y:y + h, x:x + w].copy()
    return crop_img


if __name__ == '__main__':
    img = cv2.imread(os.path.join(OUTPUT_PATH, FILENAME))
    croped_image = crop_image(img,
                              MARGIN_FROM_TOP,
                              MARGIN_FROM_LEFT,
                              HEIGHT,
                              WIDTH)
    cv2.imwrite(os.path.join(OUTPUT_PATH, 'crop_' + FILENAME),
                croped_image)
    cv2.imshow("cropped", croped_image)
    cv2.waitKey(0)
