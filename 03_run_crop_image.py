import cv2
import os
from constants import OUTPUT_PATH
import json

FILENAME = 'deskewed_constat_1.jpg'
JSON_COORDINATES = 'to_crop.json'


def get_boxing_coordinates(json_file_path):
    with open(json_file_path) as json_file:
        data = json.load(json_file)
        y = int(data['margin_from_top'])
        x = int(data['margin_from_left'])
        h = int(data['height'])
        w = int(data['width'])
    return y, x, h, w


def crop_image(image, y, x, h, w):
    crop_img = image[y:y + h, x:x + w].copy()
    return crop_img


if __name__ == '__main__':
    img = cv2.imread(os.path.join(OUTPUT_PATH, FILENAME))
    MARGIN_FROM_TOP, MARGIN_FROM_LEFT, HEIGHT, WIDTH = get_boxing_coordinates(
        JSON_COORDINATES)
    croped_image = crop_image(img,
                              MARGIN_FROM_TOP,
                              MARGIN_FROM_LEFT,
                              HEIGHT,
                              WIDTH)
    cv2.imwrite(os.path.join(OUTPUT_PATH, 'crop_' + FILENAME),
                croped_image)
    cv2.imshow('croped_image', croped_image)
    cv2.waitKey(0)
