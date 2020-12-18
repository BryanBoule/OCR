# METHOD 1
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rotate

# METHOD 2
import math
from typing import Tuple, Union
import cv2

# MANDATORY
import os
import numpy as np
from helpers import constants
from deskew import determine_skew

FILENAME = 'driving_license_resized.png'


# TODO: add docstrings
def cv2_rotate(
        image: np.ndarray, angle: float,
        background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    """

    Parameters
    ----------
    image
    angle
    background

    Returns
    -------

    """

    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(
        np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(
        np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat,
                          (int(round(height)), int(round(width))),
                          borderValue=background)


def deskew_image(method='cv2'):
    """

    Parameters
    ----------
    method

    Returns
    -------

    """

    if method == 'scikit':
        image = io.imread(os.path.join(constants.PATH, FILENAME))
        grayscale = rgb2gray(image)
        angle = determine_skew(grayscale)
        rotated = rotate(image, angle, resize=True) * 255
        io.imsave(os.path.join(constants.OUTPUT_PATH, f'{method}_output_' +
                               FILENAME), rotated.astype(np.uint8))

    elif method == 'cv2':
        image = cv2.imread(os.path.join(constants.PATH, FILENAME))
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        angle = determine_skew(grayscale)
        rotated = cv2_rotate(image, angle, (0, 0, 0))
        cv2.imwrite(os.path.join(constants.OUTPUT_PATH, f'{method}_output_' +
                                 FILENAME), rotated)

    else:
        print('chose between cv2 or scikit')


if __name__ == '__main__':
    deskew_image(method='scikit')
    deskew_image(method='cv2')
