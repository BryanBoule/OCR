import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rotate
import os
import constants
from deskew import determine_skew

image = io.imread(os.path.join(constants.PATH,'driving_license_resized.png'))
grayscale = rgb2gray(image)
angle = determine_skew(grayscale)
rotated = rotate(image, angle, resize=True) * 255
io.imsave(os.path.join(constants.PATH,'output_license.png'), rotated.astype(
    np.uint8))

# import math
# from typing import Tuple, Union
#
# import cv2
# import numpy as np
#
# from deskew import determine_skew
#
#
# def rotate(
#         image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
# ) -> np.ndarray:
#     old_width, old_height = image.shape[:2]
#     angle_radian = math.radians(angle)
#     width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
#     height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)
#
#     image_center = tuple(np.array(image.shape[1::-1]) / 2)
#     rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#     rot_mat[1, 2] += (width - old_width) / 2
#     rot_mat[0, 2] += (height - old_height) / 2
#     return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)
#
# image = cv2.imread('input.png')
# grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# angle = determine_skew(grayscale)
# rotated = rotate(image, angle, (0, 0, 0))
# cv2.imwrite('output.png', rotated)