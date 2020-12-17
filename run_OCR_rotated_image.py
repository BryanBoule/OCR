import cv2
from helpers import constants
import pytesseract as tess
import numpy as np
from tqdm import tqdm

tess.pytesseract.tesseract_cmd = \
    r'C:\Users\PC\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


def order_corner_points(corners):
    """

    Parameters
    ----------
    corners

    Returns
    -------

    """

    # Separate corners into individual points
    # Index 0 - top-right
    #       1 - top-left
    #       2 - bottom-left
    #       3 - bottom-right
    corners = [(corner[0][0], corner[0][1]) for corner in corners]
    top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], \
                                       corners[3]
    return (top_l, top_r, bottom_r, bottom_l)


def perspective_transform(image, corners):
    """

    Parameters
    ----------
    image
    corners

    Returns
    -------

    """

    # Order points in clockwise order
    ordered_corners = order_corner_points(corners)
    top_l, top_r, bottom_r, bottom_l = ordered_corners

    # Determine width of new image which is the max distance between
    # (bottom right and bottom left) or (top right and top left) x-coordinates
    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + (
            (bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(
        ((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    # Determine height of new image which is the max distance between
    # (top right and bottom right) or (top left and bottom left) y-coordinates
    height_A = np.sqrt(
        ((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(
        ((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))

    # Construct new points to obtain top-down view of image in
    # top_r, top_l, bottom_l, bottom_r order
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                           [0, height - 1]], dtype="float32")

    # Convert to Numpy format
    ordered_corners = np.array(ordered_corners, dtype="float32")

    # Find perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    return cv2.warpPerspective(image, matrix, (width, height))


def rotate_image(image, angle):
    """

    Parameters
    ----------
    image
    angle

    Returns
    -------

    """

    # Grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


FILENAME = 'scikit_output_driving_license_resized.png'

if __name__ == '__main__':

    img = cv2.imread(constants.OUTPUT_PATH + FILENAME)

    list_len = []
    for i in tqdm(range(-180, 180, 90)):
        rotated = rotate_image(img, i)
        text = tess.image_to_string(rotated)
        list_len.append(len(text))

    print(list_len)
    print(np.argmax(list_len))

    angle_fix = -180 + 90 * (np.argmax(list_len))
    final_rotation = rotate_image(img, angle_fix)
    final_text = tess.image_to_string(final_rotation)
    print(final_text)
