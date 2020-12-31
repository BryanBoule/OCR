import numpy as np
import cv2
from OCR_helpers import load_image, display_image
from skimage.filters import threshold_local
from PIL import Image
import matplotlib.pyplot as plt
import keras_ocr

if __name__ == '__main__':
    # # Sample file out of the dataset
    # file_name = './data/output/cv2_output_constat_1.jpg'
    #
    # img = load_image(file_name)
    # display_image(img)
    #
    # # Convert to grayscale for further processing
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # display_image(gray)
    #
    # # Get rid of noise with Gaussian Blur filter
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # display_image(blurred)
    #
    # # Detect white regions
    # rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    # dilated = cv2.dilate(blurred, rectKernel)
    # display_image(dilated)
    #
    # edged = cv2.Canny(dilated, 200, 250, apertureSize=3)
    # display_image(edged)
    #
    # # Detect all contours in Canny-edged image
    # contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE,
    #                                        cv2.CHAIN_APPROX_SIMPLE)
    # image_with_contours = cv2.drawContours(img.copy(), contours, -1,
    #                                        (0, 255, 0), 3)
    # display_image(image_with_contours)



    # keras-ocr will automatically download pretrained
    # weights for the detector and recognizer.
    pipeline = keras_ocr.pipeline.Pipeline()

    # Get a set of three example images
    images = [
        load_image('../data/output/cv2_output_constat_1.jpg'),
        load_image('../data/output/cv2_output_constat_2.jpg')
    ]

    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples
    prediction_groups = pipeline.recognize(images)

    # Plot the predictions
    fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
    for ax, image, predictions in zip(axs, images, prediction_groups):
        keras_ocr.tools.drawAnnotations(image=image, predictions=predictions,
                                        ax=ax)
