import os
from helpers import constants
import pytesseract as tess
from PIL import Image

tess.pytesseract.tesseract_cmd = \
    r'C:\Users\PC\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

FILENAME = "Email_example.png"

if __name__ == '__main__':
    img = Image.open(os.path.join(constants.PATH, FILENAME))
    text = tess.image_to_string(img)
    print(text)
