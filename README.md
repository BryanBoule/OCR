# OCR project

Author: Bryan Boulé
<br>Creation: 2020/12/14

### Requirements
!apt-get install ocrmypdf -q
<br>!pip install pdfplumber -q 

### Context
This project aims at having a better understanding on how OCR works
. Different OCR are tried:
- _run_OCR_pdf.py_: OCR on a pdf file
- _run_OCR_image.py_: OCR on a png file
- _run_OCR_rotated_image.py_: detect angle to fix text skew (not best)
- _run_test_deskew.py_: fix image skew
- _run_json_from_eml.py_: output json object from eml file

### Technics
- Canny edge detection is a technique to extract useful structural information from different vision objects and dramatically reduce the amount of data to be processed.
- The Hough transform is a feature extraction technique to find imperfect
 instances of objects within a certain class of shapes by a voting procedure
 . The classical Hough transform was concerned with the identification of lines in the image, but later the Hough transform has been extended to identifying positions of arbitrary shapes, most commonly circles or ellipses.
 
 ### Documentation
- pdfplumber: https://github.com/jsvine/pdfplumber
- tesseract: https://pypi.org/project/pytesseract/
- eml_parser: https://pypi.org/project/eml-parser/
- PIL: https://pillow.readthedocs.io/en/stable/
- cv2: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html
- parsr

### TO WATCH
- https://www.pyimagesearch.com/2020/08/17/ocr-with-keras-tensorflow-and-deep-learning/#pyi-pyimagesearch-plus-optin-modal
- https://towardsdatascience.com/your-guide-to-natural-language-processing-nlp-48ea2511f6e1
- https://www.youtube.com/watch?v=XaQ0CBlQ4cY
- https://www.youtube.com/watch?v=l8ZYCvgGu0o
- https://www.youtube.com/watch?v=6a6L_9USZxg&t=3s
- https://www.youtube.com/watch?v=4DrCIVS5U3Y