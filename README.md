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

### Documentation
- pdfplumber: https://github.com/jsvine/pdfplumber
- tesseract: https://pypi.org/project/pytesseract/
- eml_parser: https://pypi.org/project/eml-parser/
- PIL: https://pillow.readthedocs.io/en/stable/
- cv2: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html