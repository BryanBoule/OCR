import os
import pdfplumber
from helpers import OCR_helpers, constants

FILENAME = "Email_example.pdf"

if __name__ == '__main__':
    with pdfplumber.open(os.path.join(constants.PATH,
                                      FILENAME)) as pdf:
        page = pdf.pages[0]
        text = page.extract_text()
        print(text)

    content = OCR_helpers.get_content_list(text)
    print(content)
