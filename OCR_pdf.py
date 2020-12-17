import os
import pdfplumber
from helpers import OCR_helper, constants

if __name__ == '__main__':

    with pdfplumber.open(os.path.join(constants.PATH,
                                      "Email_example.pdf")) as pdf:
        page = pdf.pages[0]
        text = page.extract_text()
        print(text)

    content = OCR_helper.get_content_list(text)
    print(content)
