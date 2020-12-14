import os
import pdfplumber
from helpers import OCR_helper

if __name__ == '__main__':
    print(os.listdir())
    path = './Email_example.pdf'


    with pdfplumber.open(path) as pdf:
        page = pdf.pages[0]
        text = page.extract_text()
        print(text)

    content = OCR_helper.get_content_list(text)
    print(content)