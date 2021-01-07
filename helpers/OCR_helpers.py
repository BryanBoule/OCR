import requests
import cv2


def download_file(url):
    local_filename = url.split('/')[-1]
    with requests.get(url) as r:
        assert r.status_code == 200, f'error, status code is {r.status_code}'
        with open(local_filename, 'wb') as f:
            f.write(r.content)
    return local_filename


def get_content_list(text):
    lines = text.split('\n')
    return lines


def load_image(path):
    img = cv2.imread(path)
    return img


def display_image(path):
    img = load_image(path)
    cv2.imshow(str(img), img)
    cv2.waitKey(0)
