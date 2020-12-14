import requests

def download_file(url):
    """

    Parameters
    ----------
    url

    Returns
    -------

    """

    local_filename = url.split('/')[-1]
    with requests.get(url) as r:
        assert r.status_code == 200, f'error, status code is {r.status_code}'
        with open(local_filename, 'wb') as f:
            f.write(r.content)
    return local_filename

def get_content_list(text):
    lines = text.split('\n')
    return lines