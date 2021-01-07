import datetime
import json
import eml_parser
from helpers import constants
import os

FILENAME = 'Email_object_example.eml'


def json_serial(obj):
    if isinstance(obj, datetime.datetime):
        serial = obj.isoformat()
        return serial


if __name__ == '__main__':
    with open(os.path.join(constants.PATH, FILENAME),
              'rb') as fhdl:
        raw_email = fhdl.read()

    ep = eml_parser.EmlParser()
    parsed_eml = ep.decode_email_bytes(raw_email)

    print(json.dumps(parsed_eml, default=json_serial))
