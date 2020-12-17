import datetime
import json
import eml_parser
import constants
import os


def json_serial(obj):
    if isinstance(obj, datetime.datetime):
        serial = obj.isoformat()
        return serial


with open(os.path.join(constants.PATH, 'Email_object_example.eml'), 'rb') as \
        fhdl:
    raw_email = fhdl.read()

ep = eml_parser.EmlParser()
parsed_eml = ep.decode_email_bytes(raw_email)

print(json.dumps(parsed_eml, default=json_serial))
