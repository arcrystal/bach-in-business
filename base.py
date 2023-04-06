from flask import Flask
import json

api = Flask(__name__)

@api.route('/get_music')
def my_get_music():
    with open('sampleJSON.json', 'r') as f:
        sample = json.loads(f.read())
    response_body = sample
    # fill this in with returning midi file

    return response_body