from flask import Flask
from flask_cors import CORS
import json

api = Flask(__name__)
CORS(api)

@api.route('/get_music')
def my_get_music():
    #with open('fug_sin.xml', 'r') as f:
        #sample = json.loads(f.read())
    #response_body = sample
    # fill this in with returning midi file
    return {
        "filepath": "Filepath of generated file here"
    }