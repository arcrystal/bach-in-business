from flask import Flask
from flask_cors import CORS
from pipeline import get_music_filename

api = Flask(__name__)
CORS(api)

@api.route('/get_music')
def my_get_music():
    filepath = get_music_filename('fugues')

    return {
        "filepath": filepath
    }