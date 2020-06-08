from flask import Flask, render_template, request, jsonify
from src import test
import os


app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('my_html.html')


@app.route('/display', methods=['POST'])
def display():
    user_data = request.json
    text = user_data['text']
    words = test.tokenize(text)
    words_count = word_count(text)
    return jsonify({'words': words, 'words_count': words_count})


def word_count(text):
    words = text.split()
    number = test.do_numpy_stuff()
    return len(words)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
    
