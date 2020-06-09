from flask import Flask, render_template, request, jsonify, url_for
from src import find_pred as fp
from src import test
import os

IMAGE_FOLDER = os.path.join('static', 'images/interactive')
IMAGE_PATHS = [os.path.join(IMAGE_FOLDER,f) for f in os.listdir(IMAGE_FOLDER)]

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER
app.config['ALL_IMAGES'] = IMAGE_PATHS

@app.route('/', methods=['GET'])
def index():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'radon_fft_rad_2.png')
    return render_template('index.html', user_image = full_filename, all_images = IMAGE_PATHS)


# Histology page
@app.route('/histology/', methods=['GET'])
def histology():
    return render_template('histology_frame.html')

# Mammogram page
@app.route('/mammogram/', methods=['GET'])
def mammogram():
    return render_template('mammogram_frame.html')

# Machine Learning page
@app.route('/ml/', methods=['GET'])
def ml():
    return render_template('ml_frame.html')

# Interactive page
@app.route('/interactive/', methods=['GET'])
def interactive():
    return render_template('interactive_frame.html', img_paths = IMAGE_PATHS)


@app.route('/test/', methods=['GET', 'POST'])
def test():
    print('hello')
    return 'hello'


@app.route('/display', methods=['POST'])
def display():
    user_pred = request.json
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
    
