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
    page_info = request.json
    img = page_info['image']
    user_pred = page_info['user_pred']

    # Get image predictions and true labels
    y_true, y_pred, y_prob = fp.find_pred(img)
    msg = fp.compare_pred(user_pred, img)

    return jsonify({'y_true': y_true, 'y_pred': y_pred, 'img_path': img_path})


@app.route('/mew_image', methods=['POST'])
def new_image():
    page_info = request.json

    # Render new image
    path = '/static/images/interactive/'
    img = fp.pick_random_image(path)
    img_path = os.path.join(path, img)
    img_string = f'<img src="{img_path}" alt="User Image" width="600">'

    return jsonify({'img_string': img_string})




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
    
