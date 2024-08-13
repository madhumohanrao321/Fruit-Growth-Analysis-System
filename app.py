from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from PIL import Image
import pandas as pd
import sklearn
import re
import random
from random import *

app = Flask(__name__)
model = load_model('model.h5')
@app.route('/')
def main():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/dr')
def dr():
    return render_template('output.html')

def dr_image(image):
    img = image.resize((128, 128))
    img = np.array(img)
    img = img / 255.0
    return img


@app.route('/dr_result', methods=['POST'])
def dr_result():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image = request.files['image']
    img = dr_image(Image.open(image))
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)

    if prediction > 0.5:
        return render_template('output.html',pred = "The Apple is RIPPENED. You can eat it...")
    else:
        return render_template('output.html',pred = "The Apple is RAW. You can't eat it...")


if __name__ == "__main__":
    app.run(debug=True)