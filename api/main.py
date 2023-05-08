import os
import pickle
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/predict_image', methods=['POST'])
def predict_image():
    if request.method == 'POST' and 'image' in request.files:
        image_file = request.files['image']
        # Load the image to predict
        img = Image.open(image_file).convert('L')
        img = img.resize((128, 128))
        img = np.array(img) / 255.0
        img_array = img.flatten()

        # Load the trained SVM model
        with open('flask_api/api/svm_classifier.pkl', 'rb') as f:
            svm_classifier = pickle.load(f)

        # Use the model to predict the class of the image
        predicted_class = svm_classifier.predict([img_array])[0]

        # Return the predicted class as a JSON response
        return jsonify({'predicted_class': predicted_class})

    return jsonify({'error': 'Invalid request method or missing image file'})


if __name__ == '__main__':
    app.run(debug=True)
