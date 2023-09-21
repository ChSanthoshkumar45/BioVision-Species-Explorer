import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained machine learning model
model = tf.keras.models.load_model('final_project/trained_model (1).h5')

# Define a function to preprocess the uploaded image
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(150, 150))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = image.astype('float32')
    image /= 255.0
    return image

# Define the Flask route for the home page
@app.route('/')
def index():
    return render_template('index.html', result=None, file=None)  # Pass 'file' as None

# Define the Flask route for the prediction API
@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'no file uploaded'})
    file = request.files['file']
    # Check if the file is a valid image
    if file.filename == '':
        return jsonify({'error': 'no file selected'})
    if not file.content_type.startswith('image/'):
        return jsonify({'error': 'invalid file type'})

    # Save the uploaded file to a temporary location
    file_path = 'temp.jpg'
    file.save(file_path)

    # Preprocess the image and make a prediction
    image = preprocess_image(file_path)
    prediction = model.predict(image)[0]
    species = ['African_Elephant', 'Amur_Leopard', 'Arctic_Fox', 'Chimpanzee', 'Jaguars', 'Lion', 'Orangutan', 'Panda', 'Panthers', 'Rhino', 'cheetahs']
    result = {'species': species[np.argmax(prediction)], 'confidence': float(max(prediction))}

    # Remove the temporary file
    os.remove(file_path)

    return render_template('index.html', result=result, file=file)  # Pass 'file' to the template

if __name__ == '__main__':
    app.run(debug=True)
