from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image, ImageOps
import io
import tensorflow as tf
import joblib

# Load saved models
cnn_model = tf.keras.models.load_model("cnn_feature_extractor_model.h5")
xgb_model = joblib.load("xgboost_mnist_model.joblib")

app = Flask(__name__, 
            static_folder='static', 
            template_folder='static/templates')

@app.route('/')
def home():
    
    return render_template('home.html')

@app.route('/page2')
def page2():
    return render_template('page2.html')

@app.route('/predictpage')
def predictpage():

    return render_template('predictpage.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read())).convert('L')  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert colors if needed (optional for dark backgrounds)
    image = image.resize((28, 28))  # Resize to 28x28 pixels

    # Convert image to numpy array and normalize
    image_array = np.array(image).astype('float32') / 255.0  # Normalize pixel values to 0-1
    image_array = np.expand_dims(image_array, axis=(0, -1))  # Reshape to (1, 28, 28, 1)

    # Extract features using CNN
    features = cnn_model.predict(image_array)  # Shape: (1, 128)

    # Classify using XGBoost
    prediction = xgb_model.predict(features)[0]

    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
    
