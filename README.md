MNIST Digit Recognition with CNN and XGBoost
This project implements a digit recognition system using the MNIST dataset. The system extracts features using a Convolutional Neural Network (CNN) and classifies digits using an XGBoost model. The project includes a Flask web application for interactive digit prediction.

Features
Feature Extraction: Uses a CNN to extract features from handwritten digit images.
Classification: Uses XGBoost for high-performance digit classification.
Web Interface: Provides a Flask-based web application for drawing digits and viewing predictions.
Prerequisites
Python 3.7 or higher.
A virtual environment (recommended).
Required libraries (see dependencies below).
Setup Instructions
Step 1: Clone the Repository
bash
Copiar
Editar
git clone <repository_url>
cd <repository_name>
Step 2: Create a Virtual Environment
Create a virtual environment named mnist-env:

bash
Copiar
Editar
python -m venv mnist-env
source mnist-env/bin/activate  # On Windows, use mnist-env\Scripts\activate
Step 3: Install Dependencies
Install the required Python libraries:

bash
Copiar
Editar
pip install -r requirements.txt
Step 4: Train the Models
Run the train_model.py script to train the CNN and XGBoost models. This will save the trained models as cnn_feature_extractor_model.h5 and xgboost_mnist_model.joblib.

bash
Copiar
Editar
python train_model.py
Step 5: Run the Flask Application
Start the Flask web server by running app.py:

bash
Copiar
Editar
python app.py
Open your browser and navigate to http://127.0.0.1:5000/.

File Structure
csharp
Copiar
Editar
project/
│
├── app.py                  # Flask web application
├── train_model.py          # Model training script
├── static/                 # Static files (CSS, JS, images)
│   ├── css/
│   ├── images/
│   └── templates/          # HTML templates for Flask
├── cnn_feature_extractor_model.h5    # Saved CNN model
├── xgboost_mnist_model.joblib        # Saved XGBoost model
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
Dependencies
The following Python libraries are required:

plaintext
Copiar
Editar
Flask==2.2.3
tensorflow==2.10.1
xgboost==1.6.2
scikit-learn==1.2.0
numpy==1.21.6
Pillow==9.2.0
joblib==1.2.0
These are listed in the requirements.txt file. Install them with:

bash
Copiar
Editar
pip install -r requirements.txt
Usage
Train the Models: Run train_model.py to train the models and save them to disk.
Start the Web App: Run app.py to start the Flask application.
Draw and Predict: Use the web interface to draw a digit and predict its value.
Environment Details
Python Version: 3.7+
Frameworks:
Flask for the web interface.
TensorFlow for feature extraction.
XGBoost for classification.
Tools:
scikit-learn for preprocessing and metrics.
Pillow for image processing.
Example Commands
Training the Models
bash
Copiar
Editar
python train_model.py
Running the Web App
bash
Copiar
Editar
python app.py
