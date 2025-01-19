# MNIST Digit Recognition with CNN and XGBoost

This project implements a digit recognition system using the MNIST dataset. The system extracts features using a Convolutional Neural Network (CNN) and classifies digits using an XGBoost model. It includes a Flask-based web application for drawing digits and viewing predictions.

---

## Features

- **Feature Extraction**: Uses a CNN to extract features from handwritten digit images.
- **Classification**: Uses XGBoost for high-performance digit classification.
- **Web Interface**: Provides a Flask-based web application for drawing digits and viewing predictions.

---

## Prerequisites

- **Python 3.7** or higher.
- A virtual environment named `mnist-env`.
- Required libraries (see dependencies below).

---

## Setup Instructions

```bash
# Step 1: Clone the Repository
git clone <repository_url> && cd <repository_name>

# Step 2: Create a Virtual Environment
python -m venv mnist-env && source mnist-env/bin/activate  # For Windows: mnist-env\Scripts\activate

# Step 3: Install Dependencies
pip install -r requirements.txt

