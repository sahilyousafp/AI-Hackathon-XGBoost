# MNIST Digit Recognition with CNN and XGBoost

This project demonstrates a digit recognition system using the MNIST dataset. The system is designed to showcase how XGBoost, a high-performance gradient boosting framework traditionally used for tabular data, can be integrated into an image classification pipeline by combining it with Convolutional Neural Networks (CNNs). The project also includes an interactive web interface that explains the underlying concepts visually and allows users to draw digits and view predictions.

---

## Features

- **Interactive Web Interface**: Provides a user-friendly interface to draw digits, view predictions, and explore the model pipeline visually.
- **XGBoost for Tabular Data**: Demonstrates XGBoost’s role in traditional tabular data processing and how it achieves high performance in classification tasks.
- **XGBoost for Image Data**: Extends XGBoost to work with image data by extracting features using CNNs, showcasing how the pipeline adapts for MNIST classification.
- **Visual Explanation**: Offers graphics and diagrams to explain how CNN feature extraction transforms image data into tabular form for XGBoost.

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
