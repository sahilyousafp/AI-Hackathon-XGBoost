import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize and reshape data
X_train = X_train.astype('float32') / 255.0  # Normalize to 0-1 range
X_test = X_test.astype('float32') / 255.0
X_train = np.expand_dims(X_train, axis=-1)  # Reshape to (28, 28, 1)
X_test = np.expand_dims(X_test, axis=-1)

# Create CNN for feature extraction
def create_cnn_feature_extractor():
    input_layer = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    feature_layer = Dense(128, activation='relu')(x)
    cnn_model = Model(inputs=input_layer, outputs=feature_layer)
    return cnn_model

# Create and compile CNN
cnn_model = create_cnn_feature_extractor()
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Extract features
features_train = cnn_model.predict(X_train)
features_test = cnn_model.predict(X_test)

# Train XGBoost classifier on extracted features
X_train_feat, X_val_feat, y_train_feat, y_val_feat = train_test_split(features_train, y_train, test_size=0.2, random_state=42)
xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=10, eval_metric='mlogloss', use_label_encoder=False)
xgb_classifier.fit(X_train_feat, y_train_feat)

# Evaluate the model
y_pred = xgb_classifier.predict(features_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save models
cnn_model.save("cnn_feature_extractor_model.h5")
joblib.dump(xgb_classifier, "xgboost_mnist_model.joblib")

print("Models saved successfully!")


