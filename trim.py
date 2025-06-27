# test_predict.py
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model.save("poultry_disease_model.h5")


# List of class names in same order used during training
classes = ['Coccidiosis', 'Healthy', 'Newcastle', 'Salmonella']

# Load and preprocess test image
img_path = "test.jpg"  # <- Replace with your test image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make prediction
predictions = model.predict(img_array)
predicted_class = classes[np.argmax(predictions)]

print("Predicted Class:", predicted_class)
