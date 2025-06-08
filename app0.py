import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# Title and description
st.title("Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image and the model will classify it as **NORMAL** or **PNEUMONIA**.")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("pneumonia_detection_model.h5")
    return model

model = load_model()

# Preprocess function
def preprocess_image(image: Image.Image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((150, 150))
    image_array = np.array(image).reshape(1, 150, 150, 1) / 255.0
    return image_array

# Upload image
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]
    result = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Show result
    st.markdown(f"### Prediction: **{result}**")
    st.markdown(f"### Confidence: **{confidence*100:.2f}%**")
