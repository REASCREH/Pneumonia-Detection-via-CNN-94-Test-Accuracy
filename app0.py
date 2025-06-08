import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Pneumonia Detection", layout="centered")

st.title("🩺 Pneumonia Detection from Chest X-rays")

# Load the trained model
@st.cache_resource
def load_model():
    model_path = "pneumonia_detection_model.h5"
    return tf.keras.models.load_model(model_path)

model = load_model()

# Preprocess uploaded image
def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (150, 150))
    img = img.reshape(1, 150, 150, 1) / 255.0
    return img

# Image uploader
uploaded_file = st.file_uploader("Upload a chest X-ray image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded X-ray Image", use_column_width=True)

    # Predict
    img_array = preprocess_image(uploaded_file)
    prediction = model.predict(img_array)[0][0]
    result = "🫁 PNEUMONIA" if prediction > 0.5 else "✅ NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.markdown(f"### Prediction: `{result}`")
    st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")
