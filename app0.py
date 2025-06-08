import streamlit as st

# MUST be the very first Streamlit command in your script
st.set_page_config(page_title="Pneumonia Detection", layout="centered")

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.title("🩺 Pneumonia Detection from Chest X-rays")

# Cache model loading for speed
@st.cache_resource
def load_model():
    model_path = "pneumonia_detection_model.h5"
    return tf.keras.models.load_model(model_path)

model = load_model()

def preprocess_image(uploaded_file):
    # Convert uploaded file to OpenCV grayscale image, resize and normalize
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (150, 150))
    img = img.reshape(1, 150, 150, 1) / 255.0
    return img

uploaded_file = st.file_uploader("Upload a chest X-ray image (jpg/png)")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded X-ray Image", use_column_width=True)
    
    # Preprocess and predict
    img_array = preprocess_image(uploaded_file)
    prediction = model.predict(img_array)[0][0]
    
    result = "🫁 PNEUMONIA" if prediction > 0.5 else "✅ NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    st.markdown(f"### Prediction: {result}")
    st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
