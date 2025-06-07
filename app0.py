import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Constants
MODEL_PATH = "pneumonia_detection_model.h5"

IMG_SIZE = 150

# Load the model
@st.cache_resource
def load_cnn_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_cnn_model()

# App Title
st.set_page_config(page_title="Pneumonia Detection App", layout="centered")
st.title("🩺 Pneumonia Detection from Chest X-rays")
st.write(
    """
    This app uses a trained Convolutional Neural Network (CNN) to predict if a given chest X-ray image indicates **Pneumonia** or not.
    """
)

# Upload File
uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

# Prediction logic
if uploaded_file is not None:
    try:
        # Load and preprocess image
        image = Image.open(uploaded_file).convert("L")
        img_array = np.array(image)
        resized_img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        input_image = resized_img.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0

        # Display uploaded image
        st.image(image, caption="Uploaded X-ray", use_column_width=True)

        # Predict
        prediction = model.predict(input_image)[0][0]
        result_text = "🟢 PNEUMONIA NOT PRESENT" if prediction >= 0.5 else "🔴 PNEUMONIA PRESENT"

        # Show result
        st.subheader("Prediction Result")
        st.markdown(f"**{result_text}**")
        st.markdown(f"**Confidence Score:** `{prediction:.4f}`")

    except Exception as e:
        st.error(f"Error processing image: {e}")

# Model details section
with st.expander("ℹ️ Model Training Info & Performance"):
    st.markdown("""
    ### 📊 Model Training Summary:
    - **Training Data Shape:** (15116, 150, 150, 1)
    - **Test Data Shape:** (2420, 150, 150, 1)
    - **Classes:**
        - PNEUMONIA: 11168 (train), 1635 (test)
        - NORMAL: 3948 (train), 785 (test)

    ### ✅ Model Test Performance:
    - **Loss:** 0.1677
    - **Accuracy:** 93.14%
    - **Precision/Recall (Pneumonia):** 95% / 95%
    - **Precision/Recall (Normal):** 89% / 90%
    """)

# Footer
st.markdown("---")
st.caption("Developed by Qamar • Powered by Streamlit & TensorFlow")
