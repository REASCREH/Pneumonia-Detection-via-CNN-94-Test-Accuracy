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
        image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
        img_array = np.array(image)
        
        # Check if image needs to be inverted (some X-rays are white-on-black)
        if np.mean(img_array) > 127:  # If majority of pixels are white
            img_array = 255 - img_array  # Invert colors
            
        resized_img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        input_image = resized_img.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0

        # Display uploaded image with a fixed width
        st.image(image, caption="Uploaded X-ray", width=300)

        # Predict only if model loaded successfully
        if model is not None:
            prediction = model.predict(input_image)[0][0]
            pneumonia_prob = prediction if prediction >= 0.5 else 1 - prediction
            result_text = "🟢 NORMAL (No Pneumonia)" if prediction >= 0.5 else "🔴 PNEUMONIA DETECTED"
            
            # Show result with better formatting
            st.subheader("Prediction Result")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Status:** {result_text}")
            with col2:
                st.markdown(f"**Confidence:** {pneumonia_prob:.1%}")
            
            # Add interpretation
            st.info("""
            **Note:** This is an AI-assisted prediction and should not be used as a sole diagnostic tool. 
            Always consult with a medical professional for proper diagnosis.
            """)
        else:
            st.warning("Model not loaded properly. Cannot make predictions.")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Model details section
with st.expander("ℹ️ Model Information & Performance"):
    st.markdown("""
    ### 📊 Model Details:
    - **Architecture:** Convolutional Neural Network (CNN)
    - **Input Size:** 150x150 pixels (grayscale)
    - **Output:** Binary classification (Pneumonia/Normal)
    
    ### 🏋️ Training Data:
    - **Training Samples:** 15,116 images
    - **Test Samples:** 2,420 images
    - **Class Distribution:**
        - PNEUMONIA: 11,168 (train), 1,635 (test)
        - NORMAL: 3,948 (train), 785 (test)

    ### 📈 Performance Metrics:
    - **Accuracy:** 93.14%
    - **Precision (Pneumonia):** 95%
    - **Recall (Pneumonia):** 95%
    - **Precision (Normal):** 89%
    - **Recall (Normal):** 90%
    """)

# Footer
st.markdown("---")
st.caption("""
Developed by Qamar • Powered by Streamlit & TensorFlow
\n*For educational purposes only. Not for clinical use.*
""")
