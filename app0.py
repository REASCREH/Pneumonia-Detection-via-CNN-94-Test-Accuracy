import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import io
import os

# --- Configuration ---
# Define the path to your saved model
# IMPORTANT: Ensure 'pneumonia_detection_model.h5' is in the same directory as this script.
model_path = "pneumonia_detection_model.h5"
img_size = 150 # Image size used during training

# --- Model Loading ---
@st.cache_resource # Cache the model to avoid reloading it on every rerun
def get_model():
    """
    Loads the Keras model from the specified path.
    Caches the model to improve performance.
    """
    if not os.path.exists(model_path):
        st.error(f"Error: Model file '{model_path}' not found. "
                 f"Please ensure the model is in the same directory as this script.")
        return None
    try:
        model = load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = get_model()

# --- Prediction Function ---
def predict_pneumonia(image_bytes):
    """
    Processes an image and makes a prediction using the loaded model.

    Args:
        image_bytes (bytes): The raw bytes of the uploaded image file.

    Returns:
        tuple: A tuple containing (prediction_text, confidence_score) or (None, None) if an error occurs.
    """
    if model is None:
        return "Model not loaded.", None

    try:
        # Open the image using Pillow and convert to grayscale
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('L')
        # Convert PIL image to NumPy array
        img_arr = np.array(pil_image)

        # Resize the image to img_size x img_size using OpenCV (cv2)
        resized_arr = cv2.resize(img_arr, (img_size, img_size))

        # Reshape for the model (add batch dimension and channel dimension)
        # The model expects (batch_size, img_height, img_width, channels)
        input_image = resized_arr.reshape(-1, img_size, img_size, 1)

        # Normalize pixel values to 0-1 range
        input_image = input_image / 255.0

        # Make prediction
        prediction = model.predict(input_image)[0][0]

        result_text = ""
        # Assuming 0.5 is the threshold for binary classification
        if prediction <= 0.5:
            result_text = "NORMAL (Pneumonia NOT present)"
        else:
            result_text = "PNEUMONIA PRESENT"

        return result_text, prediction

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return "Error during prediction.", None

# --- Streamlit UI ---
st.set_page_config(
    page_title="Pneumonia Detection App",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Pneumonia Detection App")

st.write(
    "In this application, a Convolutional Neural Network (CNN) model has been trained "
    "to detect pneumonia from chest X-ray images."
)

st.subheader("Model Training Information:")
st.markdown("""
The model was trained on a dataset with the following characteristics:
* **Combined Training Data Shape:** (15116, 150, 150, 1)
* **Combined Training Labels Shape:** (15116,)
* **Combined Test Data Shape:** (2420, 150, 150, 1)
* **Combined Test Labels Shape:** (2420,)

**Class Distribution in Combined Training Set:**
* PNEUMONIA: 11168
* NORMAL: 3948

**Class Distribution in Combined Test Set:**
* PNEUMONIA: 1635
* NORMAL: 785
""")

st.subheader("Model Performance (on test set):")
st.markdown("""
* **Loss:** 0.1677
* **Accuracy:** 94.14%

The model achieved good performance, with 95% precision and recall for detecting Pneumonia and 89% precision and 90% recall for detecting Normal cases on the test set.
""")

st.markdown("---")

st.header("Upload an X-ray image for prediction:")

uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["png", "jpg", "jpeg"],
    help="Upload a chest X-ray image (PNG, JPG, JPEG formats are supported)."
)

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded X-ray Image', use_column_width=True)
    st.write("Processing image...")

    # Read the image bytes
    image_bytes = uploaded_file.getvalue()

    # Perform prediction
    prediction_label, confidence_score = predict_pneumonia(image_bytes)

    if prediction_label is not None:
        st.markdown("### Prediction Result:")
        if "PNEUMONIA PRESENT" in prediction_label:
            st.error(f"The model predicts: **{prediction_label}**")
        else:
            st.success(f"The model predicts: **{prediction_label}**")

        if confidence_score is not None:
            st.info(f"Confidence Score: **{confidence_score:.4f}**")

        st.markdown("---")
        st.write("Upload another image above to get a new prediction.")
