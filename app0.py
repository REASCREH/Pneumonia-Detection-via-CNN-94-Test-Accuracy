import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Set page config
st.set_page_config(page_title="Pneumonia Detection", page_icon="🩺", layout="wide")

# Load the model
@st.cache_resource
def load_pneumonia_model():
    model_path = "pneumonia_detection_model.h5"
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_pneumonia_model()

# Preprocess the image
def preprocess_image(image):
    try:
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize to 150x150
        image = cv2.resize(image, (150, 150))
        
        # Normalize and reshape for model input
        image = image / 255.0
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        image = np.expand_dims(image, axis=0)   # Add batch dimension
        
        return image
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Main app function
def main():
    st.title("🩺 Pneumonia Detection from Chest X-ray")
    st.write("""
    This app uses a deep learning model to detect pneumonia from chest X-ray images.
    Upload a chest X-ray image, and the model will predict whether it shows signs of pneumonia.
    """)
    
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray", use_column_width=True)
            
            # Convert to numpy array
            image_np = np.array(image)
        
        with col2:
            st.subheader("Prediction")
            
            if model is not None:
                # Preprocess and predict
                processed_image = preprocess_image(image_np)
                
                if processed_image is not None:
                    with st.spinner("Analyzing the X-ray..."):
                        prediction = model.predict(processed_image)
                        pneumonia_prob = prediction[0][0]
                        
                    # Display results
                    st.write("## Results")
                    
                    if pneumonia_prob > 0.5:
                        st.error(f"🚨 **Pneumonia Detected** (confidence: {pneumonia_prob*100:.2f}%)")
                        st.warning("This result suggests signs of pneumonia. Please consult with a healthcare professional for proper diagnosis.")
                    else:
                        st.success(f"✅ **Normal** (confidence: {(1-pneumonia_prob)*100:.2f}%)")
                        st.info("No signs of pneumonia detected. However, always consult with a doctor for medical diagnosis.")
                    
                    # Show probability meter
                    st.markdown("### Prediction Confidence")
                    st.progress(float(pneumonia_prob if pneumonia_prob > 0.5 else 1-pneumonia_prob))
                    
                    st.write(f"Probability of Pneumonia: {pneumonia_prob*100:.2f}%")
                    st.write(f"Probability of Normal: {(1-pneumonia_prob)*100:.2f}%")
                    
                    # Disclaimer
                    st.markdown("---")
                    st.warning("""
                    **Disclaimer:**  
                    This tool is for informational purposes only and is not a substitute for professional medical advice, 
                    diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider 
                    with any questions you may have regarding a medical condition.
                    """)
            else:
                st.error("Model failed to load. Please check the model file path.")

# Run the app
if __name__ == "__main__":
    main()
