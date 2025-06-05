from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Define the path to your saved model
model_path ="pneumonia_detection_model.h5""

# Load the trained model
try:
    model = load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Set model to None if loading fails

img_size = 150 # Image size used during training

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Displays a welcome message and information about the model,
    along with an image upload form.
    """
    
    return """
    <html>
    <head>
        <title>Pneumonia Detection App</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
            h1 { color: #333; }
            h2 { color: #555; }
            p { line-height: 1.6; }
            .upload-form { margin-top: 30px; }
            .button { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; }
            .button:hover { background-color: #45a049; }
            .result { margin-top: 20px; padding: 15px; border: 1px solid #eee; border-radius: 5px; background-color: #f9f9f9; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Welcome to the Pneumonia Detection App!</h1>
            <p>In this application, a Convolutional Neural Network (CNN) model has been trained to detect pneumonia from chest X-ray images.</p>

            <h2>Model Training Information:</h2>
            <p>The model was trained on a dataset with the following characteristics:</p>
            <ul>
                <li><strong>Combined Training Data Shape:</strong> (15116, 150, 150, 1)</li>
                <li><strong>Combined Training Labels Shape:</strong> (15116,)</li>
                <li><strong>Combined Test Data Shape:</strong> (2420, 150, 150, 1)</li>
                <li><strong>Combined Test Labels Shape:</strong> (2420,)</li>
            </ul>
            <p><strong>Class Distribution in Combined Training Set:</strong></p>
            <ul>
                <li>PNEUMONIA: 11168</li>
                <li>NORMAL: 3948</li>
            </ul>
            <p><strong>Class Distribution in Combined Test Set:</strong></p>
            <ul>
                <li>PNEUMONIA: 1635</li>
                <li>NORMAL: 785</li>
            </ul>

            <h2>Model Performance (on test set):</h2>
            <ul>
                <li><strong>Loss:</strong> 0.1677</li>
                <li><strong>Accuracy:</strong> 94.14%</li>
            </ul>
            
            <p>The model achieved good performance, with 95% precision and recall for detecting Pneumonia and 89% precision and 90% recall for detecting Normal cases on the test set.</p>

            <hr>

            <form action="/predict/" method="post" enctype="multipart/form-data" class="upload-form">
                <h2>Upload an X-ray image:</h2>
                <input type="file" name="file" accept="image/*">
                <br><br>
                <input type="submit" value="Upload Image" class="button">
            </form>
        </div>
    </body>
    </html>
    """

@app.post("/predict/", response_class=HTMLResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Receives an uploaded image, processes it, and returns a prediction.
    """
    if model is None:
        return HTMLResponse(content="<h1>Error: Model not loaded. Please check the server logs.</h1>", status_code=500)

    try:
        # Read the image file
        contents = await file.read()
        # Open the image using Pillow and convert to grayscale
        pil_image = Image.open(io.BytesIO(contents)).convert('L')
        # Convert PIL image to NumPy array
        img_arr = np.array(pil_image)

        # Resize the image to 150x150
        resized_arr = cv2.resize(img_arr, (img_size, img_size))

        # Reshape for the model (add batch dimension and channel dimension)
        # The model expects (batch_size, img_height, img_width, channels)
        input_image = resized_arr.reshape(-1, img_size, img_size, 1)

        # Normalize pixel values
        input_image = input_image / 255.0

        # Make prediction
        prediction = model.predict(input_image)[0][0]

        result_text = ""
        if prediction <= 0.5: # Assuming 0.5 is the threshold for binary classification
            result_text = "PNEUMONIA NOT PRESENT"
        else:
            result_text = "PNEUMONIA PRESENT"

        return f"""
        <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 800px; margin: auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
                h1 {{ color: #333; }}
                .result {{ margin-top: 20px; padding: 15px; border: 1px solid #eee; border-radius: 5px; background-color: #f9f9f9; font-size: 1.2em; font-weight: bold; }}
                .back-button {{ margin-top: 20px; padding: 10px 15px; background-color: #007bff; color: white; text-decoration: none; border-radius: 4px; }}
                .back-button:hover {{ background-color: #0056b3; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Prediction Result</h1>
                <div class="result">
                    <p>The model predicts: <strong>{result_text}</strong></p>
                    <p>Confidence Score: {prediction:.4f}</p>
                </div>
                <a href="/" class="back-button">Upload another image</a>
            </div>
        </body>
        </html>
        """

    except Exception as e:
        return HTMLResponse(content=f"<h1>Error processing image: {e}</h1>", status_code=500)
