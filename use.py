import tensorflow as tf
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import uvicorn

# Constants
MODEL_PATH = "plant_disease_model"  # Path to your saved model
IMG_SIZE = 456  # Image size expected by the model

# Load the trained model
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Preprocess the image: Apply convolution + Adaptive Resizing
def preprocess_image(image: Image.Image):
    """
    Preprocess an input image by applying convolution and resizing it adaptively to 456x456.

    Args:
        image (PIL.Image): Input image.

    Returns:
        np.ndarray: Preprocessed image ready for model prediction.
    """

    # Convert PIL image to NumPy array (RGB format)
    img = np.array(image)

    # Convert RGB to BGR (OpenCV format)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Apply a convolution using Gaussian Blur to retain important features
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Get original dimensions
    height, width, _ = img.shape

    # Resize while maintaining aspect ratio (Adaptive Downscaling)
    if height > IMG_SIZE or width > IMG_SIZE:
        scaling_factor = IMG_SIZE / max(height, width)  # Compute scale factor
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Pad image to maintain 456x456 size (Center Padding)
    padded_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    pad_top = (IMG_SIZE - img.shape[0]) // 2
    pad_left = (IMG_SIZE - img.shape[1]) // 2
    padded_img[pad_top:pad_top+img.shape[0], pad_left:pad_left+img.shape[1], :] = img

    # Convert back to RGB
    padded_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)

    # Normalize (Convert to float32 and scale to [0,1])
    padded_img = padded_img.astype(np.float32) / 255.0

    # Expand dimensions to match model input shape
    padded_img = np.expand_dims(padded_img, axis=0)

    return padded_img

# Predict the class of the image
def predict(model, image_array):
    """
    Perform prediction on a preprocessed image using the loaded model.

    Args:
        model (tf.keras.Model): Loaded TensorFlow model.
        image_array (np.ndarray): Preprocessed image array.

    Returns:
        dict: Predicted class and confidence score.
    """
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))

    return {
        "predicted_class": int(predicted_class),
        "confidence": confidence
    }

# Initialize FastAPI app
app = FastAPI()

# Load model at startup
@app.on_event("startup")
async def startup_event():
    global model
    model = load_model()

# API endpoint for image prediction
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read image file
        image_data = await file.read()
        # Open image
        image = Image.open(BytesIO(image_data)).convert("RGB")
        # Preprocess image
        image_array = preprocess_image(image)
        # Perform prediction
        prediction = predict(model, image_array)
        # Return prediction
        return JSONResponse(content=prediction)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
