import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
# Ensure the model file 'asl_mnist_model.h5' is in the same directory or provide a path
MODEL_PATH = 'hs1.h5'

# Set up Streamlit app
st.title("ASL to Text Converter")
st.markdown("This app recognizes American Sign Language (ASL) digits (0-9) using a trained MNIST model.")

# Load the model
@st.cache_resource
def load_asl_model():
    return load_model(MODEL_PATH)

model = load_asl_model()

# Define helper function to preprocess image
def preprocess_image(image):
    """Preprocess the image for prediction."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized = cv2.resize(gray, (28, 28))           # Resize to 28x28
    normalized = resized / 255.0                  # Normalize pixel values
    reshaped = normalized.reshape(1, 28, 28, 1)  # Add batch and channel dimensions
    return reshaped

# Initialize camera capture
capture_video = st.checkbox("Enable Webcam")
if capture_video:
    st.write("### Webcam Preview")
    run = st.checkbox("Start Recognition")

    # Access webcam using OpenCV
    camera = cv2.VideoCapture(0)
    
    if run:
        placeholder = st.empty()
        while run:
            ret, frame = camera.read()
            if not ret:
                st.warning("Unable to access the camera.")
                break

            # Display live camera feed
            frame = cv2.flip(frame, 1)
            placeholder.image(frame, channels="BGR")

            # Preprocess and predict
            preprocessed_frame = preprocess_image(frame)
            prediction = model.predict(preprocessed_frame)
            predicted_class = np.argmax(prediction, axis=1)[0]

            st.markdown(f"### Predicted Digit: `{predicted_class}`")

    camera.release()
    cv2.destroyAllWindows()

# Upload an image for recognition
uploaded_file = st.file_uploader("Upload an image of a hand sign (for digits 0-9):", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Convert the uploaded image into OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    st.markdown(f"### Predicted Digit: `{predicted_class}`")

st.write("---")
st.markdown("Created using Streamlit, OpenCV, and TensorFlow.")
