import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog
import json
from PIL import Image
import tempfile
import os

# Load trained SVM model
model_filename = "tulu_ocr_model.pkl"
svm_model = joblib.load(model_filename)

# Load Tulu-to-Kannada mapping JSON file
mapping_filename = "tulu_to_kannada_mapping.json"
with open(mapping_filename, "r", encoding="utf-8") as f:
    character_mapping = json.load(f)

# Feature extraction function
def extract_features(image_path, img_size=(64, 64)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        st.error("Error: Could not load the image.")
        return None
    img = cv2.resize(img, img_size)

    # Extract HOG features
    features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys')
    
    return np.array(features).reshape(1, -1)

# Prediction function
def predict_character(image_path):
    features = extract_features(image_path)
    if features is None:
        return "Error in feature extraction"

    prediction_index = svm_model.predict(features)[0]  # Get predicted index
    
    # Debugging steps
    st.markdown(f"<div style='color:coral; font-size:24px; font-weight:bold;'>PREDICTED KANNADA CHARACTER : {prediction_index}</div>",unsafe_allow_html=True)
    # st.write(f"Available JSON Keys: {list(character_mapping.keys())[:10]} ...")  # Show first 10 keys for debugging

    # Ensure index is string before lookup
    predicted_character = character_mapping.get(str(prediction_index), "Unknown")

    return predicted_character

# Streamlit UI
st.markdown("<h1 style='text-align:center; color:teal;'>ECHOES OF CONJUNCT LETTERS - TULU TO KANNADA TRANSLATION</h1>",unsafe_allow_html=True)
# st.write("Upload an image of a Tulu character to get its Kannada translation.")

# File uploader
uploaded_file = st.file_uploader("Upload an image of a Tulu character to get its Kannada Translation", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_path = temp_file.name
        image.save(temp_path)

    # Run prediction
    predicted_character = predict_character(temp_path)

    
    # Cleanup temporary file
    os.remove(temp_path)