# app.py
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Set the page title and layout
st.set_page_config(page_title="Rice Leaf Disease Detection", layout="centered")
st.title("🌾 Rice Leaf Disease Detection using InceptionV3")
st.markdown("Upload an image of a rice leaf to detect whether it's healthy or has a disease.")

# Load the trained model
MODEL_PATH = "Model_inceptionv3_SGD_Epochs_30_(TRYSplit_dataset).keras"
model = load_model(MODEL_PATH)

# Define your class labels (order must match training order!)
class_names = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']  # 🔁 Replace with actual folder names from training if different

# File uploader UI
uploaded_file = st.file_uploader("Choose a rice leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize like in your training
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    confidence = prediction[0][predicted_index]

    # Show result
    st.markdown(f"### 🧪 Prediction: **{predicted_label}**")
    st.markdown(f"#### 🔍 Confidence: `{confidence * 100:.2f}%`")
