# streamlit run predict.py command to run code

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model

def load_trained_model():
    model = load_model('potato_disease.h5')  # Ensure this path is correct
    return model

model = load_trained_model()

# Image preprocessing function
IMG_SIZE = (256, 256)  # Same size used during training

def preprocess_image(image):
    img = image.resize(IMG_SIZE)  # Resize image
    img_array = img_to_array(img)  # Convert image to array
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit App UI
st.markdown("<h1 style='text-align: center;'>🍀⚕️ Medicinal Plant Health Checker</h1>", unsafe_allow_html=True)

st.title("🍃 Potato Disease Prediction")
st.subheader("Classify images into **Early**, **Healthy**, or **Late** stages.")

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Predict button
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            # Preprocess and predict
            image = load_img(uploaded_file)
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            
            # Class labels
            class_labels = ['Early', 'Healthy', 'Late']
            
            # Get predicted class
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_label = class_labels[predicted_class_index]
            #confidence = np.max(prediction) * 100  # Confidence score
            
            # Display prediction
            st.success(f"🌿 Predicted Class: **{predicted_label}** ")
