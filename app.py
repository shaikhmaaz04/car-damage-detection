# app.py

import streamlit as st
from model_helper import predict
import os 
import uuid # Used to create unique filenames

st.title("Car Damage Detection")
st.markdown("Upload an image of a car to classify the type of damage.")

uploaded_file = st.file_uploader("Choose an image of a vehicle", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Ensure the temporary directory exists
    temp_dir = "tmp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create a unique filename to prevent clashes
    unique_filename = str(uuid.uuid4()) + "_" + uploaded_file.name
    image_path = os.path.join(temp_dir, unique_filename)
    
    # 2. Save the uploaded file and run prediction
    try:
        # Save the file temporarily
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.image(image_path, caption='Uploaded Image', use_container_width=True)
        
        # 3. Get the prediction
        with st.spinner('Analyzing image and predicting damage...'):
            prediction = predict(image_path)
        
        st.success(f"Prediction: **{prediction}**")

    except Exception as e:
        # Handle errors during saving or prediction
        st.error(f"An error occurred during prediction: {e}")
        
    finally:
        # 4. Clean up the temporary file using os.remove()
        if os.path.exists(image_path):
            os.remove(image_path)
            # print(f"Cleaned up {image_path}")