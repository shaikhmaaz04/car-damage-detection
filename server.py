# server.py

from fastapi import FastAPI, File, UploadFile
from model_helper import predict
import os  # Import the correct module for file removal

app = FastAPI()

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        # Use os.path.join for better path handling
        image_path = os.path.join("tmp", f"temp_{file.filename}") 
        
        # Save the file temporarily
        with open(image_path, "wb") as f:
            f.write(image_data)
            
        # Get the prediction
        prediction = predict(image_path)

        # Correct file cleanup using os.remove()
        os.remove(image_path) 

        return {"prediction": prediction}

    except Exception as e:
        # It's good practice to try and remove the file even if prediction fails
        if 'image_path' in locals() and os.path.exists(image_path):
            os.remove(image_path)
            
        return {"error": str(e)}